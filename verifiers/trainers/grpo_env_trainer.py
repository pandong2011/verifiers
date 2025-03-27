from typing import Callable, Optional, Union, Any

import torch
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig  # type: ignore

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        if not args.use_vllm:  # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))):
            raise ValueError(
                "reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env

    # _generate_and_score_completions 函数实现了 GRPO 的生成和奖励计算部分， “Policy Model → Reward Model → Group Computation”。
    # compute_loss 函数实现了 GRPO 的损失计算部分，对应图中的 KL 正则化和策略更新。

    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]  # type: ignore
        # maybe_apply_chat_template 函数根据 self.processing_class（通常是分词器）将提示转换为适合模型的格式（例如对话格式）
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in
                        inputs]  # type: ignore
        # 将文本提示 prompts_text 编码为张量格式，准备输入模型
        # self.processing_class 是一个分词器（如 transformers 库中的 Tokenizer）
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            # type: ignore
        )  # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)  # type: ignore
        # 提取prompt的 token ID 和注意力掩码，为模型生成或训练做准备
        # attention_mask 是一个张量，用于指示模型在处理输入（prompt）时哪些 token 需要关注，哪些是填充（padding）部分
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            # 截取右边，因为填充左边
            # batch_size维度不管
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        # 收集 prompt，主进程生成补全（ID、消息、掩码），非主进程初始化为空结果，为分布式生成做准备
        # gather_object(prompts) 确保所有进程的 prompt 按固定顺序（通常是进程索引顺序）合并
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_prompts,
                llm=self.llm,
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']

        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        # 将主进程生成的补全结果广播到所有进程
        # 确保分布式训练中所有进程都能访问主进程生成的补全结果，以便后续计算
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)

        # 计算当前进程在分布式环境中的数据切片范围
        # 用于分布式训练中，将全局数据（如补全结果）分配给当前进程，确保每个进程处理自己的数据子集
        # 类似于一个数据分配的滚动窗口，窗口长度为len(prompts)；process_index从零开始
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        # 根据当前进程的切片范围，提取对应的补全 ID、消息和掩码，确保分布式环境中各进程处理自己的数据
        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]

        # Pad + mask after per-sequence EOS tokens
        # prompt 的填充 token 添加在序列左侧，实际内容在右侧
        # 补全的实际内容在左侧，填充 token 在右侧
        # prompt_ids: [pad, pad, t1, t2, t3]
        # completion_ids: [c1, c2, c3, pad]
        # 拼接后：[pad, pad, t1, t2, t3, c1, c2, c3, pad]
        # attention_mask：[0, 0, 1, 1, 1, 1, 1, 1, 0]
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)  # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # 取 completion_ids 的序列长度（第二个维度），赋值给 logits_to_keep
        # completion_ids 是一个张量，形状为 (batch_size, completion_length)，表示补全的 token ID
        # logits_to_keep 表示在后续计算中，只需要为补全部分的 token 计算 logits（对数概率），忽略 prompt 部分
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            # logps: log probabilities
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            # 如果 self.beta == 0.0（无 KL 正则化），ref_per_token_logps 设为 None
            if self.beta == 0.0:
                ref_per_token_logps = None
            # 如果存在参考模型（self.ref_model），用 self.ref_model 计算对数概率
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            # 否则，使用当前模型（self.model）但禁用适配器（disable_adapter），模拟参考模型行为，计算对数概率
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            # 从 inputs 中提取除 prompt 和 completion 外的其他键（如元数据）
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]  # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}  # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)  # type: ignore
            # 将奖励值转换为张量，存储在 rewards_per_func 的第 i 列
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        # 多个奖励加权求和；得到每个prompt的总奖励
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        # one_group: num_generations
        # reshape 后按组计算mean and std
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)  # type: ignore
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)  # type: ignore

        # Normalize the rewards to compute the advantages
        # 将组内均值和标准差重复 self.num_generations 次(reshape时相当于除以num_generations)，扩展到与 rewards 长度一致，便于后续优势计算
        # 确保每组补全使用相同的均值和标准差进行归一化
        # interleave: 交织
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)  # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)  # type: ignore
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        # 按索引给gpu分发优势数据
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        # completion_mask.sum(1)
        # 沿序列维度（dim=1）求和，计算每个补全的实际 token 数量（1 的数量）
        # gather_for_metrics：收集所有进程的数据
        # completion_length：补全的平均长度
        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)).float().mean().item()  # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        # 对 rewards_per_func 沿第 0 维（样本维度）求均值，得到每个奖励函数的平均奖励
        # rewards_per_func shape：(num_samples, num_reward_funcs)
        reward_per_func = rewards_per_func.mean(0)  # type: ignore
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__  # type: ignore
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # log: 日志
        # log_completions: 打印补全
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    # 如果 rich 库可用，打印第一个 prompt、补全和奖励的样本，附带当前步数
                    # prompts_to_log[0][-1]["content"] 是“第一个 prompt 的最后一个消息内容”，通常是用户的最新输入
                    # 模型的回复在 completions_to_log 中
                    print_prompt_completions_sample(
                        [str(prompts_to_log[0][-1]["content"])],
                        [completions_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:  # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})  # type: ignore

        # num_iterations > 1 时计算
        # old_per_token_logps：反映旧策略的对数概率，用于策略比率计算
        # beta != 0 时计算
        # ref_per_token_logps：反映参考策略的对数概率，用于 KL 正则化
        # 两者共同确保 GRPO 训练稳定：old 控制更新幅度，ref 防止偏离初始分布
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
