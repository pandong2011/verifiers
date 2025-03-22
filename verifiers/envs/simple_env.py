import random
from typing import List, Dict, Sequence, Any

from datasets import Dataset

from verifiers.envs.environment import Environment
from ..imports import LLM, SamplingParams  # type: ignore


class SimpleEnv(Environment):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            # 含义：指定生成多少个候选序列（即采样次数）
            # 1：表示只生成一个输出，而不是多个候选答案
            "n": 1
        }
        # 作用：用外部传入的 sampling_args（例如 {"temperature": 0.7, "max_tokens": 100}）更新默认的 self.sampling_args
        self.sampling_args.update(sampling_args)

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def format_prompt(self, prompt: str, fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.few_shot and random.random() < fewshot_prob:
            messages.extend(self.few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    # prompts: List[List[Dict[str, Any]]] 的设计：
    # 外层 List：支持批量处理多个任务
    # 内层 List[Dict[str, Any]]：表示每个任务的对话历史，支持 CoT 的多步推理
    # 字典结构：适配对话模型的标准输入格式，保留灵活性
    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:

        custom_sp = sampling_params.clone()
        # 设置sampling_args
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)
        states = [{
            "messages": m,
            "prompt_ids": [],
            "completion_ids": [],
            "completion_mask": []
        } for m in prompts]

        # get completions
        completions = llm.chat(prompts, sampling_params=custom_sp, use_tqdm=False)  # type: ignore
        # 单一步更新state
        # state move
        for i, completion in enumerate(completions):
            states[i]["messages"].append({"role": "assistant", "content": completion.outputs[0].text})
            states[i]["prompt_ids"] = list(completion.prompt_token_ids)  # type: ignore
            states[i]["completion_ids"] = list(completion.outputs[0].token_ids)
            # 初始化为空列表，后续填充为与 completion_ids 长度相同的 [1, 1, ...]，表示每个生成的 token 是否有效。
            # 在 CoT 中，掩码可能用于标记哪些 token 是推理过程中的关键部分，或者用于过滤无关内容（虽然这里只是简单地全设为 1）
            states[i]["completion_mask"] = [1] * len(states[i]["completion_ids"])

        output = {
            "ids": [states[i]["completion_ids"] for i in range(len(states))],
            # [-1:] 的作用：
            # 提取 states[i]["messages"] 的最后一个元素（即模型的最新回答）
            # 保持列表形式，以符合 "messages": List[List[Dict[str, Any]]] 的类型定义
            "messages": [states[i]["messages"][-1:] for i in range(len(states))],
            "mask": [states[i]["completion_mask"] for i in range(len(states))]
        }
        return output
