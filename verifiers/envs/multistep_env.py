import random
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Sequence, Any, Tuple

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.environment import Environment
from ..imports import LLM, SamplingParams  # type: ignore


class MultiStepEnv(Environment):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 10,
                 max_steps: int = 10,
                 sleep_time: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)
        # mask_env_response: 是否屏蔽环境响应（CoT 中可能隐藏中间反馈）
        # （0 表示屏蔽，1 表示保留）
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        # 每次请求间的休眠时间，避免过载
        self.sleep_time = sleep_time
        # max_steps: 最大推理步骤，防止无限循环
        self.max_steps = max_steps

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    # 返回奖励函数列表，用于评估生成结果（CoT 的每一步可能需要评分）
    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    # 判断当前对话是否完成（例如，问题已解决）
    @abstractmethod
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        pass

    # 环境对模型输出的响应（如提供提示或验证答案）
    @abstractmethod
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        pass

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:

        # 筛选未完成状态: 找出未完成（completed=False）的对话
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        # 获取未完成状态的对话历史
        messages_to_step = [states[i]["messages"] for i in live_indices]
        # 使用 llm.chat 生成下一步推理
        # llm_response 的结构是一个包含输入和输出信息的对象
        # prompt_token_ids：表示输入的 token
        # outputs[0].text：生成的新文本
        # outputs[0].token_ids：生成文本的 token ID
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False)  # type: ignore

        # for i, j in enumerate(live_indices):
        def update_state(j, llm_response):
            # sleep for 0-1 seconds to avoid rate limiting
            time.sleep(self.sleep_time * random.random())

            # 创建 states[j] 的副本，避免修改原始数据
            state = states[j].copy()
            if len(state["prompt_ids"]) == 0:
                # 如果 prompt_ids 为空，则用当前prompt的 token ID 初始化
                state["prompt_ids"] = llm_response.prompt_token_ids
            # 操作: 将模型生成的响应添加到对话历史（包含prompt,环境响应,模型生成）
            # state["messages"]初始为：prompt(dict)，当前操作又追加模型生成，所以此时既包含prompt又包含模型生成
            # prompts: List[List[Dict[str, Any]]] 的设计：
            # 外层 List：支持批量处理多个任务
            # 内层 (states；dict chain)List[Dict[str, Any]]：表示每个任务的对话历史，支持 CoT 的多步推理
            # 即每个state为一个dict
            state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text})

            # 输入 + 输出
            # total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            # 输入
            total_prev_len = len(state["prompt_ids"])
            # 环境响应部分的 token 数
            env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len  # type: ignore
            # 新生成内容的 token 数
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # completion_mask 用于标记 completion_ids 中哪些 token 是环境响应（通常标记为 0），哪些是模型生成的内容（通常标记为 1）
            # （0 表示屏蔽，1 表示保留）
            # 标记环境响应
            state["completion_mask"].extend([self.env_mask] * env_response_len)
            # 标记新生成内容
            state["completion_mask"].extend([1] * new_completion_len)

            # prompt_token_ids = prompt + 环境响应（如果有）
            # 合并 prompt_token_ids 和模型生成 token
            state["completion_ids"] = list(llm_response.prompt_token_ids)  # type: ignore
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            # 去掉prompt部分
            # 剩余：环境响应（如果有）+ 模型生成 token
            state["completion_ids"] = state["completion_ids"][len(state["prompt_ids"]):]
            # 同步截断 completion_mask；从后往前截取长度len(state["completion_ids"])
            # list[-n:] 表示取列表的最后 n 个元素
            state["completion_mask"] = state["completion_mask"][-(len(state["completion_ids"])):]

            # 对话完成(问题已解决)或截断
            if self.is_completed(state["messages"]) or len(
                    state["completion_ids"]) > sampling_params.max_tokens:  # type: ignore
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])]
            else:
                # 追加到对话历史中
                state["messages"].append(self.env_response(state["messages"]))

            # check 异常
            if not len(state["completion_mask"]) == len(state["completion_ids"]):
                print(state["messages"])
                print(state["completion_mask"])
                print(state["completion_ids"])
                raise ValueError(f"Completion mask and completion ids are not the same length for state {j}")

            return j, state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # executor.map: 线程池的 map 方法，类似于 Python 的内置 map 函数，但会并行执行
            # *args 表示将元组或列表解包，例如 (j, llm_response) 变成 j 和 llm_response
            # 多线程并行执行未完成状态的状态更新
            # update_state(j, llm_response)
            results = list(executor.map(
                lambda args: update_state(*args),
                # 生成任务参数列表，作为 executor.map 的第二个参数
                # i 是 live_indices 中的位置（从 0 开始）
                # j 是未完成状态在 states 中的原始索引
                # llm_responses[i]: 对应状态的语言模型响应
                [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
            ))

        # 将结果写回 states
        for j, state in results:
            states[j] = state

        return states

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False

        states = [{
            # messages（对话历史）
            # 初始为输入prompt
            "messages": m,
            # 输入prompt的长度
            "prompt_messages": len(m),
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": []
        } for m in prompts]

        # main loop
        # 效果与递归调用LLM.chat()一样
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        # completion_messages：环境响应（如果有）+ 模型生成 token
        # 使output中的三部分内容保持一致
        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask
        }
        return output

    def step_api(self,
                 client: Any,
                 model: str,
                 messages: List[Dict[str, str]],
                 **kwargs: Any) -> Tuple[List[Dict[str, str]], bool]:
        """
        Execute a single step using OpenAI API, including environment response if needed.
        
        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name to use
            **kwargs: Additional arguments for the chat completion API
        
        Returns:
            Updated messages list with assistant response and possibly environment response
        """
        messages_copy = messages.copy()

        try:
            # Get assistant response
            response = client.chat.completions.create(
                model=model,
                messages=messages_copy,
            )

            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant",
                "content": response.choices[0].message.content
            }
            messages_copy.append(assistant_msg)

            # Check if we're done
            if self.is_completed(messages_copy):
                # rollout: step
                rollout_is_completed = True
            else:
                rollout_is_completed = False
                # If not done, get and add environment response
                env_msg = self.env_response(messages_copy)
                messages_copy.append(env_msg)

            return messages_copy, rollout_is_completed

        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error in API call: {str(e)}"}
            messages_copy.append(error_msg)
            return messages_copy, True

    def eval_api(self,
                 client: Any,
                 model: str,
                 max_concurrent: int = 32,
                 timeout: int = 60,
                 sampling_args: Dict[str, Any] = {},
                 **kwargs: Any):
        """
        Evaluate model using OpenAI API with proper concurrency.
        
        Args:
            client: OpenAI client instance
            model: Model name as string
            max_concurrent: Maximum number of concurrent API calls
            timeout: Maximum seconds to wait for each example
            sampling_args: Arguments specific to sampling (separate from env sampling_args)
            **kwargs: Additional arguments for evaluation
        
        Returns:
            Tuple of (eval_dataset, rewards)
        """

        def run_evaluation():
            # Import libraries here to avoid requiring them for normal operation
            import asyncio
            from asyncio import Semaphore
            # Get the evaluation dataset
            if self.eval_dataset is None:
                self.eval_dataset = self.get_eval_dataset(**kwargs)

            if self.eval_dataset is None:
                raise ValueError("Failed to load evaluation dataset")

            eval_dataset = self.eval_dataset

            async def process_example(example, semaphore):
                async with semaphore:
                    # Initialize conversation with system prompt and few-shot examples
                    prompt = example["prompt"]
                    messages = example["prompt"].copy()
                    answer = example["answer"]

                    # Save the length of initial messages to extract just the interaction part later
                    initial_length = len(messages)

                    # Run the conversation loop until completion or max steps
                    for _ in range(self.max_steps):  # Safety limit on conversation turns
                        try:
                            # Run step_api to get model and environment response
                            # Note: step_api now returns a tuple (messages, is_completed)
                            step_result = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.step_api(
                                    client=client,
                                    model=model,
                                    messages=messages,
                                    **sampling_args
                                )
                            )

                            # Unpack the step_api result
                            messages, is_completed = step_result

                            # If the rollout is completed, break the loop
                            if is_completed:
                                break

                        except Exception as e:
                            print(f"Error processing example {example.get('id', 'unknown')}: {str(e)}")
                            break

                    # Extract only the interaction part (not system/few-shot)
                    completions = messages[initial_length:]

                    return {
                        "prompt": prompt,
                        "completions": completions,
                        "answer": answer
                    }

            async def run_all_examples():
                # Create semaphore for concurrency control
                from tqdm.asyncio import tqdm_asyncio

                semaphore = Semaphore(max_concurrent)

                # Process all examples concurrently
                tasks = [process_example(example, semaphore) for example in eval_dataset]
                results = await tqdm_asyncio.gather(
                    *tasks,
                    total=len(eval_dataset),
                    desc=f"Evaluating {len(eval_dataset)} examples"
                )

                return results

            # Run the async evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()

            # Calculate rewards
            results_prompt = [result["prompt"] for result in results]
            results_answer = [result["answer"] for result in results]
            results_completions = [result["completions"] for result in results]
            results = {"prompt": results_prompt, "answer": results_answer, "completions": results_completions}

            reward_funcs = self.get_rubric()
            rewards = {}

            for reward_func in reward_funcs:
                func_rewards = reward_func(**results)  # type: ignore
                func_reward_avg = sum(func_rewards) / len(func_rewards)
                func_name = reward_func.__name__  # type: ignore
                print(f"{func_name}: {func_reward_avg}")
                rewards[func_name] = func_reward_avg

            return rewards

        # Run the evaluation function
        return run_evaluation()
