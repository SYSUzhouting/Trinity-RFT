"""Base Workflow Class"""

from __future__ import annotations

from typing import List, Optional

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.math_reward import MathRewardFn
from trinity.common.rewards.reward_fn import RewardFn
from trinity.common.workflows.workflow import WORKFLOWS, Workflow, Task



@WORKFLOWS.register_module("dialogue_api_workflow")
class DialogueApiWorkflow(Workflow):
    """A workflow for simple single-round task."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self.reset(task)
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

        if len(auxiliary_models)>0:
            self.auxiliary_models = auxiliary_models[0]

        self.outier_detection_type = 'reward_hidden'

    @property
    def resettable(self):
        return True

    def reset(self, task: Task):
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix
        self.reward_fn_args = task.reward_fn_args

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        reward_fn = task.reward_fn


        if reward_fn is None:
            reward_fn = MathRewardFn

        
        if isinstance(reward_fn, type) and issubclass(reward_fn, RewardFn):
            self.reward_fn: RewardFn = reward_fn(**self.reward_fn_args)
        else:
            raise ValueError("`reward_fn` must be a subclass of `RewardFn`")

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def format_messages(self):
        
        messages = self.task_desc
        
        return messages

    def run(self) -> List[Experience]:
        messages = self.format_messages()
        self.logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)

        assert (
            self.auxiliary_models is not None
        ), "Current implementation of RULER requires that auxiliary_models is not None."
        

        for response in responses:
            reward, hidden_state = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                prompt=messages,
                truth=self.truth,
            )

            response.metrics.update({'proxy_reward': reward})
            
            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
            if isinstance(reward, dict):
                if response.metrics is None:
                    response.metrics = {}
                response.metrics.update(reward)
                reward = sum(reward.values())
            response.reward = reward

            if self.outier_detection_type == 'reward_hidden':
                response.reward_hidden_state = hidden_state
        
        # Get all original reward values for this group
        # response_injection_rate = 0.05
        # print(f'Injecting noise at rate {response_injection_rate}...')
        # orig_rewards = [float(r.reward) for r in responses]
        # max_r = max(orig_rewards)
        # min_r = min(orig_rewards)

        # # Only inject when rewards are not all equal
        # if max_r != min_r:
        #     for resp in responses:
        #         if random.random() < response_injection_rate:
        #             current_val = float(resp.reward)
                    
        #             # --- Flip logic ---
        #             # If current value is closer to max, set to min; otherwise set to max
        #             if abs(current_val - max_r) < abs(current_val - min_r):
        #                 resp.reward = min_r
        #             else:
        #                 resp.reward = max_r
                    
        #             # (Optional) Sync proxy_reward in metrics to keep consistency
        #             # Note: proxy_reward here reflects the true reward
        #             # if hasattr(resp, 'metrics') and 'proxy_reward' in resp.metrics:
        #             #     resp.metrics['proxy_reward'] = resp.reward

        #             self.logger.debug(f"Noise Injected! New reward: {resp.reward}")
        
        return responses


