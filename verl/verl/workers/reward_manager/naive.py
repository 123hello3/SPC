
import math
from collections import defaultdict
from typing import Any, List

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# from verl.utils.reward_score.self_gsm8k import _extract_answer, _normalize_and_compare
from verl.utils.reward_score.self_gsm8k1 import _extract_answer, _normalize_and_compare
import re


def extract_think_content(solution_str: str) -> str | None:
   
    if not isinstance(solution_str, str):
        return None

 
    solution_str = solution_str.strip()

   
    match = re.match(r'(?P<think>.*?)\\boxed\{(?P<answer>.*?)\}', solution_str, re.DOTALL)

    if not match:
        return None

    think = match.group('think')
    answer = match.group('answer')
   
    if not answer.strip():
        return None

    return think

import math

def _percentile(lst: List[float], p: float) -> float:
   
    if not lst:
        return float('inf')
    sorted_lst = sorted(lst)
    idx = int(len(sorted_lst) * p / 100.0)
    return sorted_lst[min(idx, len(sorted_lst) - 1)]

@register("naive")
class NaiveRewardManager(AbstractRewardManager):
 

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        ema_decay=0.95,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.ema_decay = ema_decay

     
        self._ema_acc = 0.0                     
        self._best_compression_len = float('inf')  
        self._plateau_steps = 0                 

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        is_correct_list = []
        token_lengths = []
        response_lengths = []
        valid_sample_mask = []

       
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_attention = data_item.batch["attention_mask"][prompt_length:]
            valid_response_length = response_attention.sum().item()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]

            prompt_str = self.tokenizer.decode(prompt_ids[-prompt_length:], skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

           
            print(f'debug:think_content="{think_content}"')
            if think_content:
                think_ids = self.tokenizer.encode(think_content, add_special_tokens=False)
                think_token_len = len(think_ids)
            else:
                think_token_len = 0

            pred_answer = _extract_answer(response_str)
            is_correct = _normalize_and_compare(pred_answer, ground_truth)

            is_valid = valid_response_length > 0
            valid_sample_mask.append(is_valid)
            is_correct_list.append(is_correct and is_valid)
            token_lengths.append(think_token_len if is_valid else 0)
            response_lengths.append(valid_response_length)

           
            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            if is_valid:
                reward_tensor[i, valid_response_length - 1] = reward

            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        
        total_valid = sum(valid_sample_mask)
        current_batch_acc = 0.0
        if total_valid > 0:
            correct_count = sum(is_correct_list)
            current_batch_acc = correct_count / total_valid

       
        self._ema_acc = self.ema_decay * self._ema_acc + (1 - self.ema_decay) * current_batch_acc

        
        correct_lengths = [
            token_lengths[i]
            for i in range(len(data))
            if valid_sample_mask[i] and is_correct_list[i] and token_lengths[i] > 0
        ]
        print(f'debug:correct_lengths={correct_lengths}')
      
        IMPROVEMENT_RATIO = 0.98   

        if correct_lengths:
            current_q25 = _percentile(correct_lengths, 25)
            if current_q25 < self._best_compression_len * IMPROVEMENT_RATIO:
                self._best_compression_len = current_q25
                self._plateau_steps = 0
            else:
                self._plateau_steps += 1
        else:
           
            self._plateau_steps += 1

       
        acc_stable = (self._ema_acc >= ACC_STABLE_THRESHOLD)
        length_converged = (self._plateau_steps >= PLATEAU_PATIENCE)
        should_apply_len_reward = not (acc_stable or length_converged)
        print(f'debug:should_apply_len_reward={should_apply_len_reward}, ema_acc={self._ema_acc}, plateau_steps={self._plateau_steps}, best_compression_len={self._best_compression_len}')
        
        
        if should_apply_len_reward and self._best_compression_len < float('inf'):
            print('debug:applying length reward...')
            target_len = self._best_compression_len  

            for i in range(len(data)):
                if not (valid_sample_mask[i] and is_correct_list[i]):
                    continue

                think_len = token_lengths[i]
                resp_len = response_lengths[i]

                if think_len > target_len:
                    
                    relative_excess = (think_len - target_len) / max(target_len, 1.0)
                    bonus = -0.2 * min(relative_excess, 1.0)   
                   
                    reward_tensor[i, resp_len - 1] += bonus
                    
                

        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info.update({
                "ema_acc": [self._ema_acc] * len(data),
                "best_compression_len": [self._best_compression_len] * len(data),
                "plateau_steps": [self._plateau_steps] * len(data),
            })
            # print(f'debug:{reward_tensor}')
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        else:
            return reward_tensor
