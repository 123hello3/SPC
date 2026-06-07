# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# 
_CORRECTNESS_REWARD =1
# 0.7
import re
def _extract_answer(solution_str: str) -> str:
    matches = re.findall(r'\\boxed\{([^}]*)\}', solution_str)
    return matches[-1].strip() if matches else ""


def _normalize_and_compare(pred: str, target: str) -> bool:
    
    def clean(s):
        
        s = re.sub(r'[^0-9\-+.]', '', s.split('\n')[0].strip())
        return s

    pred_clean = clean(pred)
    target_clean = clean(target)

    if not pred_clean or not target_clean:
        return False

    try:
        return int(float(pred_clean)) == int(float(target_clean))
    except (ValueError, OverflowError):
        return pred_clean == target_clean


def _compute_correctness_reward(solution_str: str, ground_truth: str) -> float:
    
    pred_answer = _extract_answer(solution_str)
    if _normalize_and_compare(pred_answer, ground_truth):
        return _CORRECTNESS_REWARD
    return 0.0
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
import re

_FORMAT_REWARD = 0.2


def _compute_format_reward(solution_str: str) -> float:
    if not isinstance(solution_str, str):
        return 0.0

   
    boxed_occurrences = re.findall(r'\\boxed\{', solution_str)
    if len(boxed_occurrences) != 1:
        return 0.0

    stripped = solution_str.rstrip()

    match = re.fullmatch(r'(?P<think>.*?)\\boxed\{(?P<answer>.*?)\}', stripped, re.DOTALL)

    if not match:
        return 0.0

    answer = match.group('answer')
    if not answer.strip():
        return 0.0
    return _FORMAT_REWARD
   
_CONSISTENCY_REWARD =0.2
import re


def _extract_predicted_difficulty(solution_str: str) -> str | None:
    
    match = re.search(r'\[(easy|hard)\]\s*(.*)', solution_str, re.DOTALL)
    if match:
        difficulty = match.group(1)     
        
        return difficulty
    else:
        return None


def _compute_consistency_reward(solution_str: str, ground_truth: str) -> float:
   
    predicted_difficulty = _extract_predicted_difficulty(solution_str)

    pred_answer = _extract_answer(solution_str)
    

    is_correct = _normalize_and_compare(pred_answer, ground_truth)

    if predicted_difficulty == 'easy' and is_correct:
        return _CONSISTENCY_REWARD
    else:
        return 0.0
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def compute_total_reward(solution_str: str, ground_truth: str,extra_info: dict) -> float:
   
    format_score = _compute_format_reward(solution_str)
    correctness_score = _compute_correctness_reward(solution_str, ground_truth)
    consistency_score = _compute_consistency_reward(solution_str, ground_truth)
    print(f'debug: format_score={format_score}, correctness_score={correctness_score}, consistency_score={consistency_score}')
    return correctness_score +  +format_score + consistency_score  



