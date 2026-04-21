from typing import List, Union
import re

def gpqa_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False, num_tokens = -1, valid_response_length = -1):
    def get_model_choice(res):
        for i in range(len(res)-1, -1, -1):
            if res[i] == 'A' or res[i] == 'B' or res[i] == 'C' or res[i] == 'D':
                # Check if res[i-1] is not a character
                if not (res[i-1] >= 'a' and res[i-1] <= 'z') and not (res[i-1] >= 'A' and res[i-1] <= 'Z'):
                    return res[i]
                    break
        return ''
    model_choice = get_model_choice(solution_str)
    if model_choice == ground_truth:
        return 1.0
    else:
        return 0.0