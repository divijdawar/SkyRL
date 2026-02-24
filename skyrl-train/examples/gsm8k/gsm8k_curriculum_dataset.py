"""
Generate a GSM8K parquet with a 'difficulty' column for testing curriculum sampling.

Difficulty = number of <<...>> calculation steps in the ground-truth answer (1-8).

Usage:
    cd skyrl-train && uv run python examples/gsm8k/gsm8k_curriculum_dataset.py
"""

import os
import re

import datasets


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    return solution.group(0).split("#### ")[1].replace(",", "")


if __name__ == "__main__":
    output_dir = os.path.expanduser("~/data/gsm8k_curriculum")
    ds = datasets.load_dataset("openai/gsm8k", "main")

    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def process(example, idx):
        question_raw = example.pop("question")
        answer_raw = example.pop("answer")
        return {
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": question_raw + " " + instruction}],
            "env_class": "gsm8k",
            "reward_spec": {"method": "rule", "ground_truth": extract_solution(answer_raw)},
            "extra_info": {"split": "train", "index": idx, "answer": answer_raw, "question": question_raw},
            "difficulty": float(len(re.findall(r"<<.*?>>", answer_raw))),
        }

    os.makedirs(output_dir, exist_ok=True)
    for split, name in [("train", "train"), ("test", "validation")]:
        mapped = ds[split].map(process, with_indices=True)
        mapped.to_parquet(os.path.join(output_dir, f"{name}.parquet"))

    print(f"Saved to {output_dir} with difficulty column added.")
