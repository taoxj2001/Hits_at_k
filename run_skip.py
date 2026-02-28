import argparse
import json
import math
import time
from collections import Counter
from statistics import mean
from typing import Any, Dict, List, Union

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path


JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Table-2 Hits@k (skip 'uns' / empty generation) with vLLM.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--prompt_path", type=str, required=True)

    p.add_argument("--subset", type=str, default="head", choices=["head", "torso", "tail"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)

    p.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    p.add_argument("--common_substr_len", type=int, default=3)
    p.add_argument("--max_logprobs", type=int, default=1000)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=100)
    p.add_argument("--stop", type=str, default="\n")

    p.add_argument("--tensor_parallel_size", type=int, default=0)
    p.add_argument("--max_model_len", type=int, default=4096)

    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--output_stats", type=str, default="")

    return p.parse_args()


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path: str) -> JsonType:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def perplexity(logprobs: List[float]) -> float:
    if not logprobs:
        return float("inf")
    avg_logprob = sum(logprobs) / len(logprobs)
    return math.exp(-avg_logprob)


def check_string(a: str, b: Union[str, List[str]]) -> bool:
    a_lower = str(a).lower()
    if isinstance(b, list):
        b_lower = [str(x).lower() for x in b]
        return a_lower in b_lower
    return a_lower == str(b).lower()


def have_common_substring_of_length(str1: str, str2: str, length: int) -> bool:
    if length <= 0:
        return False
    if len(str1) < length or len(str2) < length:
        return False
    substrings1 = {str1[i : i + length] for i in range(len(str1) - length + 1)}
    for j in range(len(str2) - length + 1):
        if str2[j : j + length] in substrings1:
            return True
    return False


def normalize_gen_like_original(gen: str) -> str:
    gen = "" if gen is None else str(gen)
    if len(gen) == 0:
        gen = " " + gen
    if len(gen) > 0 and gen[0].isspace():
        gen = gen[1:]
    return gen


def skip_token_from_first_step(logprobs: Dict[str, Dict[str, Any]], rank: int) -> str:
    values = list(logprobs.values())
    if rank >= len(values):
        return ""
    next_rank_token = values[rank].get("decoded_token", "")
    if ("uns" in next_rank_token) or (next_rank_token.strip() == ""):
        return skip_token_from_first_step(logprobs, rank + 1)
    return next_rank_token


def compute_rank(first_step_logprobs: Dict[str, Dict[str, Any]], truth: Union[str, List[str]], common_len: int) -> int:
    value_list = list(first_step_logprobs.values())
    flag = False
    skip_num = 0
    rank = 1
    for v in value_list:
        ans = v.get("decoded_token", "")
        if ans.isspace():
            skip_num += 1
            continue

        if isinstance(truth, list):
            for t in truth:
                flag = have_common_substring_of_length(ans, str(t), length=common_len)
                if flag:
                    break
        else:
            flag = have_common_substring_of_length(ans, str(truth), length=common_len)

        if flag:
            rank = int(v.get("rank", 1000))
            rank = rank - skip_num
            break

    if not flag:
        rank = 1000
    return rank


def ranks_to_hits(ranks: List[int], ks: List[int]) -> Dict[int, float]:
    total = max(1, len(ranks))
    out: Dict[int, float] = {}
    for k in ks:
        out[k] = sum(1 for r in ranks if r <= k) / total
    return out


def main() -> None:
    args = parse_args()

    tp = args.tensor_parallel_size
    if tp <= 0:
        tp = max(1, torch.cuda.device_count())

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp,
        max_logprobs=args.max_logprobs,
        max_model_len=args.max_model_len,
    )
    AutoTokenizer.from_pretrained(args.model_path)

    task_prompt = load_text(args.prompt_path)
    dataset = load_json(args.dataset_path)

    data_list = dataset[args.subset]
    if args.end is None or args.end < 0:
        data_list = data_list[args.start :]
    else:
        data_list = data_list[args.start : args.end]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        max_tokens=args.max_tokens,
        logprobs=args.max_logprobs,
        prompt_logprobs=5,
    )

    details: List[Dict[str, Any]] = []
    perplexities: List[float] = []
    right_perplexities: List[float] = []
    wrong_perplexities: List[float] = []
    unsure_perplexities: List[float] = []

    ranks: List[int] = []
    total = 0
    correct = 0
    unsure_cnt = 0

    t0 = time.time()
    for d in tqdm(data_list):
        total += 1
        question = d[2]
        truth = d[3]

        prompt = task_prompt.format(question) + "\xa0"
        outputs = llm.generate(prompt, sampling_params)
        gen = outputs[0].outputs[0].text
        gen = normalize_gen_like_original(gen)

        first_step = outputs[0].outputs[0].logprobs[0]
        first_step = {k: v.__dict__ for k, v in first_step.items()}

        triggered_skip = False
        if ("uns" in gen) or (gen.strip() == ""):
            triggered_skip = True
            unsure_cnt += 1
            new_prompt = skip_token_from_first_step(first_step, rank=1)
            prompt2 = prompt + new_prompt
            outputs2 = llm.generate(prompt2, sampling_params)
            gen2 = outputs2[0].outputs[0].text
            gen2 = normalize_gen_like_original(gen2)
            gen = new_prompt + gen2
            if len(gen) > 0 and gen[0].isspace():
                gen = gen[1:]
            outputs = outputs2

        logprob_list: List[float] = []
        for step in outputs[0].outputs[0].logprobs:
            step_d = {k: v.__dict__ for k, v in step.items()}
            first_item = next(iter(step_d.items()))
            logprob_list.append(float(first_item[1]["logprob"]))

        ppx = perplexity(logprob_list)
        perplexities.append(ppx)

        if triggered_skip:
            unsure_perplexities.append(ppx)

        rank = compute_rank(first_step, truth=truth, common_len=args.common_substr_len)
        ranks.append(rank)

        if check_string(gen, truth):
            correct += 1
            right_perplexities.append(ppx)
        else:
            wrong_perplexities.append(ppx)

        if args.output_json:
            details.append(
                {
                    "question": question,
                    "truth": truth,
                    "answer": gen,
                    "logprobs": first_step,
                    "rank": rank,
                    "perplexity": ppx,
                    "triggered_skip": triggered_skip,
                }
            )

    t1 = time.time()

    stats = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "prompt_path": args.prompt_path,
        "subset": args.subset,
        "range": [args.start, args.end],
        "k": args.k,
        "hits_at_k": ranks_to_hits(ranks, args.k),
        "accuracy": correct / max(1, total),
        "total": total,
        "correct": correct,
        "unsure_count": unsure_cnt,
        "rank_histogram": dict(Counter(ranks)),
        "mean_perplexity": mean(perplexities) if perplexities else None,
        "mean_right_perplexity": mean(right_perplexities) if right_perplexities else None,
        "mean_wrong_perplexity": mean(wrong_perplexities) if wrong_perplexities else None,
        "mean_unsure_perplexity": mean(unsure_perplexities) if unsure_perplexities else None,
        "runtime_seconds": t1 - t0,
    }

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.output_stats:
        Path(args.output_stats).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
