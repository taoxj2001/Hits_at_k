#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path


def compute_perplexity(token_logprobs: List[float]) -> float:
    if not token_logprobs:
        return float("inf")
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(-avg_logprob)


def normalize_answer(ans: str) -> str:
    ans = ans or ""
    if ans and ans[0].isspace():
        ans = ans.lstrip()
    return ans


def check_exact_match(pred: str, truth: Union[str, List[str]]) -> bool:
    pred_l = (pred or "").strip().lower()
    if isinstance(truth, list):
        return pred_l in {str(x).strip().lower() for x in truth}
    return pred_l == str(truth).strip().lower()


def have_common_substring_of_length(a: str, b: str, length: int) -> bool:
    if length <= 0:
        return True
    a = (a or "").lower()
    b = (b or "").lower()
    if len(a) < length or len(b) < length:
        return False
    subs_a = {a[i:i + length] for i in range(len(a) - length + 1)}
    for j in range(len(b) - length + 1):
        if b[j:j + length] in subs_a:
            return True
    return False


def is_uninformative_token(tok: str) -> bool:
    if tok is None:
        return True
    t = tok.strip().lower()
    if t == "":
        return True
    if t.startswith("uns"):
        return True
    if len(t) < 3:
        return True
    return False


def build_llm(model_path: str, tp_size: int, max_logprobs: int, max_model_len: int) -> Tuple[LLM, Any]:
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        max_logprobs=max_logprobs,
        max_model_len=max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return llm, tokenizer


def run_once(
    llm: LLM,
    prompt: str,
    stop: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_logprobs: int,
) -> Tuple[str, Any]:
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        logprobs=max_logprobs,
        prompt_logprobs=5,
    )
    outputs = llm.generate(prompt, sampling_params)
    text = outputs[0].outputs[0].text
    return text, outputs


def extract_first_token_logprob_sequence(outputs: Any) -> List[float]:
    lp_seq = []
    for step in outputs[0].outputs[0].logprobs:
        step_dict = {k: v.__dict__ for k, v in step.items()}
        first_item = next(iter(step_dict.items()))
        lp_seq.append(float(first_item[1]["logprob"]))
    return lp_seq


def extract_topk_candidates(outputs: Any) -> List[Dict[str, Any]]:
    first_step = outputs[0].outputs[0].logprobs[0]
    d = {k: v.__dict__ for k, v in first_step.items()}
    return list(d.values())


def compute_rank_of_truth(
    candidates: List[Dict[str, Any]],
    truth: Union[str, List[str]],
    common_substr_len: int,
    max_logprobs: int,
    skip_uninformative: bool,
) -> int:
    skip_num = 0
    for item in candidates:
        tok = item.get("decoded_token", "")
        if skip_uninformative and (tok.isspace() or is_uninformative_token(tok)):
            skip_num += 1
            continue
        matched = False
        if isinstance(truth, list):
            for t in truth:
                if have_common_substring_of_length(tok, str(t), common_substr_len):
                    matched = True
                    break
        else:
            matched = have_common_substring_of_length(tok, str(truth), common_substr_len)
        if matched:
            r = int(item.get("rank", max_logprobs))
            return max(1, r - skip_num)
    return max_logprobs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Hits@k (latent knowledge) from first-step logits.")
    ap.add_argument("--model_path", type=str, required=True, help="HF model name or local path for vLLM.")
    ap.add_argument("--dataset_path", type=str, required=True, help="Path to head_to_tail_*.json.")
    ap.add_argument("--prompt_path", type=str, required=True, help="Prompt template file. Use one {} placeholder for question.")
    ap.add_argument("--subset", type=str, default="tail", choices=["head", "torso", "tail"], help="Which subset to evaluate.")
    ap.add_argument("--start", type=int, default=0, help="Start index within the chosen subset.")
    ap.add_argument("--end", type=int, default=None, help="End index (exclusive) within the chosen subset.")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50, 100], help="Report Hits@k for these k values.")
    ap.add_argument("--common_substr_len", type=int, default=3, help="A match if token shares >= this many consecutive chars with truth.")
    ap.add_argument("--max_logprobs", type=int, default=1000, help="How many top tokens to request from vLLM and max rank cap.")
    ap.add_argument("--max_tokens", type=int, default=20, help="Max generated tokens for the answer.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature.")
    ap.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
    ap.add_argument("--stop", type=str, default="\n", help="Stop string for generation.")
    ap.add_argument("--max_model_len", type=int, default=4096, help="Max model length in vLLM.")
    ap.add_argument("--tensor_parallel_size", type=int, default=None, help="Tensor parallel size. Default: torch.cuda.device_count().")
    ap.add_argument("--skip_uninformative", action="store_true", help="Skip whitespace/unsure/short tokens when computing rank.")
    ap.add_argument("--save_details", type=str, default=None, help="If set, save per-example details JSON to this path.")
    args = ap.parse_args()

    tp = args.tensor_parallel_size or max(1, torch.cuda.device_count())

    llm, _ = build_llm(
        model_path=args.model_path,
        tp_size=tp,
        max_logprobs=args.max_logprobs,
        max_model_len=args.max_model_len,
    )

    task_prompt = Path(args.prompt_path).read_text(encoding="utf-8")
    dataset = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))

    data_slice = dataset[args.subset][args.start:args.end] if args.end is not None else dataset[args.subset][args.start:]
    ranks: List[int] = []
    n = 0
    n_exact = 0
    ppl_all: List[float] = []
    ppl_right: List[float] = []
    ppl_wrong: List[float] = []
    ppl_unsure: List[float] = []
    details: List[Dict[str, Any]] = []

    for d in tqdm(data_slice, desc=f"Evaluating {args.subset}[{args.start}:{args.end}]"):
        n += 1
        question = d[2]
        truth = d[3]
        prompt = task_prompt.format(question) + "\xa0"

        pred, outputs = run_once(
            llm=llm,
            prompt=prompt,
            stop=args.stop,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_logprobs=args.max_logprobs,
        )
        pred = normalize_answer(pred)
        if pred == "":
            pred = " "

        lp_seq = extract_first_token_logprob_sequence(outputs)
        ppl = compute_perplexity(lp_seq)
        ppl_all.append(ppl)

        candidates = extract_topk_candidates(outputs)
        rank = compute_rank_of_truth(
            candidates=candidates,
            truth=truth,
            common_substr_len=args.common_substr_len,
            max_logprobs=args.max_logprobs,
            skip_uninformative=args.skip_uninformative,
        )
        ranks.append(rank)

        if pred.strip().lower() == "unsure":
            ppl_unsure.append(ppl)

        if check_exact_match(pred, truth):
            n_exact += 1
            ppl_right.append(ppl)
        else:
            ppl_wrong.append(ppl)

        if args.save_details is not None:
            details.append(
                {
                    "question": question,
                    "truth": truth,
                    "answer": pred,
                    "rank": rank,
                    "topk_first_step": candidates,
                }
            )

    counter = Counter(ranks)
    max_rank = max(counter.keys()) if counter else args.max_logprobs
    filled = {i: counter.get(i, 0) for i in range(1, max_rank + 1)}
    values = np.array([filled[i] for i in range(1, max_rank + 1)], dtype=np.int64)
    cum = np.cumsum(values)
    total = int(cum[-1]) if len(cum) else 0

    print(f"\nN = {n}")
    print(f"Exact-match accuracy (Hits@1 by surface) = {n_exact / max(1, n) * 100:.2f}% ({n_exact}/{n})")
    for kk in sorted(set(args.k)):
        if kk <= 0:
            continue
        idx = min(kk, len(cum)) - 1
        hit = int(cum[idx]) / max(1, total) * 100.0
        print(f"Hits@{kk} = {hit:.2f}%")

    if args.save_details is not None:
        out_path = Path(args.save_details)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved details to: {out_path}")


if __name__ == "__main__":
    main()
