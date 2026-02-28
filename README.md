# Hits@k Evaluation Script

This repository contains a standalone evaluation script to compute **Hits@k** as defined in our paper *“Are LLMs Really Not Knowledgeable? Mining the Submerged Knowledge in LLMs’ Memory”*. 

**Hits@k** measures whether the **ground-truth answer appears within the top-k tokens** (by logit rank) at the **first decoding step**, which helps quantify latent knowledge that may not be surfaced by greedy decoding.

## What the script does

Given:
- a QA prompt template (with one `{}` placeholder for the question),
- a dataset json in the `head/torso/tail` format (same as used in the paper),
- a vLLM-loadable model,

the script will:
1. run generation for each question,
2. read the **top-N first-step logits** (via vLLM `logprobs`),
3. compute the rank where the correct answer (or a subword match) first appears,
4. report `Hits@k` for one or multiple `k`.

Matching protocol follows the paper: **a token is considered a match if it shares at least `common_substr_len` consecutive characters with the ground truth** (default `3`). 

## Files

- `hits_at_k.py`: main script (argparse-based, comment-free; ready for open-sourcing).

## Requirements

- Python 3.9+
- CUDA-capable GPU environment for vLLM
- Packages:
  - `vllm`
  - `torch`
  - `transformers`
  - `tqdm`
  - `numpy`

Example installation:

```bash
pip install vllm torch transformers tqdm numpy
```

> Note: vLLM installation varies by CUDA/driver version. Follow the official vLLM installation guide for your system.

## Dataset format

The script expects a JSON file like:

```json
{
  "head":  [[..., ..., "question", "truth"], ...],
  "torso": [[..., ..., "question", "truth"], ...],
  "tail":  [[..., ..., "question", "truth"], ...]
}
```

Where:
- `d[2]` is the question string
- `d[3]` is the ground truth (string or list of strings)

## Prompt template

The `--prompt_path` file should be a text template containing exactly one `{}` placeholder for the question, e.g.:

```
Answer the following questions in as few words as possible. Say "unsure" if you don’t know.
Question: {}
Answer:
```

(See Appendix C in the paper for the prompt used in our experiments.) 

## Usage

### Minimal example

```bash
python hits_at_k.py \
  --model_path /path/to/your/model \
  --dataset_path head_to_tail_dbpedia.json \
  --prompt_path prompts/QA.txt
```

### Common settings (match paper-style evaluation)

```bash
python hits_at_k.py \
  --model_path /path/to/your/model \
  --dataset_path head_to_tail_dbpedia.json \
  --prompt_path prompts/QA.txt \
  --subset tail \
  --k 1 5 10 50 100 \
  --common_substr_len 3 \
  --max_logprobs 1000 \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_tokens 20 \
  --stop "\n"
```

### Save per-example details

```bash
python hits_at_k.py \
  --model_path /path/to/your/model \
  --dataset_path head_to_tail_dbpedia.json \
  --prompt_path prompts/QA.txt \
  --subset tail \
  --save_details outputs/dp_logprobs_tail.json
```

The saved JSON includes:
- model answer
- rank of the first matching token
- full first-step `topk_first_step` candidate list

## Notes

- `--tensor_parallel_size` defaults to `torch.cuda.device_count()`. Override it if needed.
- If you want to emulate the paper’s “filter uninformative tokens” analysis, add `--skip_uninformative` to ignore whitespace/`unsure`/very short tokens when computing the match rank. (This is an analysis knob; use consistently when comparing results.)

## License

Choose a license (MIT/Apache-2.0) before release.

---

# Table 2 reproduction script

The repository also includes a second script that reproduces **Table 2** using the **original “uns/blank” skip rule** from the provided `runskip.py` logic.

## Files (additional)

- `run_skip.py`: Table 2 reproduction script (original “uns/blank” skip rule)

## How Table 2 evaluation differs

The Table 2 script performs a *single retry* when the initial generation is uninformative:

- If the initial generation contains `"uns"` or is blank (`gen.strip()==""`), it selects the next best first-step token using the same recursive `skip(logprobs, rank)` logic (starting from `rank=1`), appends that token to the prompt, and regenerates once.

No new filtering rules are introduced beyond the original behavior.

# Table 2 Reproduction (Hits@k with the original “uns/blank” skip rule)

## Usage

### Basic run (subset=tail)

```bash
python run_skip.py   --model_path /path/to/your/model   --dataset_path head_to_tail_dbpedia.json   --prompt_path prompts/QA.txt   --subset tail
```

### Save per-sample details + summary stats

```bash
python run_skip.py   --model_path /path/to/your/model   --dataset_path head_to_tail_dbpedia.json   --prompt_path prompts/QA.txt   --subset tail   --output_json outputs/details_tail.json   --output_stats outputs/stats_tail.json
```

### Basic run (subset=tail)

```bash
python run_skip.py   --model_path /path/to/your/model   --dataset_path head_to_tail_dbpedia.json   --prompt_path prompts/QA.txt   --subset tail
```

### Save per-sample details + summary stats

```bash
python run_skip.py   --model_path /path/to/your/model   --dataset_path head_to_tail_dbpedia.json   --prompt_path prompts/QA.txt   --subset tail   --output_json outputs/details_tail.json   --output_stats outputs/stats_tail.json
```

### Save per-sample details + summary stats

```bash
python run_skip.py   --model_path /path/to/your/model   --dataset_path head_to_tail_dbpedia.json   --prompt_path prompts/QA.txt   --subset tail   --output_json outputs/details_tail.json   --output_stats outputs/stats_tail.json
```



