import argparse
import math
import re
import sys

from nltk.util import ngrams
import pandas as pd
import torch
from tqdm import tqdm

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


@torch.no_grad()
def compute_ppl(texts, model_name="Qwen/Qwen2-1.5B", device="cuda", batch_size=8, max_length=1024):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[INFO] Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    model.eval()

    total_nll, total_tokens = 0.0, 0
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(model.device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        outputs = model(**enc, labels=labels)
        nll = outputs.loss.item() * int(enc["attention_mask"].sum())
        total_nll += nll
        total_tokens += int(enc["attention_mask"].sum())

    return math.exp(total_nll / max(total_tokens, 1))


def calculate_distinct(responses):
    unigrams = []
    bigrams = []
    for res in tqdm(responses, desc="Computing Distinct"):
        res = str(res)
        tokens = res.lower().split()
        unigrams.extend(tokens)
        if len(tokens) > 1:
            bigrams.extend(list(ngrams(tokens, 2)))

    if not unigrams:
        return 0.0, 0.0

    distinct_1 = len(set(unigrams)) / len(unigrams)
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return distinct_1, distinct_2


def compute_ead(texts, n=1):
    all_ids = []
    for t in tqdm(texts, desc=f"Computing EAD-{n}"):
        all_ids.extend(enc.encode(t))
    grams = list(zip(*[all_ids[i:] for i in range(n)])) if len(all_ids) >= n else []
    total = len(grams)
    distinct = len(set(grams))
    if total == 0:
        return 0.0
    V = enc.n_vocab
    expected_upper = V * (1 - ((V - 1) / V) ** total)
    return distinct / expected_upper if expected_upper > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Compute PPL , Dist-1/2, EAD-1/2 (cl100k)")
    parser.add_argument("--input_csv", default='./output/.csv')  # input the path of the file you need to evaluate
    parser.add_argument("--pred_col", default="final_empathetic_response")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ppl_model", default="gpt2")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.pred_col not in df.columns:
        print(f"[ERROR] CSV can not find {args.pred_col}")
        sys.exit(1)
    texts = [normalize_text(t) for t in df[args.pred_col].astype(str).tolist() if t and str(t).strip()]
    if len(texts) == 0:
        print("[ERROR] text is empty")
        sys.exit(1)

    ppl = compute_ppl(texts, model_name=args.ppl_model, device=args.device)
    dist1, dist2 = calculate_distinct(texts)
    ead1 = compute_ead(texts, n=1)
    ead2 = compute_ead(texts, n=2)

    print("\n===== RESULTS =====")
    print(f"PPL (corpus, cl100k): {ppl:.4f}  ↓")
    print(f"Distinct-1: {dist1:.6f}  ↑")
    print(f"Distinct-2: {dist2:.6f}  ↑")
    print(f"EAD-1: {ead1:.6f}  ↑")
    print(f"EAD-2: {ead2:.6f}  ↑")


if __name__ == "__main__":
    main()
