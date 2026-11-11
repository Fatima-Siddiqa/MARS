"""Shared helper for the noise-aware workflow: LoRA fine-tune FinBERT on rows, return probabilities.

Same recipe as finetune_finbert_labelled.py (LoRA r=16 on query/value + classifier head,
AdamW lr 2e-4, class-weighted cross-entropy, 10% warm-up) but with a FIXED number of epochs and
no early stopping, because validation labels from the LLM are themselves noisy
(the three seeds showed validation F1 0.59-0.74 while test F1 was 0.74-0.79).
"""
import random

import numpy as np

from app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled import CLASSES


def train_and_predict(train_rows, texts, seed=1, epochs=5, lr=2e-4, rank=16, batch=16,
                      base="ProsusAI/finbert", dry_run=False):
    """Returns an array (len(texts), 3) of probabilities in CLASSES order (positive, negative, neutral)."""
    if dry_run:  # random numbers, only for testing that the pipeline runs. NOT valid results.
        return np.random.default_rng(seed).dirichlet([0.5, 0.5, 0.5], size=len(texts))

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base)
    label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    cfg = LoraConfig(task_type="SEQ_CLS", r=rank, lora_alpha=rank, lora_dropout=0.1,
                     target_modules=["query", "value"], modules_to_save=["classifier"])
    model = get_peft_model(model, cfg).to(dev)

    x = [r["headline"] for r in train_rows]
    y = np.array([label2id[r["label"]] for r in train_rows])
    n = len(x)
    cnt = np.bincount(y, minlength=3)
    weights = torch.tensor(n / (3 * np.maximum(cnt, 1)), dtype=torch.float, device=dev)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    steps = epochs * ((n + batch - 1) // batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            b = tok([x[j] for j in idx], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(dev)
            loss = F.cross_entropy(model(**b).logits, torch.tensor(y[idx], device=dev), weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(dev)
            out.append(torch.softmax(model(**b).logits, dim=-1).cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, 3))
    order = [label2id[c] for c in CLASSES]
    del model, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return probs[:, order]
