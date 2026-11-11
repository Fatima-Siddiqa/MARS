"""STEP 1. Summarise several runs on the gold test set: mean +/- SD, and how stable predictions are.

    python -m app.agents.sentiment.newsagent.evaluation.scripts.aggregate_seeds ^
        --runs app\\agents\\sentiment\\newsagent\\training\\runs\\ft1 ^
               app\\agents\\sentiment\\newsagent\\training\\runs\\ft2 ^
               app\\agents\\sentiment\\newsagent\\training\\runs\\ft3 ^
        --gold app\\agents\\sentiment\\newsagent\\training\\data\\gold.csv
Prints macro-F1 / accuracy / per-class F1 as mean +/- SD (baseline once, it does not train).
If torch+peft are installed it also reloads each saved adapter, predicts the gold headlines and reports
  * mean pairwise agreement between seeds and Fleiss' kappa across seeds (prediction stability)
  * macro-F1 of the ENSEMBLE (average of the seeds' probabilities)
Use --no-predict to skip the second part.
"""
import argparse
import json
import statistics as st
from itertools import combinations
from pathlib import Path

import numpy as np

from app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled import (
    CLASSES,
    macro_f1,
    read_labels,
)


def fleiss_kappa(pred):          # pred: (n_raters, n_items) of ints 0..2
    m, n = pred.shape
    counts = np.stack([(pred == c).sum(0) for c in range(3)], 1)       # items x categories
    p_i = ((counts ** 2).sum(1) - m) / (m * (m - 1))
    p_j = counts.sum(0) / (n * m)
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else float("nan")


def ms(v):
    return f"{st.mean(v):.3f} +/- {st.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--no-predict", action="store_true")
    a = ap.parse_args()

    ms_ = [json.loads((Path(r) / "metrics.json").read_text()) for r in a.runs]
    base = ms_[0]["baseline"]
    print(f"runs: {len(ms_)}  (test n={ms_[0]['n_test']}, same baseline for all)")
    print(f"BASELINE   macro-F1 {base['macro_f1']:.3f}  acc {base['accuracy']:.3f}")
    print(f"FINE-TUNED macro-F1 {ms([m['finetuned']['macro_f1'] for m in ms_])}  "
          f"acc {ms([m['finetuned']['accuracy'] for m in ms_])}")
    for c in CLASSES:
        print(f"   {c:<9} F1 {ms([m['finetuned']['per_class_f1'][c] for m in ms_])}"
              f"   (baseline {base['per_class_f1'][c]:.3f})")
    print("gain 95% CI per run: " + "; ".join(
        f"[{m['macro_f1_gain_95ci'][0]:+.3f}, {m['macro_f1_gain_95ci'][1]:+.3f}]" for m in ms_))

    if a.no_predict:
        return
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("\n(torch/peft not installed: skipping stability and ensemble. Use --no-predict to silence.)")
        return
    gold = read_labels(a.gold)
    texts = [r["headline"] for r in gold]
    base_name = ms_[0]["args"].get("base", "ProsusAI/finbert")
    tok = AutoTokenizer.from_pretrained(base_name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    all_probs = []
    for run in a.runs:
        model = AutoModelForSequenceClassification.from_pretrained(base_name)
        label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
        model = PeftModel.from_pretrained(model, str(Path(run) / "adapter")).to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128,
                        return_tensors="pt").to(dev)
                out.append(torch.softmax(model(**b).logits, -1).cpu().numpy())
        p = np.concatenate(out)[:, [label2id[c] for c in CLASSES]]
        all_probs.append(p)
    y = np.array([CLASSES.index(r["label"]) for r in gold])
    preds = np.array([p.argmax(1) for p in all_probs])
    pair = [float((preds[i] == preds[j]).mean()) for i, j in combinations(range(len(preds)), 2)]
    print(f"\nprediction stability across {len(preds)} seeds: mean pairwise agreement {np.mean(pair):.1%}, "
          f"Fleiss kappa {fleiss_kappa(preds):.3f}")
    ens = np.mean(all_probs, 0).argmax(1)
    print(f"ENSEMBLE of the seeds: macro-F1 {macro_f1(y, ens):.3f}  acc {(ens == y).mean():.3f}")


if __name__ == "__main__":
    main()
