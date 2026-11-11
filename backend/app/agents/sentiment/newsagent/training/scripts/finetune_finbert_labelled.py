"""Fine-tune FinBERT on hand-labelled PSX headlines and compare it with the original.

Input : the labelling CSV you filled in (columns: article_id, headline, label; kind/date optional).
Output: <out>/adapter (LoRA) or <out>/model (--full), metrics.json, splits.csv

What it does
  1. Reads labels (positive/negative/neutral; pos/neg/neu and +1/-1/0 also accepted).
  2. Splits train/val/test with GROUPS (identical or number-only-different headlines stay
     together) so near-duplicates cannot leak between splits. --split time uses date order.
  3. Scores val and test with the UNTOUCHED FinBERT first (the baseline).
  4. Trains with class-weighted cross-entropy, early stopping on validation macro-F1.
  5. On the held-out TEST set reports accuracy, macro-F1, per-class F1, confusion matrix and
     a paired bootstrap 95% CI for (fine-tuned - baseline) macro-F1.

Install once:  pip install peft
Run:
    python -m app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled ^
        --labels app\\agents\\sentiment\\newsagent\\training\\data\\labelling.csv ^
        --out app\\agents\\sentiment\\newsagent\\training\\runs\\ft1
Self-consistency check of your labelling (after relabelling a week later):
    python -m app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled ^
        --labels app\\agents\\sentiment\\newsagent\\training\\data\\labelling.csv ^
        --agreement app\\agents\\sentiment\\newsagent\\training\\data\\relabel_100.csv
If most training labels came from an assistant, label a random subset yourself (blind) and
use it as the test set; the same --agreement call then gives assistant-vs-you kappa:
    python -m app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled ^
        --labels app\\agents\\sentiment\\newsagent\\training\\data\\labelling.csv ^
        --agreement app\\agents\\sentiment\\newsagent\\training\\data\\gold.csv
    python -m app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled ^
        --labels app\\agents\\sentiment\\newsagent\\training\\data\\labelling.csv ^
        --gold app\\agents\\sentiment\\newsagent\\training\\data\\gold.csv ^
        --out app\\agents\\sentiment\\newsagent\\training\\runs\\ft1
"""
import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np

CLASSES = ("positive", "negative", "neutral")
TRAINING_DIR = Path(__file__).resolve().parents[1]
TRAINING_RUNS_DIR = TRAINING_DIR / "runs"
ALIASES = {
    "positive": "positive", "pos": "positive", "+": "positive", "1": "positive", "+1": "positive",
    "negative": "negative", "neg": "negative", "-": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "0": "neutral",
}


def norm(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"\d+", " ", s)).strip()


def read_labels(path):
    rows, bad, seen = [], 0, set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            lab = ALIASES.get((r.get("label") or "").strip().lower())
            head = (r.get("headline") or "").strip()
            aid = (r.get("article_id") or "").strip()
            if not head or aid in seen:
                continue
            if lab is None:
                if (r.get("label") or "").strip():
                    bad += 1
                continue
            seen.add(aid)
            rows.append({"article_id": aid, "headline": head, "label": lab,
                         "kind": (r.get("kind") or "").strip(), "date": (r.get("date") or "").strip()})
    if bad:
        print(f"WARNING: {bad} rows had an unrecognised label and were skipped")
    return rows


def split_rows(rows, seed, mode, frac=(0.70, 0.15, 0.15), gold=None):
    """Return dict split -> list of rows. Groups never straddle splits.
    gold: optional list of human-labelled rows that form the whole test set."""
    if gold:
        gold_ids = {r["article_id"] for r in gold}
        gold_keys = {norm(r["headline"]) for r in gold}
        rest = [r for r in rows if r["article_id"] not in gold_ids and norm(r["headline"]) not in gold_keys]
        parts = split_rows(rest, seed, mode, frac=(0.85, 0.15, 0.0))
        parts["test"] = list(gold)
        return parts
    groups = {}
    for r in rows:
        groups.setdefault(norm(r["headline"]), []).append(r)
    keys = sorted(groups)
    if mode == "time":
        order = sorted(rows, key=lambda r: (r["date"], r["article_id"]))
        n = len(order)
        cut1, cut2 = int(n * frac[0]), int(n * (frac[0] + frac[1]))
        parts = {"train": order[:cut1], "val": order[cut1:cut2], "test": order[cut2:]}
        train_keys = {norm(r["headline"]) for r in parts["train"]}
        parts["val"] = [r for r in parts["val"] if norm(r["headline"]) not in train_keys]
        seen_keys = train_keys | {norm(r["headline"]) for r in parts["val"]}
        parts["test"] = [r for r in parts["test"] if norm(r["headline"]) not in seen_keys]
        return parts
    random.Random(seed).shuffle(keys)
    total, parts, acc = len(rows), {"train": [], "val": [], "test": []}, 0
    for k in keys:
        share = acc / total
        s = "train" if share < frac[0] else "val" if share < frac[0] + frac[1] else "test"
        parts[s].extend(groups[k])
        acc += len(groups[k])
    return parts


def macro_f1(y, p, k=3):
    f = []
    for c in range(k):
        if not (y == c).any():
            continue
        tp = ((y == c) & (p == c)).sum()
        fp = ((y != c) & (p == c)).sum()
        fn = ((y == c) & (p != c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else 0.0


def per_class_f1(y, p, k=3):
    out = []
    for c in range(k):
        tp = ((y == c) & (p == c)).sum()
        fp = ((y != c) & (p == c)).sum()
        fn = ((y == c) & (p != c)).sum()
        out.append(0.0 if tp == 0 else float(2 * tp / (2 * tp + fp + fn)))
    return out


def confusion(y, p, k=3):
    m = np.zeros((k, k), dtype=int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def cohen_kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = sum(((a == c).mean() * (b == c).mean()) for c in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def paired_boot_diff(y, p_new, p_old, reps=2000, seed=3):
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        diffs.append(macro_f1(y[i], p_new[i]) - macro_f1(y[i], p_old[i]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def agreement(labels_csv, relabel_csv, out_csv="agreement_disagreements.csv"):
    """Compare two label files on the rows they share: raw agreement, kappa, confusion matrix,
    and a CSV listing every disagreement (read it to see WHERE the two labellers differ)."""
    first = {r["article_id"]: r for r in read_labels(labels_csv)}
    second = {r["article_id"]: r for r in read_labels(relabel_csv)}
    ids = [i for i in second if i in first]
    if len(ids) < 20:
        raise SystemExit(f"Only {len(ids)} overlapping labelled rows; need more for a meaningful check.")
    a = [first[i]["label"] for i in ids]
    b = [second[i]["label"] for i in ids]
    print(f"rows compared: {len(ids)}  raw agreement: {np.mean(np.array(a) == np.array(b)):.1%}  "
          f"Cohen's kappa between the two label sets: {cohen_kappa(a, b):.3f}")
    print(f"\nconfusion: rows = {Path(labels_csv).name}, columns = {Path(relabel_csv).name}")
    print(f"{'':<10}" + "".join(f"{c:>10}" for c in CLASSES))
    for ca in CLASSES:
        print(f"{ca:<10}" + "".join(f"{sum(1 for x, y in zip(a, b) if x == ca and y == cb):>10}" for cb in CLASSES))
    print("\nlabel share  " + "  ".join(
        f"{c}: first {np.mean(np.array(a) == c):.0%} / second {np.mean(np.array(b) == c):.0%}" for c in CLASSES))
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["article_id", "headline", "first_label", "second_label"])
        for i, x, y in zip(ids, a, b):
            if x != y:
                w.writerow([i, first[i]["headline"], x, y])
    print(f"\ndisagreements written to {out_csv}")
    print("Rule of thumb: kappa above ~0.6 is substantial; much lower means the two labellers "
          "use different conventions, which must be resolved before training.")


def run_training(args, parts):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base)
    label2id = {k.lower(): int(v) for k, v in model.config.label2id.items()}
    assert set(CLASSES) <= set(label2id), f"unexpected labels in base model: {label2id}"
    model.to(dev)

    def enc(rows):
        return [r["headline"] for r in rows], np.array([label2id[r["label"]] for r in rows])

    x_tr, y_tr = enc(parts["train"])
    x_va, y_va = enc(parts["val"])
    x_te, y_te = enc(parts["test"])

    @torch.no_grad()
    def predict(texts):
        model.eval()
        out = []
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(dev)
            out.append(torch.softmax(model(**b).logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    # 1) baseline = the untouched model, scored BEFORE any adapter is attached
    base_va, base_te = predict(x_va).argmax(1), predict(x_te).argmax(1)
    print(f"baseline  val macro-F1 {macro_f1(y_va, base_va):.3f} | test macro-F1 {macro_f1(y_te, base_te):.3f}")

    # 2) attach LoRA (or train everything with --full)
    if not args.full:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(task_type="SEQ_CLS", r=args.rank, lora_alpha=args.rank, lora_dropout=0.1,
                         target_modules=["query", "value"], modules_to_save=["classifier"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    lr = args.lr if args.lr else (2e-5 if args.full else 2e-4)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _, p in trainable], lr=lr, weight_decay=0.01)
    n, bs = len(x_tr), args.batch
    steps = args.epochs * math.ceil(n / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    cnt = np.bincount(y_tr, minlength=3)
    weights = torch.tensor(n / (3 * np.maximum(cnt, 1)), dtype=torch.float, device=dev)
    rng = np.random.default_rng(args.seed)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b = tok([x_tr[j] for j in idx], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(dev)
            loss = F.cross_entropy(model(**b).logits, torch.tensor(y_tr[idx], device=dev), weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        f1 = macro_f1(y_va, predict(x_va).argmax(1))
        history.append({"epoch": ep, "val_macro_f1": f1})
        print(f"epoch {ep}: val macro-F1 {f1:.3f}")
        if f1 > best:
            best, bad = f1, 0
            best_state = {nme: p.detach().cpu().clone() for nme, p in trainable}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    with torch.no_grad():
        for nme, p in model.named_parameters():
            if nme in best_state:
                p.copy_(best_state[nme].to(p.device))

    # 3) held-out test
    probs = predict(x_te)
    ft = probs.argmax(1)
    lo, hi = paired_boot_diff(y_te, ft, base_te)
    id2name = {v: k for k, v in label2id.items()}
    order = [label2id[c] for c in CLASSES]
    res = {
        "n_train": len(x_tr), "n_val": len(x_va), "n_test": len(x_te),
        "baseline": {"accuracy": float((base_te == y_te).mean()), "macro_f1": macro_f1(y_te, base_te),
                     "per_class_f1": dict(zip(id2name.values(), per_class_f1(y_te, base_te)))},
        "finetuned": {"accuracy": float((ft == y_te).mean()), "macro_f1": macro_f1(y_te, ft),
                      "per_class_f1": dict(zip(id2name.values(), per_class_f1(y_te, ft)))},
        "macro_f1_gain_95ci": [lo, hi],
        "confusion_finetuned_rows_true_cols_pred": {"order": [id2name[i] for i in order],
                                                    "matrix": confusion(y_te, ft)[np.ix_(order, order)].tolist()},
        "history": history, "args": vars(args),
    }
    kinds = [r["kind"] for r in parts["test"]]
    for kd in sorted(set(kinds)):
        m = np.array([k == kd for k in kinds])
        if kd and m.sum() >= 10:
            res.setdefault("by_kind", {})[kd] = {
                "n": int(m.sum()), "baseline_macro_f1": macro_f1(y_te[m], base_te[m]),
                "finetuned_macro_f1": macro_f1(y_te[m], ft[m])}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.full:
        model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    else:
        model.save_pretrained(out / "adapter")
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    with open(out / "splits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["article_id", "split"])
        for s, rows in parts.items():
            for r in rows:
                w.writerow([r["article_id"], s])
    print(f"\nTEST (n={len(x_te)})  baseline macro-F1 {res['baseline']['macro_f1']:.3f}  "
          f"fine-tuned {res['finetuned']['macro_f1']:.3f}  gain 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(json.dumps({k: res[k] for k in ("baseline", "finetuned", "confusion_finetuned_rows_true_cols_pred")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--agreement", help="second labelling CSV: print agreement with --labels and exit")
    ap.add_argument("--gold", help="CSV labelled by a human, used as the ENTIRE test set "
                                   "(rows are removed from training)")
    ap.add_argument("--out", default=str(TRAINING_RUNS_DIR / "ft1"))
    ap.add_argument("--base", default="ProsusAI/finbert")
    ap.add_argument("--split", choices=["group", "time"], default="group")
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    if a.agreement:
        return agreement(a.labels, a.agreement)

    rows = read_labels(a.labels)
    counts = {c: sum(r["label"] == c for r in rows) for c in CLASSES}
    print(f"labelled rows: {len(rows)}  {counts}")
    if len(rows) < 300:
        print("WARNING: under 300 labels; the held-out test set will be too small to trust.")
    gold = read_labels(a.gold) if a.gold else None
    if gold:
        print(f"gold test set: {len(gold)} human-labelled rows")
    parts = split_rows(rows, a.seed, a.split, gold=gold)
    for s, rs in parts.items():
        print(f"  {s:<5}{len(rs):>5}  " + "  ".join(f"{c[:3]}={sum(r['label'] == c for r in rs)}" for c in CLASSES))
    if min(len(parts["train"]), len(parts["val"]), len(parts["test"])) < 30:
        raise SystemExit("A split has fewer than 30 rows; label more headlines first.")
    run_training(a, parts)


if __name__ == "__main__":
    main()