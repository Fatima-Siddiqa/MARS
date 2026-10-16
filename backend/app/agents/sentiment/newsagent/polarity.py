"""Article polarity. FinBERT inference only (no training): three softmax
probabilities per headline. tone = P(pos) - P(neg); confidence = 1 - P(neu).

A tiny lexicon scorer exists ONLY so the pipeline and tests run on machines
without torch/transformers. Its output is tagged scorer='lexicon-fallback' and
the pipeline refuses to use it unless --allow-fallback is passed."""
import re

from app.agents.sentiment.newsagent import config


class FinBertScorer:
    name = "finbert"

    def __init__(self, model_name=None, temperature=None, batch_size=32, device=None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "FinBERT needs torch and transformers: pip install -r requirements-model.txt") from exc
        self._torch = torch
        self.model_name = model_name or config.FINBERT_MODEL
        self.temperature = temperature or config.FINBERT_TEMPERATURE
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        id2label = {int(i): l.lower() for i, l in self.model.config.id2label.items()}
        self.idx = {v: k for k, v in id2label.items()}
        for needed in ("positive", "negative", "neutral"):
            if needed not in self.idx:
                raise RuntimeError(f"Unexpected FinBERT labels: {id2label}")
        self.version = f"{self.model_name}@T={self.temperature}"

    def score(self, texts):
        torch = self._torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t or "" for t in texts[i:i + self.batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits / self.temperature
                probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for p in probs:
                out.append((p[self.idx["positive"]], p[self.idx["negative"]], p[self.idx["neutral"]]))
        return out


POS = set("""profit profits rise rises rose up jump jumps surge surges gain gains gained growth grow grows
record strong beat beats upgrade upgraded dividend bonus approve approves approved boost boosts improve
improves recovery rally rallies expands expansion win wins payment""".split())
NEG = set("""loss losses fall falls fell down drop drops plunge plunges slump decline declines weak miss misses
downgrade downgraded default fraud probe penalty fine fined ban cut cuts crisis delay delays shortage
warn warns warning concern concerns risk risks halt halts suspend suspends raid""".split())


class LexiconScorer:
    name = "lexicon-fallback"
    version = "lexicon-fallback-1"

    def score(self, texts):
        out = []
        for t in texts:
            words = re.findall(r"[a-z]+", (t or "").lower())
            p = sum(w in POS for w in words)
            n = sum(w in NEG for w in words)
            total = p + n + 1.0
            pp, pn = p / total, n / total
            out.append((pp, pn, max(0.0, 1.0 - pp - pn)))
        return out


def get_scorer(allow_fallback=False):
    try:
        return FinBertScorer()
    except Exception as exc:
        if allow_fallback:
            return LexiconScorer()
        raise RuntimeError(
            f"FinBERT is not available ({exc}). Install requirements-model.txt, or pass "
            "--allow-fallback for a test run only (scores are NOT valid for evaluation).") from exc


def tone_from_probs(p_pos, p_neg):
    return float(p_pos - p_neg)


def confidence_from_probs(p_neu):
    return float(1.0 - p_neu)


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))
