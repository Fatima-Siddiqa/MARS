from app.agents.sentiment.newsagent.polarity import FinBertScorer

s = FinBertScorer()
print(s.version)
heads = ["PTCL posts Rs9.7b consolidated loss",
         "PPL's profit rises 18% to Rs16.9 billion"]
for h, (pos, neg, neu) in zip(heads, s.score(heads)):
    print(f"{h}\n  positive={pos:.3f}  negative={neg:.3f}  neutral={neu:.3f}")