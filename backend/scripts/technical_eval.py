"""Run the Level 2 walk-forward evaluation: python -m scripts.technical_eval"""
import json

from app.agents.technical.evaluation import run

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=float))
