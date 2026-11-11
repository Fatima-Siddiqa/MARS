import sqlite3

from app.agents.sentiment.newsagent.config import DB_PATH

c = sqlite3.connect(DB_PATH)
q = """SELECT ticker, round(quality, 2), state FROM agent_outputs
       WHERE as_of = (SELECT max(as_of) FROM agent_outputs) ORDER BY ticker"""
for r in c.execute(q):
    print(r)