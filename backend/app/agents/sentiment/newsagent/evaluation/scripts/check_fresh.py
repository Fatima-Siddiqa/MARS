import sqlite3

from app.agents.sentiment.newsagent.config import DB_PATH

c = sqlite3.connect(DB_PATH)
q = """SELECT e.entity_id, a.kind, max(a.effective_at), count(*)
       FROM articles a JOIN article_entities e ON e.article_id = a.id
       WHERE e.entity_type = 'ticker' AND e.relevance >= 1
       GROUP BY 1, 2 ORDER BY 1, 2"""
for r in c.execute(q):
    print(r)