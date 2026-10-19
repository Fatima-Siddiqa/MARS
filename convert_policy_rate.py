import csv
from datetime import datetime, date

SRC = "raw_policy_rate_sbp.csv"
OUT = "policy_rate_clean.csv"

rows = []
with open(SRC, newline="", encoding="utf-8-sig") as f:
    for r in csv.reader(f):
        if len(r) < 5 or "SBPOL0030" not in r[2]:
            continue
        d = datetime.strptime(r[1].strip(), "%d-%b-%y").date()
        rows.append((d, float(r[4])))

rows = sorted(set(rows))
# sanity filter: the Target Rate only exists from 2015-05-25, and no future dates
rows = [(d, v) for d, v in rows if date(2015, 5, 25) <= d <= date.today()]

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Observation Date", "Observation Value", "Observation Status"])
    for d, v in rows:
        w.writerow([d.isoformat(), v, "actual"])

print("Rows:", len(rows))
print("First:", rows[0])
print("Last:", rows[-1])