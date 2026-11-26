@echo off
REM Daily refresh of the fundamentals agent's official PSX data, model and calibration,
REM and of the sentiment agent's news (scored with the fine-tuned LoRA FinBERT).
REM Scheduled by Windows Task Scheduler (task "MARS fundamentals refresh"); see docs/fundamentals_agent.md.
REM Uses the repo-root virtual environment (MARS\.venv). Output is appended to the log below.
cd /d "%~dp0.."
if not exist "..\data\processed\fundamentals" mkdir "..\data\processed\fundamentals"
"..\.venv\Scripts\python.exe" -m app.agents.fundamentals.pipeline >> "..\data\processed\fundamentals\refresh_task.log" 2>&1
"..\.venv\Scripts\python.exe" -m app.agents.sentiment.newsagent.pipeline --google-days 3 --psx-days 3 >> "..\data\processed\fundamentals\refresh_task.log" 2>&1
