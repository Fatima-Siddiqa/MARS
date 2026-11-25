@echo off
REM Daily refresh of the fundamentals agent's official PSX data, model and calibration.
REM Scheduled by Windows Task Scheduler (task "MARS fundamentals refresh"); see docs/fundamentals_agent.md.
REM Uses the repo-root virtual environment (MARS\.venv). Output is appended to the log below.
cd /d "%~dp0.."
if not exist "..\data\processed\fundamentals" mkdir "..\data\processed\fundamentals"
"..\.venv\Scripts\python.exe" -m app.agents.fundamentals.pipeline >> "..\data\processed\fundamentals\refresh_task.log" 2>&1
