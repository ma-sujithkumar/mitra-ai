# Database Build Monitoring Plan

## Objectives
1. Start the database build process in the background: `/home/sujithma/venv/bin/python build_leaderboard_db.py -c config/config.ini --resume -v > log 2>&1`
2. Configure a recurring schedule (cronjob) via the `schedule` tool to check the log every 10 minutes.
3. Automatically inspect the log and identify any issues upon notification.
4. Implement generic or operational fixes as needed to keep the process running.
5. Keep the process running silently, responding only when required or when a fix is implemented.

## Steps
1. **Initiate background process**: Propose running the database build script with output redirected to `log`.
2. **Set up cron schedule**: Call the `schedule` tool with `CronExpression="*/10 * * * *"` to check the log for errors.
3. **Log checking logic**: On each trigger, check if the process is running, inspect the last lines of the log, grep for tracebacks or error messages, and handle any operational failures (e.g. database locks, memory issues, missing parameters).
