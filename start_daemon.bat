@echo off
REM Start AURA daemon in background
cd /d D:\Aura
start /B pythonw aura_daemon.py > NUL 2>&1
echo AURA daemon started.
