@echo off
cd /d G:\Project\docheck
call .venv\Scripts\activate
start http://localhost:8000
uvicorn app.main:app --port 8000