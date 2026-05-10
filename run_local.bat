@echo off
title TIC Research — Local Server
set PYTHONPATH=C:\Users\vince\Lib\site-packages;%PYTHONPATH%
set FLASK_ENV=development
echo.
echo  TIC Research starting at http://localhost:5000
echo.
C:\Python314\python.exe app.py
pause
