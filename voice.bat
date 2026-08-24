@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

where ffmpeg >nul 2>&1
if errorlevel 1 (
    for /f "delims=" %%f in ('dir /s /b "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg.exe" 2^>nul') do set "FFMPEG_EXE=%%f"
    if defined FFMPEG_EXE (
        for %%d in ("!FFMPEG_EXE!") do set "PATH=%%~dpd;%PATH%"
    ) else (
        echo WARNING: could not find ffmpeg.exe automatically. Make sure ffmpeg is on PATH.
    )
)

".venv\Scripts\python.exe" transcribe.py %*
