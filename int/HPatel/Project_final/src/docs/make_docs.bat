@echo off
REM Run from anywhere; this script will cd here first.
cd /d "%~dp0"

REM Ensure output dir exists
if not exist "source\api" mkdir "source\api"

REM Generate API .rst files from your code in ..\..\src
sphinx-apidoc -o source\api ..\..\src

REM Build HTML
sphinx-build -b html source build

REM Open docs
start build\html\index.html
echo.
echo Done. Docs at: %CD%\build\html\index.html
