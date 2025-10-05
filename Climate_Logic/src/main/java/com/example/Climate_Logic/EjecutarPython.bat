@echo off
REM === Archivo: run_python.bat ===
REM %1, %2, %3 son los parámetros recibidos desde Java

REM Ruta completa del ejecutable de Python
set PYTHON_EXE="C:\laragon\bin\python\python-3.10\python.exe"

REM Ruta completa del script
set SCRIPT_PATH="C:\Users\Alejandra\Documents\Personal\Climate_Logic\Climate_Logic\src\main\java\com\example\Climate_Logic\prueba.py"

%PYTHON_EXE% %SCRIPT_PATH% %1 %2 %3
