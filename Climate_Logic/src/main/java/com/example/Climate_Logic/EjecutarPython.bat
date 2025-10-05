@echo off
REM === Archivo: EjecutarPython.bat ===
REM %1, %2, %3 son parámetros que puede recibir desde Java

REM Ir al directorio del BAT (por seguridad)
cd /d "%~dp0"

REM Ruta relativa al script Python desde este BAT
set SCRIPT_PATH=..\..\..\..\..\..\..\ClimatoLogic\ClimatoLogic.py

REM Ejecutable de Python (usa el del PATH del sistema)
python "%SCRIPT_PATH%" %1 %2 %3
