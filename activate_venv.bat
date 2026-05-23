@echo off
REM Script para ativar venv com encoding UTF-8 correto
echo Setting Python encoding to UTF-8...
set PYTHONIOENCODING=utf-8

echo Ativando venv...
call venv\Scripts\activate.bat

echo.
echo Virtual environment ativada!
echo Python version:
python --version
echo.
echo Para executar o sistema, use:
echo   python teste_sistema_completo.py
echo   python run_simulation.py
echo.
