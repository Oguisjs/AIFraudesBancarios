@echo off
echo ============================================================
echo   Sistema de Deteccion de Fraudes Bancarios - IA Prototipo
echo ============================================================
echo.
echo Iniciando servidor Streamlit...
echo Accede en: http://localhost:8501
echo.
echo Presiona CTRL+C para detener el servidor.
echo.
"%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0app\app.py"
pause
