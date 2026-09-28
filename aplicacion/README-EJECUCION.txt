========================================================================
 GUÍA DE PUESTA EN MARCHA Y EJECUCIÓN - PROTOTIPO DE DETECCIÓN DE FRAUDES
 Trabajo Final de Grado (TFG) — Universidad O&M
========================================================================

REQUISITOS PREVIOS:
-------------------
- Python 3.10 o 3.11 instalado en el sistema.
- Acceso a la terminal de comandos (PowerShell / CMD en Windows, o Terminal en macOS/Linux).

PASOS PARA LA INSTALACIÓN Y EJECUCIÓN:
--------------------------------------

1. ABRIR TERMINAL EN LA RAÍZ DEL PROYECTO:
   Navegar a la carpeta principal del proyecto (AIFraudesBancarios).

2. CREAR EL ENTORNO VIRTUAL (venv):
   python -m venv venv

3. ACTIVAR EL ENTORNO VIRTUAL:
   - En Windows (PowerShell / CMD):
     venv\Scripts\activate

   - En macOS / Linux:
     source venv/bin/activate

4. INSTALAR LAS DEPENDENCIAS:
   pip install -r requirements.txt

5. EJECUTAR LA APLICACIÓN STREAMLIT:
   streamlit run app/app.py

6. ACCESO A LA APLICACIÓN:
   Una vez iniciado el servidor local de Streamlit, se abrirá automáticamente en su navegador web en la dirección:
   http://localhost:8501

EJECUCIÓN RÁPIDA EN WINDOWS:
----------------------------
Si está en Windows y prefiere no usar comandos en la terminal, puede hacer doble clic en el archivo lanzador:
run_app.bat

NOTAS IMPORTANTES:
------------------
- Los modelos ya están previamente entrenados y serializados en la carpeta 'models/' (o 'modelos/'). No es necesario reentrenar ningún modelo para probar la aplicación.
- Para realizar pruebas de análisis masivo por lotes (CSV), utilice los archivos de prueba provistos en la carpeta 'data/' ('test_samples.csv', 'lote_pruebas_fraude.csv', 'lote_pruebas_legitimo.csv').
