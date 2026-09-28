# Sistema Híbrido de Detección de Fraudes Bancarios en Tiempo Real (Machine Learning & Reglas Heurísticas)

Este repositorio contiene la implementación completa, reproducible y validada del sistema antifraude bancario desarrollado como trabajo de titulación / tesis de ingeniería. El sistema opera sobre una arquitectura híbrida de **Pre-Autorización** (10 variables operativas sin fuga de información temporal) combinando modelos supervisados, no supervisados, redes neuronales profundas y un motor de reglas de negocio con clasificación semáforo.

---

## 🏛️ Arquitectura del Sistema y Flujo de Decisión

El flujo de procesamiento evalúa cada transacción en milisegundos mediante un esquema híbrido de dos capas:

```
                      [ Transacción Entrante ]
                       (10 variables limpias)
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Motor de Reglas     │ ──(Violación Crítica)──► [ RECHAZO INMEDIATO ]
                     │   (Heurísticas Duras) │
                     └───────────────────────┘
                                 │
                          (Cumple Reglas)
                                 ▼
                     ┌───────────────────────┐
                     │    Random Forest /    │
                     │  XGBoost Calibrado    │
                     └───────────────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               ▼                 ▼                 ▼
          Score < 0.20     0.20 ≤ Score < 0.70    Score ≥ 0.70
         [ APROBADA ]         [ REVISIÓN ]       [ RECHAZADA ]
       (Semáforo Verde)     (Semáforo Amarillo)  (Semáforo Rojo)
```

---

## 🔬 Metodología: Eliminación Rigurosa de Fuga de Información (*Data Leakage*)

A diferencia de aproximaciones tradicionales que utilizan variables retrospectivas (`newbalanceOrig`, `newbalanceDest`), este sistema opera **exclusivamente con 10 variables disponibles antes de la liquidación bancaria**:

1. `amount`: Monto transferido o debitado.
2. `oldbalanceOrg`: Saldo previo en la cuenta origen.
3. `oldbalanceDest`: Saldo previo en la cuenta receptora.
4. `ratio_monto_saldo`: Proporción del saldo comprometido ($monto / (saldo_{orig} + 1)$).
5. `dif_saldo_monto`: Saldo remanente teórico ($saldo_{orig} - monto$).
6. `dest_saldo_cero`: Indicador binario de cuenta destino sin fondos previos.
7. `es_extraccion_alta`: Retiro anómalo sobre umbrales de seguridad.
8. `hora_dia`: Ciclo temporal de la transacción ($step \pmod{24}$).
9. `es_madrugada`: Indicador binario de operación nocturna (0:00 - 6:00 hrs).
10. `type_TRANSFER`: Tipo de transacción codificada.

---

## 📊 Comparativa de Desempeño de Modelos (Conjunto de Prueba Limpio)

Evaluación sobre **263,302 transacciones independientes** de prueba (`test set` desbalanceado real):

| Modelo | Tipo | AUPRC | ROC-AUC | F1-Score | Recall Fraude | Latencia / Tx |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** 🏆 | Supervisado (Ensemble) | **0.8732** | **0.9984** | **0.8711** | **83.1%** | 0.051 ms |
| **XGBoost** | Supervisado (Boosting) | 0.8654 | 0.9979 | 0.8590 | 81.8% | 0.038 ms |
| **Regresión Logística** | Supervisado (Lineal) | 0.4120 | 0.8841 | 0.4412 | 68.2% | 0.002 ms |
| **Autoencoder Profundo** | Red Neuronal (No supervisado) | 0.2845 | 0.8410 | 0.3120 | 59.4% | 0.095 ms |
| **Isolation Forest** | Detección de Anomalías | 0.1980 | 0.7925 | 0.2210 | 51.0% | 0.042 ms |

---

## 📁 Estructura del Repositorio

```text
AIFraudesBancarios/
│
├── Documentacion/               # Documentación académica, manuales y anexos de tesis
│   ├── ANEXOS_TESIS.md          # Anexos completos con tablas, hiperparámetros y figuras
│   ├── MANUAL_USUARIO.md        # Manual de usuario del prototipo interactivo
│   └── analysis_results.md      # Resumen ejecutivo de resultados experimentales
│
├── aplicacion/                  # Ejecutables rápidos y scripts de arranque local
│   ├── run_app.bat              # Script batch para iniciar con un solo clic
│   └── README-EJECUCION.txt     # Guía rápida de ejecución
│
├── app/                         # Prototipo interactivo (Streamlit Dashboard)
│   └── app.py                   # Simulador de transacciones con semáforo y SHAP
│
├── data/                        # Datasets procesados y lotes de prueba
│   ├── lote_pruebas_fraude.csv  # Muestras reales etiquetadas como fraude
│   ├── lote_pruebas_legitimo.csv# Muestras legítimas para pruebas unitarias
│   └── processed/               # Métricas tabulares de latencia, matrices y semillas
│
├── evidencia/anexos_imagenes/   # 13 figuras oficiales en alta resolución (300 DPI)
│   ├── curvas_precision_recall.png
│   ├── curvas_roc.png
│   ├── fig_importancia_features.png
│   ├── matrices_confusion_grid.png
│   └── flujo_motor_hibrido_semaforo.png
│
├── models/                      # Modelos entrenados y serializados (.pkl, .h5)
│   ├── random_forest.pkl        # Modelo principal de producción
│   ├── xgboost.pkl              # Modelo alternativo optimizado
│   ├── logistic_regression.pkl  # Baseline supervisado
│   ├── autoencoder.h5           # Modelo de reconstrucción neuronal
│   └── scaler.pkl               # Escalador ajustado exclusivamente en entrenamiento
│
├── notebooks/                   # Cuadernos Jupyter del ciclo de vida del dato
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocesamiento.ipynb
│   ├── 04_Modelo_RandomForest.ipynb
│   └── 07_Evaluacion_Comparativa.ipynb
│
├── scripts/                     # Scripts modulares de ejecución reproducible
│   ├── 01_regenerar_datos.py    # Pipeline de partición y feature engineering
│   ├── 02_reentrenar_y_evaluar.py # Entrenamiento y evaluación de los 5 modelos
│   ├── ejecutar_mediciones_y_pruebas_cap4.py # Medición de latencia y pruebas
│   └── generar_figuras_anexos.py# Generación de gráficos y curvas de calibración
│
├── src/                         # Módulos centrales de lógica de negocio y preprocesamiento
│   ├── preprocessing.py         # Transformación y validación de features
│   └── rules_engine.py          # Motor de reglas heurísticas
│
├── requirements.txt             # Dependencias del entorno Python
└── README.md                    # Este archivo
```

---

## 🚀 Instalación y Puesta en Marcha

### Prerrequisitos
- Python 3.10 o superior (validado en Python 3.11.9)
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/Oguisjs/AIFraudesBancarios.git
cd AIFraudesBancarios
```

### 2. Crear y activar entorno virtual
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecutar el Prototipo Interactivo
```bash
streamlit run app/app.py
```
O en Windows simplemente haciendo doble clic sobre `aplicacion/run_app.bat`.

---

## 📜 Licencia y Cita Académica

Proyecto desarrollado con fines académicos y de investigación. Si utilizas este código o los resultados para trabajos de titulación o publicaciones científicas, por favor cita:

> **Oguis, J.** (2026). *Sistema Híbrido Basado en Inteligencia Artificial y Reglas Heurísticas para la Detección Temprana de Fraudes en Transacciones Bancarias*. Trabajo de Titulación de Ingeniería.
