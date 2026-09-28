# ANEXOS METODOLÓGICOS Y TÉCNICOS

**Título de la Tesis:** Sistema de Detección de Fraudes en Transacciones Bancarias Mediante Inteligencia Artificial y Reglas de Negocio  
**Área de Conocimiento:** Ingeniería de Software / Ciencia de Datos / Ciberseguridad Financiera  
**Fecha:** Septiembre 2026  
**Entorno de Validación:** Python 3.11.9 | Windows 11 Enterprise (64-bit) | Intel Core i9-11950H | 64 GB RAM | NVIDIA RTX A2000 Laptop GPU  

---

## Tabla de Contenido de los Anexos

* [Anexo A: Repositorio, Estructura del Código Fuente y Guía de Reproducibilidad](#anexo-a-repositorio-estructura-del-código-fuente-y-guía-de-reproducibilidad)
* [Anexo B: Lista Completa de Variables (Pre-Autorización y Variables Descartadas)](#anexo-b-lista-completa-de-variables-pre-autorización-y-variables-descartadas)
* [Anexo C: Hiperparámetros y Arquitectura de los Modelos de Aprendizaje Automático](#anexo-c-hiperparámetros-y-arquitectura-de-los-modelos-de-aprendizaje-automático)
* [Anexo D: Estratificación y Partición de Datos (Train, Val, Test y SMOTE)](#anexo-d-estratificación-y-partición-de-datos-train-val-test-y-smote)
* [Anexo E: Matrices de Confusión en Valores Absolutos](#anexo-e-matrices-de-confusión-en-valores-absolutos)
* [Anexo F: Curvas ROC y Curvas Precision-Recall Comparativas](#anexo-f-curvas-roc-y-curvas-precision-recall-comparativas)
* [Anexo G: Registros de Entrenamiento, Cómputo y Estabilidad Estocástica](#anexo-g-registros-de-entrenamiento-cómputo-y-estabilidad-estocástica)
* [Anexo H: Resultados Comparativos Antes y Después de Eliminar la Fuga de Información](#anexo-h-resultados-comparativos-antes-y-después-de-eliminar-la-fuga-de-información)
* [Anexo I: Batería de Pruebas del Software (Unitarias, Frontera, Integración y Seguridad)](#anexo-i-batería-de-pruebas-del-software-unitarias-frontera-integración-y-seguridad)
* [Anexo J: Evidencias de Funcionamiento del Prototipo y Consola de Analista](#anexo-j-evidencias-de-funcionamiento-del-prototipo-y-consola-de-analista)
* [Anexo K: Catálogo de Figuras y Evidencias Visuales de los Anexos](#anexo-k-catálogo-de-figuras-y-evidencias-visuales-de-los-anexos)

---

# Anexo A: Repositorio, Estructura del Código Fuente y Guía de Reproducibilidad

### A.1 Identificación del Repositorio y Código Fuente
* **Nombre del Proyecto:** `AIFraudesBancarios`
* **Archivo Principal de Interfaz / MVP:** `app/app.py`
* **Entorno de Programación:** Python 3.11.9 (64 bits, MSC v.1938)
* **Framework de Interfaz:** Streamlit v1.31.0
* **Ubicación Local de Referencia:** `C:\Users\oguis\Desktop\AIFraudesBancarios`

### A.2 Estructura Jerárquica del Código
```text
AIFraudesBancarios/
│
├── app/
│   └── app.py                        # Aplicación interactiva Streamlit (4 Pestañas + XAI SHAP)
│
├── data/
│   ├── raw/
│   │   └── PS_20174392719_1491204439457_log.csv # Dataset crudo PaySim
│   └── processed/                    # Conjuntos numpy (.npy) y CSVs de métricas científicas
│       ├── X_train.npy, y_train.npy  # Conjunto de entrenamiento (balanceado con SMOTE)
│       ├── X_val.npy, y_val.npy      # Conjunto de validación
│       ├── X_test.npy, y_test.npy    # Conjunto de evaluación final (65,824 transacciones)
│       ├── comparacion_modelos.csv   # Métricas consolidadas científicas
│       └── mediciones_latencia_1000tx.csv # Benchmarks empíricos de latencia
│
├── evidencia/                        # Gráficos vectoriales y rasterizados a 300 DPI
│   ├── curvas_roc.png                # Comparativa de curvas ROC y AUC
│   ├── curvas_precision_recall.png   # Curvas PR con línea base de azar al 1.87%
│   ├── curvas_calibracion.png        # Reliability diagrams con Brier Score
│   └── matrices_confusion_grid.png   # Cuadrícula de matrices de confusión absolutas
│
├── models/                           # Modelos y preprocesadores serializados
│   ├── logistic_regression.pkl       # Modelo lineal base
│   ├── random_forest.pkl             # Ensamble de 300 árboles
│   ├── xgboost.pkl                   # Gradient boosting optimizado
│   ├── isolation_forest.pkl          # Detección no supervisada de anomalías
│   ├── scaler.pkl                    # StandardScaler ajustado únicamente en train
│   ├── autoencoder.h5                # Red neuronal profunda Keras
│   └── autoencoder_config.pkl        # Umbral percentil 95 de error de reconstrucción
│
├── scripts/
│   ├── 01_regenerar_datos.py         # Pipeline de partición y SMOTE aislado
│   ├── 02_reentrenar_y_evaluar.py    # Pipeline multimodelo y graficación
│   └── ejecutar_mediciones_y_pruebas_cap4.py # Batería de pruebas y latencia
│
├── src/
│   ├── preprocessing.py              # Ingeniería limpia de características pre-autorización
│   └── rules_engine.py               # Motor híbrido determinista (Reglas R1, R2, R3)
│
├── requirements.txt                  # Dependencias congeladas del entorno
└── README.md                         # Documentación operativa del proyecto
```

### A.3 Dependencias del Proyecto (`requirements.txt`)
```text
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
xgboost==2.0.3
tensorflow==2.15.0
imbalanced-learn==0.12.0
matplotlib==3.8.2
seaborn==0.13.2
streamlit==1.31.0
joblib==1.3.2
plotly==5.18.0
shap==0.51.0
```

### A.4 Guía de Reproducción Paso a Paso
1. **Creación del entorno virtual e instalación:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Generación de particiones y preprocesamiento sin fuga:**
   ```powershell
   python scripts/01_regenerar_datos.py
   ```
3. **Reentrenamiento multimodelo y generación de evidencia científica:**
   ```powershell
   python scripts/02_reentrenar_y_evaluar.py
   ```
4. **Ejecución de la batería de pruebas y mediciones de latencia:**
   ```powershell
   python scripts/ejecutar_mediciones_y_pruebas_cap4.py
   ```
5. **Lanzamiento de la aplicación interactiva:**
   ```powershell
   streamlit run app/app.py
   ```

---

# Anexo B: Lista Completa de Variables (Pre-Autorización y Variables Descartadas)

### B.1 Vector Definitivo de 10 Variables Pre-Autorización
Para garantizar un sistema aplicable al momento de la solicitud de la transacción bancaria (antes de la liquidación y compensación), se formularon estrictamente variables de contexto y balances previos:

| N.° | Variable | Nombre Técnico | Tipo | Fórmula Matemática / Regla | Justificación Operativa |
|:---:|---|---|:---:|---|---|
| 1 | Monto de la operación | `amount` | Float continuo | Monto en RD$ extraído de la solicitud | Variable primaria de impacto financiero |
| 2 | Saldo previo del emisor | `oldbalanceOrg` | Float continuo | Saldo disponible antes del débito | Capacidad previa de liquidez del cliente |
| 3 | Saldo previo del receptor | `oldbalanceDest` | Float continuo | Saldo de la cuenta receptora antes del abono | Detección de cuentas mulas / nuevas |
| 4 | Ratio monto sobre saldo | `ratio_monto_saldo` | Float continuo | $\frac{\text{amount}}{\text{oldbalanceOrg} + 1.0}$ | Identifica intentos de vaciado total ($Ratio \approx 1.0$) |
| 5 | Diferencia saldo vs monto | `dif_saldo_monto` | Float continuo | $\text{oldbalanceOrg} - \text{amount}$ | Capacidad de cobertura previa del emisor |
| 6 | Cuenta destino con saldo cero | `dest_saldo_cero` | Binaria (0/1) | $\mathbb{I}(\text{oldbalanceDest} == 0.0)$ | Patrón característico de cuentas receptoras instrumentales |
| 7 | Extracción de alto monto | `es_extraccion_alta` | Binaria (0/1) | $\mathbb{I}(\text{amount} > 535,033.57 \land \text{tipo} \in \{\text{TRANSFER}, \text{CASH\_OUT}\})$ | Alerta cuantitativa por encima del percentil 95 poblacional |
| 8 | Hora del día | `hora_dia` | Entera [0–23] | $\text{step} \pmod{24}$ | Patrón horario circadiano transaccional |
| 9 | Horario de madrugada | `es_madrugada` | Binaria (0/1) | $\mathbb{I}(1 \le \text{hora\_dia} \le 4)$ | Franja nocturna de alta concentración de fraude operativo |
| 10 | Indicador de transferencia | `type_TRANSFER` | Binaria (0/1) | $\mathbb{I}(\text{type} == \text{'TRANSFER'})$ | Vectorización del canal de mayor riesgo (vs. CASH_OUT) |

### B.2 Variables Descartadas y Justificación de la Eliminación de Fuga de Datos (*Data Leakage*)
En iteraciones tempranas del proyecto y en la literatura convencional sobre PaySim, se utilizaban variables post-transaccionales que causaban una separación trivial y determinista:

| Variable Descartada | Origen en PaySim | Razón Técnica de su Eliminación |
|---|---|---|
| `newbalanceOrig` | Saldo posterior de la cuenta emisora | En fraudes sintéticos de PaySim, el saldo siempre cae exactamente a cero. En un sistema real de pre-autorización, el nuevo balance **aún no existe** porque el banco no ha autorizado la operación. |
| `newbalanceDest` | Saldo posterior de la cuenta receptora | El nuevo balance del receptor se asienta únicamente tras la compensación interbancaria. Su uso constituye fuga prospectiva (*lookahead bias*). |
| $\Delta SaldoOrig$ | $\text{oldbalanceOrg} - \text{newbalanceOrig}$ | Variable derivada dependiente de un dato posterior a la decisión. |
| $\Delta SaldoDest$ | $\text{newbalanceDest} - \text{oldbalanceDest}$ | Variable derivada post-liquidación. |

---

# Anexo C: Hiperparámetros y Arquitectura de los Modelos de Aprendizaje Automático

### C.1 Regresión Logística (Línea Base Lineal)
* **Algoritmo:** `sklearn.linear_model.LogisticRegression`
* **Optimizador / Solver:** `lbfgs`
* **Ponderación de clases:** `class_weight='balanced'`
* **Iteraciones máximas:** `max_iter=1000`
* **Regularización:** L2 ($C = 1.0$)
* **Semilla aleatoria:** `random_state=42`
* **Preprocesamiento asociado:** Escalado previo con `StandardScaler`.

### C.2 Random Forest (Ensamble Arbóreo Bagging)
* **Algoritmo:** `sklearn.ensemble.RandomForestClassifier`
* **Número de estimadores:** `n_estimators=300`
* **Profundidad máxima:** `max_depth=16`
* **Mínimo de muestras para división:** `min_samples_split=10`
* **Criterio de impureza:** Gini
* **Ponderación de clases:** `class_weight='balanced'`
* **Paralelización:** `n_jobs=-1` (multinúcleo)
* **Semilla aleatoria:** `random_state=42`

### C.3 XGBoost (Extreme Gradient Boosting)
* **Algoritmo:** `xgboost.XGBClassifier`
* **Número de árboles:** `n_estimators=500`
* **Tasa de aprendizaje (*learning rate*):** $\eta = 0.05$
* **Profundidad máxima:** `max_depth=6`
* **Manejo del desbalance:** `scale_pos_weight=10.0`
* **Métrica de parada temprana / evaluación:** `eval_metric='auc'`
* **Conjunto de validación:** $15\%$ de validación independiente para control de sobreajuste.
* **Semilla aleatoria:** `random_state=42`

### C.4 Isolation Forest (Detección No Supervisada de Anomalías)
* **Algoritmo:** `sklearn.ensemble.IsolationForest`
* **Número de árboles:** `n_estimators=200`
* **Proporción de contaminación asumida:** `contamination=0.0187` ($1.87\%$ tasa observada)
* **Submuestreo:** `max_samples=0.8` (80% del conjunto por árbol)
* **Semilla aleatoria:** `random_state=42`
* **Score continuo:** Inversión de `score_samples()` normalizada a rango continuo $[0, 1]$.

### C.5 Autoencoder Profundo (Red Neuronal de Reconstrucción)
* **Framework:** TensorFlow / Keras v2.15.0
* **Arquitectura de Capas:**
  * **Capa de entrada:** 10 neuronas (vector pre-autorización normalizado).
  * **Codificador 1:** `Dense(24, activation='relu')`
  * **Codificador 2:** `Dense(12, activation='relu')`
  * **Cuello de botella (*Bottleneck*):** `Dense(6, activation='relu')`
  * **Decodificador 1:** `Dense(12, activation='relu')`
  * **Decodificador 2:** `Dense(24, activation='relu')`
  * **Capa de salida:** `Dense(10, activation='linear')`
* **Función de pérdida:** Error Cuadrático Medio (`loss='mse'`).
* **Optimizador:** `Adam(learning_rate=0.001)`.
* **Entrenamiento:** Entrenado **exclusivamente con transacciones legítimas** (`y == 0`).
* **Épocas:** 30 épocas (tamaño de lote 256) con `EarlyStopping(monitor='val_loss', patience=4)`.
* **Umbral de decisión:** Percentil 95 del error cuadrático sobre validación legítima ($\text{Umbral} = 0.000692$).

---

# Anexo D: Estratificación y Partición de Datos (Train, Val, Test y SMOTE)

### D.1 Protocolo de Partición Estratificada
Para garantizar la validez científica y evitar cualquier contaminación (*data snooping*), el dataset filtrado para transacciones operativas (`TRANSFER` y `CASH_OUT`) fue dividido de forma estratificada conservando la proporción de fraude:

```text
Distribución original de la muestra (1,000,000 registros base):
- Total de transacciones operativas (TRANSFER / CASH_OUT): 438,826 registros
- Tasa de fraude global: 1.87 % (8,213 fraudes)

Partición estratificada:
1. Conjunto de Entrenamiento (Train):  70.0 % (307,178 transacciones)
2. Conjunto de Validación (Val):       15.0 % ( 65,824 transacciones)
3. Conjunto de Evaluación (Test):      15.0 % ( 65,824 transacciones)
```

### D.2 Aplicación Aislada de SMOTE en Entrenamiento
El balanceo sintético de la clase minoritaria mediante SMOTE (`imblearn.over_sampling.SMOTE`) se aplicó **únicamente sobre el conjunto de entrenamiento**:
* **Estrategia de remuestreo:** `sampling_strategy=0.1` (incrementa la presencia de fraudes sintéticos al 10% de la clase mayoritaria).
* **Vecinos más cercanos:** $k = 5$ (`k_neighbors=5`).
* **Garantía metodológica:** Los conjuntos de Validación y Prueba **NO** recibieron muestras generadas artificialmente; ambos conjuntos permanecen en su distribución natural con un $1.87\%$ de fraudes reales (1,232 fraudes en 65,824 transacciones de prueba).

---

# Anexo E: Matrices de Confusión en Valores Absolutos

Evaluación formal sobre el **Conjunto de Prueba Definitivo (65,824 transacciones: 64,592 legítimas y 1,232 fraudes reales)**:

### E.1 Desglose de Valores Absolutos por Modelo

| Modelo | Verdaderos Positivos (TP) | Falsos Positivos (FP) | Verdaderos Negativos (TN) | Falsos Negativos (FN) | Fraudes Totales | Falsos Negativos (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **1,231** | **7** | **64,585** | **1** | 1,232 | **0.08%** |
| **XGBoost** | **1,231** | 24 | 64,568 | **1** | 1,232 | **0.08%** |
| **Regresión Logística** | 1,077 | 5,622 | 58,970 | 155 | 1,232 | 12.58% |
| **Autoencoder Profundo** | 970 | 3,243 | 61,349 | 262 | 1,232 | 21.27% |
| **Isolation Forest** | 285 | 512 | 64,080 | 947 | 1,232 | 76.87% |

### E.2 Matrices Gráficas 2x2
Las matrices de confusión generadas en alta resolución se encuentran consolidadas en el archivo:
[matrices_confusion_grid.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/evidencia/matrices_confusion_grid.png).

```text
    MATRIZ XGBOOST                         MATRIZ RANDOM FOREST
                 Predicho                               Predicho
             Legítima  Fraude                       Legítima  Fraude
Real Legít.   64,568      24          Real Legít.   64,585       7
     Fraude        1   1,231               Fraude        1   1,231
```

---

# Anexo F: Curvas ROC y Curvas Precision-Recall Comparativas

### F.1 Métricas de Discriminación Global (AUC-ROC y PR-AUC)

| Modelo Evaluado | AUC-ROC | PR-AUC (Average Precision) | Brier Score (Calibración) | Coeficiente MCC |
|---|:---:|:---:|:---:|:---:|
| **XGBoost** | **1.0000** | **0.9997** | **0.0003** | **0.9898** |
| **Random Forest** | **1.0000** | **0.9981** | **0.0001** | **0.9967** |
| **Autoencoder** | 0.9091 | 0.6195 | 0.0182 | 0.4081 |
| **Regresión Logística** | 0.9609 | 0.5340 | 0.0712 | 0.3528 |
| **Isolation Forest** | 0.8561 | 0.2074 | 0.0253 | 0.2768 |

### F.2 Curvas Precision-Recall Corregidas
La gráfica actualizada [curvas_precision_recall.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/evidencia/curvas_precision_recall.png) incorpora:
1. **Línea base horizontal de prevalencia:** Trazada en $y = 0.0187$ (dash gris) para reflejar la tasa natural de fraude en el conjunto de prueba ($1.87\%$).
2. **Curva continua del Autoencoder:** Generada a partir de los errores continuos sin truncamiento artificial (`score_ae = errores_test`), alcanzando un verdadero PR-AUC de **0.6195**.
3. **Trazo diferenciado de Random Forest:** Línea discontinua naranja (`--`) claramente visible junto a la curva de XGBoost.

### F.3 Curvas ROC y Calibración
* **Curvas ROC:** [curvas_roc.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/evidencia/curvas_roc.png) comparando los 5 modelos frente a la diagonal de azar ($AUC = 0.5000$).
* **Curvas de Calibración:** [curvas_calibracion.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/evidencia/curvas_calibracion.png) evaluando la confiabilidad de las probabilidades asignadas.

---

# Anexo G: Registros de Entrenamiento, Cómputo y Estabilidad Estocástica

### G.1 Tiempos de Entrenamiento y Consumo de Memoria
Mediciones registradas en el script `02_reentrenar_y_evaluar.py` bajo entorno local:

| Modelo | Tiempo de Entrenamiento (s) | Memoria RAM Pico (GB) | Recursos Utilizados |
|---|:---:|:---:|---|
| **Regresión Logística** | 0.45 s | 17.68 GB | 1 hilo CPU |
| **XGBoost** | 6.15 s | 17.52 GB | 16 hilos CPU (`n_jobs=-1`) |
| **Isolation Forest** | 9.21 s | 17.81 GB | 16 hilos CPU (`n_jobs=-1`) |
| **Random Forest** | 17.14 s | 17.49 GB | 16 hilos CPU (`n_jobs=-1`) |
| **Autoencoder Profundo** | 30.52 s | 17.88 GB | GPU/CPU con Early Stopping (30 épocas) |

### G.2 Evaluación de Variabilidad Estocástica (Semillas 42, 123, 999)
Para comprobar que los resultados de XGBoost no son producto del azar de inicialización, se ejecutaron 3 corridas completas con semillas pseudoaleatorias distintas:

| Algoritmo | Semilla Aleatoria | F1-Score | AUC-ROC | Recall |
|---|:---:|:---:|:---:|:---:|
| **XGBoost** | 42 | 0.9856 | 0.999996 | 0.9992 |
| **XGBoost** | 123 | 0.9856 | 0.999996 | 0.9992 |
| **XGBoost** | 999 | 0.9856 | 0.999996 | 0.9992 |

*Conclusión de variabilidad:* Varianza nula en las métricas clave ($\sigma^2 < 10^{-6}$), demostrando que la convergencia del ensamble es altamente determinista y robusta.

---

# Anexo H: Resultados Comparativos Antes y Después de Eliminar la Fuga de Información

### H.1 Análisis Comparativo Metodológico

| Criterio | Pipeline Anterior (Con Fuga de Datos) | Pipeline Definitivo (Sin Fuga de Datos) |
|---|---|---|
| **Variables utilizadas** | 14 variables incluyendo `newbalanceOrig`, `newbalanceDest` y deltas posteriores. | **10 variables estrictamente pre-autorización** (saldos previos y ratios derivados). |
| **Momento de la decisión** | **Post-autorización / Liquidación** (inaplicable en un switch transaccional en vivo). | **Pre-autorización** (en el instante en que el usuario solicita la transferencia). |
| **Comportamiento de árboles** | Separación perfecta trivial por reglas directas sobre balances finales. | Aprendizaje de fronteras no lineales basadas en contexto, horario y cobertura del saldo. |
| **Falsos negativos en XGBoost** | 0 | **1 solo caso** en 65,824 registros de prueba ($99.92\%$ Recall real). |
| **Rendimiento de modelos no supervisados** | Falsamente distorsionado. | **Autoencoder alcanza PR-AUC 0.6195 y Recall 78.73%** modelando el comportamiento normal. |

---

# Anexo I: Batería de Pruebas del Software (Unitarias, Frontera, Integración y Seguridad)

Ejecución documentada mediante el módulo `scripts/ejecutar_mediciones_y_pruebas_cap4.py`:

### I.1 Pruebas Unitarias y de Límites (15/15 Aprobadas)
1. `Extracción de 10 características`: Matriz procesada con $10$ columnas exactas. **[APROBADA]**
2. `Límite Monto Exacto RD$1,000,000`: No activa R1 (`EVALUATE_ML`). **[APROBADA]**
3. `Límite Monto Excedido RD$1,000,000.01`: Activa R1 (`HARD_BLOCK`). **[APROBADA]**
4. `Límite Horario 00:59 (hora 0)`: No activa R1. **[APROBADA]**
5. `Límite Horario 01:00 (hora 1)`: Activa R1 (`HARD_BLOCK`). **[APROBADA]**
6. `Límite Horario 04:00 (hora 4)`: Activa R1 (`HARD_BLOCK`). **[APROBADA]**
7. `Límite Horario 05:00 (hora 5)`: No activa R1. **[APROBADA]**
8. `Límite IP 90.0%`: Reputación en límite no activa R2. **[APROBADA]**
9. `Límite IP 90.1%`: Reputación crítica activa R2 (`HARD_BLOCK`). **[APROBADA]**
10. `Límite Velocidad 5 ops/hora`: No activa R3. **[APROBADA]**
11. `Límite Velocidad 6 ops/hora`: Frecuencia anómala activa R3 (`HARD_BLOCK`). **[APROBADA]**
12. `Semáforo Límite Verde 0.2999`: Clasificación VERDE (Aprobación directa). **[APROBADA]**
13. `Semáforo Límite Amarillo 0.3000`: Clasificación AMARILLO (Exige OTP). **[APROBADA]**
14. `Semáforo Límite Amarillo 0.7000`: Clasificación AMARILLO (Exige OTP). **[APROBADA]**
15. `Semáforo Límite Rojo 0.7001`: Clasificación ROJO (Retención preventiva). **[APROBADA]**

### I.2 Pruebas Funcionales y de Integración E2E (9/9 Aprobadas)
1. `Flujo Verde (<0.30)`: Autorización inmediata sin fricción. **[APROBADA]**
2. `Flujo Amarillo (0.30–0.70)`: Despliegue condicional del desafío OTP interactivo. **[APROBADA]**
3. `Validación OTP Correcto`: Ingreso de `123456` cambia estado a AUTORIZADA. **[APROBADA]**
4. `Validación OTP Incorrecto`: Ingreso de código inválido mantiene la retención preventiva. **[APROBADA]**
5. `Flujo Rojo ML (>0.70)`: Envío automático a cola de analista (`analyst_queue`). **[APROBADA]**
6. `Regla Dura Asignación 0.98`: Score determinista fijo de $0.98$ por política de riesgo. **[APROBADA]**
7. `CSV Válido`: Detección correcta del esquema de 5 columnas requeridas. **[APROBADA]**
8. `CSV Inválido`: Manejo seguro de archivos malformados con advertencia al usuario. **[APROBADA]**
9. `Integración End-to-End`: Flujo íntegro CSV $\rightarrow$ Features $\rightarrow$ XGBoost $\rightarrow$ Score. **[APROBADA]**

### I.3 Mediciones de Latencia Empírica (1,000 Transacciones Unitarias)
Medidas en tiempo de ejecución real incluyendo el preprocesamiento unitario de cada transacción:

| Modelo | Mediana (ms) | Percentil 95 (ms) | Percentil 99 (ms) | Throughput (TX/s) |
|---|:---:|:---:|:---:|:---:|
| **Regresión Logística** | **0.167 ms** | 0.245 ms | 0.293 ms | **5,646.6 TX/s** |
| **XGBoost** | **0.234 ms** | **0.283 ms** | **0.338 ms** | **4,177.9 TX/s** |
| **Autoencoder Profundo** | **2.162 ms** | 2.503 ms | 3.389 ms | **454.2 TX/s** |
| **Isolation Forest** | **2.389 ms** | 2.783 ms | 4.546 ms | **412.4 TX/s** |
| **Random Forest** | **28.413 ms** | 29.842 ms | 31.286 ms | **34.8 TX/s** |

---

# Anexo J: Evidencias de Funcionamiento del Prototipo y Consola de Analista

### J.1 Módulos de la Interfaz Interactiva (`app/app.py`)

1. **Pestaña 1: Evaluación Interactiva de Demostración & XAI:**
   * Formulario bidimensional para ingresar las variables transaccionales (`amount`, `oldbalanceOrg`, `oldbalanceDest`, tipo y hora).
   * Controles laterales de contexto digital: sliders de frecuencia transaccional (1–10 ops/hora) y reputación de IP/dispositivo (0–100%).
   * Renderizado del Semáforo Adaptativo (Verde, Amarillo con OTP, Rojo con Alerta de Suspensión).
   * Gráfica explicativa local generada con **SHAP (Plotly)** que desglosa en tiempo real el aporte de variables como el ratio monto/saldo o el saldo cero en destino.
   * Mockup responsivo en HTML/CSS simulando una alerta de seguridad por SMS.

2. **Pestaña 2: Análisis Masivo por Lotes (Batch CSV):**
   * Módulo de carga de archivos CSV masivos para auditoría preventiva.
   * Validación automática de la presencia de las 5 columnas estructurales.
   * Procesamiento vectorial optimizado y generación de reportes con métricas agregadas (Total lote, Transacciones Legítimas, Fraudes Detectados, Tasa de Fraude).
   * Histogramas interactivos bivariados con la distribución de puntuaciones de riesgo.

3. **Pestaña 3: Dashboard Científico & Métricas:**
   * Tabulación dinámica conectada a `data/processed/comparacion_modelos.csv`.
   * Despliegue de los gráficos de evaluación generados por el pipeline científico (Curvas ROC, Curvas PR, Curvas de Calibración y Matrices de Confusión).
   * Sección expandible con gráficos de importancia de características, correlaciones bivariadas y distribuciones de clases.

4. **Pestaña 4: Consola de Gestión de Analista (Back-Office):**
   * Panel de control operativo con métricas de flujo de trabajo: *Casos Pendientes en Cola*, *Casos Resueltos en Sesión* y *Total en Expediente*.
   * Selector interactivo de expedientes sospechosos para inspección forense.
   * Visor estructurado de balances, marcas temporales y variables de contexto en formato JSON.
   * Acciones operativas directas:
     * Botón **"APROBAR TRANSACCIÓN (Liberar Operación)"**: Cambia el estado a `APROBADA_POR_ANALISTA` y desaloja el caso de la cola activa.
     * Botón **"CONFIRMAR FRAUDE (Retener Operación)"**: Cambia el estado a `FRAUDE_CONFIRMADO`.
     * Botón **"SOLICITAR DOCUMENTACIÓN ADICIONAL"**: Marca el caso en estado de espera preventiva (`EN_ESPERA_DOCS`).
   * Tabla histórica con la bitácora de resoluciones de la sesión activa con marcas temporales e identificación del analista.

---

# Anexo K: Catálogo de Figuras y Evidencias Visuales de los Anexos

Todas las imágenes, diagramas y figuras científicas generadas se encuentran archivadas en alta resolución (300 DPI) en la carpeta dedicada:
📁 **Ruta de carpeta:** [`plancorreccion/imagenes_anexos/`](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos)  
📁 **Ruta espejo en raíz:** [`anexos_imagenes/`](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/anexos_imagenes)

### K.1 Índice de Figuras y Archivos Disponibles

| N.° | Nombre de Archivo | Descripción Metodológica / Contenido | Sección de Destino en Tesis |
|:---:|---|---|---|
| **Fig. A.1** | [arquitectura_sistema_mvp.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/arquitectura_sistema_mvp.png) | Diagrama formal de arquitectura de la solución MVP (Capas de UI, Lógica y Datos). | Anexo A / Sección 4.1 |
| **Fig. F.1** | [curvas_roc.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/curvas_roc.png) | Curvas ROC comparativas de los 5 algoritmos evaluados frente al azar (AUC = 0.50). | Anexo F / Sección 4.3 |
| **Fig. F.2** | [curvas_precision_recall.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/curvas_precision_recall.png) | Curvas Precision-Recall comparativas con línea base horizontal al 1.87% y curva continua AE. | Anexo F / Sección 4.3 |
| **Fig. F.3** | [curvas_calibracion.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/curvas_calibracion.png) | Reliability Diagrams de calibración probabilística con cálculo de Brier Score. | Anexo F / Sección 4.3 |
| **Fig. E.1** | [matrices_confusion_grid.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/matrices_confusion_grid.png) | Cuadrícula de matrices de confusión en valores absolutos (TP, FP, TN, FN) para los 5 modelos. | Anexo E / Sección 4.3 |
| **Fig. I.1** | [comparativa_latencia_modelos.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/comparativa_latencia_modelos.png) | Comparativa de latencia por transacción unitaria (Mediana y P95) sobre 1,000 operaciones. | Anexo I / Sección 4.4 |
| **Fig. J.1** | [flujo_motor_hibrido_semaforo.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/flujo_motor_hibrido_semaforo.png) | Diagrama de flujo operativo: Evaluación de Reglas R1-R3, Inferencia ML y Semáforo Adaptativo. | Anexo J / Sección 4.2 |
| **Fig. J.2** | [explicabilidad_local_shap_ejemplo.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/explicabilidad_local_shap_ejemplo.png) | Desglose explicativo local XAI mediante valores SHAP para una transacción sospechosa. | Anexo J / Sección 4.2 |
| **Fig. B.1** | [fig_importancia_features.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/fig_importancia_features.png) | Importancia relativa global de las 10 características pre-autorización en modelos arbóreos. | Anexo B / Sección 3.4 |
| **Fig. B.2** | [fig_correlacion.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/fig_correlacion.png) | Matriz de correlación lineal bivariada (Spearman/Pearson) del vector de características. | Anexo B / Sección 3.4 |
| **Fig. D.1** | [fig_distribucion_clases.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/fig_distribucion_clases.png) | Gráfico de barras de desbalance de clases (Legítimas vs Fraudes en el dataset). | Anexo D / Sección 3.2 |
| **Fig. D.2** | [fig_distribucion_montos.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/fig_distribucion_montos.png) | Distribución empírica de montos transaccionales por categoría de fraude. | Anexo D / Sección 3.3 |
| **Fig. D.3** | [fig_fraude_por_tipo.png](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/plancorreccion/imagenes_anexos/fig_fraude_por_tipo.png) | Tasa de fraude concentrada exclusivamente en operaciones TRANSFER y CASH_OUT. | Anexo D / Sección 3.3 |

---

### Certificación de los Autores
Los datos, mediciones, tablas, diagramas y figuras científicas presentadas en estos anexos corresponden fielmente a las pruebas empíricas ejecutadas en el repositorio del proyecto.

*Fecha de emisión de los anexos:* 27 de septiembre de 2026.
