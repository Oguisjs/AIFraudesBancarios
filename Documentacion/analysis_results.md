# Reporte de Análisis Técnico y Corrección de Errores

Este documento presenta los hallazgos del análisis de código realizado sobre el proyecto de **Detección de Fraudes Bancarios**. Se identificaron tres problemas principales que afectan la robustez, consistencia científica y experiencia de usuario del prototipo.

---

## 1. Errores Críticos Detectados

### 1.1 Inconsistencia en Ingeniería de Características (Data Leakage)
> [!IMPORTANT]
> **Severidad:** Alta (Impacto directo en la precisión de las predicciones)
> **Ubicación:** [app.py](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/app/app.py#L320-L324)

* **Problema:** En la pestaña de **Procesamiento Masivo (Batch)**, el umbral para determinar si una transacción es de alto monto (`es_extraccion_alta`) se calcula dinámicamente usando el percentil 95 del archivo CSV subido por el usuario (`df_proc['amount'].quantile(0.95)`). 
* **Causa:** Si el usuario sube un lote pequeño o con una distribución sesgada (por ejemplo, transacciones de montos muy bajos, donde el percentil 95 es de solo RD$15,000), transacciones normales serán erróneamente clasificadas con `es_extraccion_alta = 1`. Durante el entrenamiento, este umbral se fijó de manera absoluta en base a la población global (calculado en **RD$535,033.57**).
* **Impacto:** Falsos positivos/negativos masivos al evaluar lotes externos en producción.

### 1.2 Advertencias de Scikit-Learn (UserWarning por nombres de características)
> [!WARNING]
> **Severidad:** Media (Genera ruido en consola e indica discrepancias de entrada)
> **Ubicación:** [app.py](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/app/app.py#L163) y llamadas de inferencia.

* **Problema:** Se presentan dos advertencias de discrepancia en la consola:
  1. `UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names`: Ocurre porque el escalador se entrenó con un DataFrame pero recibe un array de NumPy en la inferencia individual.
  2. `UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names`: Ocurre en el procesamiento por lotes porque los modelos supervisados se entrenaron con un array sin nombres de columnas (debido al balanceo de clases con SMOTE que descarta metadatos) pero reciben un DataFrame.
* **Causa:** Diferentes tipos de datos de entrada (DataFrames vs NumPy Arrays) en el flujo de entrenamiento y la aplicación.

### 1.3 Ausencia de Validación de Estructura de Archivo CSV
> [!WARNING]
> **Severidad:** Media (Inestabilidad de la aplicación ante errores de usuario)
> **Ubicación:** [app.py](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/app/app.py#L296-L327)

* **Problema:** Si un usuario sube un archivo CSV que no tiene la estructura de columnas exacta del dataset PaySim (ej. faltan columnas como `oldbalanceOrg` o `newbalanceDest`), la aplicación Streamlit falla inmediatamente mostrando un traceback de Python en pantalla.
* **Causa:** Falta de un paso de validación previo al procesamiento de características.

---

## 2. Plan de Correcciones Propuesto

### Modificación en [app.py](file:///c:/Users/oguis/Desktop/AIFraudesBancarios/app/app.py)

A continuación se detallan los cambios necesarios para corregir cada uno de los problemas identificados.

```diff
# 1. Definir constante del percentil 95 global obtenido del dataset de entrenamiento
+ UMBRAL_95_ENTRENAMIENTO = 535033.57

def preprocesar_transaccion(tipo, monto, saldo_orig_antes,
                             saldo_orig_despues, saldo_dest_antes,
                             saldo_dest_despues, step=1):
    """Calcula las variables derivadas requeridas por el modelo"""
    error_orig = saldo_orig_despues + monto - saldo_orig_antes
    error_dest = saldo_dest_antes + monto - saldo_dest_despues
    orig_cero  = int(saldo_orig_despues == 0)
    ratio      = monto / (saldo_orig_antes + 1)
-   umbral_95  = 500000  # aproximado del percentil 95
-   extraccion = int(tipo in ['TRANSFER','CASH_OUT'] and monto > umbral_95)
+   extraccion = int(tipo in ['TRANSFER','CASH_OUT'] and monto > UMBRAL_95_ENTRENAMIENTO)
    hora       = step % 24
    es_transfer = int(tipo == 'TRANSFER')

-   features = np.array([[monto, saldo_orig_antes, saldo_orig_despues,
-                          saldo_dest_antes, saldo_dest_despues,
-                          error_orig, error_dest, orig_cero,
-                          ratio, extraccion, hora, es_transfer]])
+   # Creamos un DataFrame con nombres de columnas para que el StandardScaler no lance UserWarnings
+   features = pd.DataFrame([[monto, saldo_orig_antes, saldo_orig_despues,
+                             saldo_dest_antes, saldo_dest_despues,
+                             error_orig, error_dest, orig_cero,
+                             ratio, extraccion, hora, es_transfer]],
+                           columns=['amount', 'oldbalanceOrg', 'newbalanceOrig',
+                                    'oldbalanceDest', 'newbalanceDest',
+                                    'error_balance_orig', 'error_balance_dest',
+                                    'orig_queda_cero', 'ratio_monto_saldo',
+                                    'es_extraccion_alta', 'hora_dia', 'type_TRANSFER'])
    return features

def predecir(modelo_nombre, features):
    """Obtiene la predicción y el score de riesgo según el modelo seleccionado"""
    if modelo_nombre == 'Autoencoder':
        # Autoencoder necesita datos escalados (features ya es un DataFrame estructurado)
        features_sc = scaler.transform(features)
        recon = autoencoder.predict(features_sc, verbose=0)
        error = float(np.mean(np.power(features_sc - recon, 2), axis=1)[0])
        umbral = ae_cfg['umbral']
        pred = int(error > umbral)
        score = min(1.0, error / (umbral * 2.0))
    elif modelo_nombre == 'Isolation Forest':
        modelo = modelos[modelo_nombre]
-       raw = modelo.predict(features)
+       # Se convierte a array NumPy con .values para evitar UserWarnings de feature names
+       raw = modelo.predict(features.values)
        pred = int(raw[0] == -1)
-       score_raw = float(-modelo.score_samples(features)[0])
+       score_raw = float(-modelo.score_samples(features.values)[0])
        score = min(1.0, max(0.0, (score_raw - 0.35) / 0.3))
    else:
        modelo = modelos[modelo_nombre]
-       pred = int(modelo.predict(features)[0])
-       score = float(modelo.predict_proba(features)[0][1])
+       pred = int(modelo.predict(features.values)[0])
+       score = float(modelo.predict_proba(features.values)[0][1])
    return pred, score
```

```diff
# 2. Validación de columnas y procesamiento consistente en la Pestaña Batch (Pestaña 2)
    if archivo:
        df_batch = pd.DataFrame(pd.read_csv(archivo))
-       st.success(f"Archivo cargado correctamente. Registros detectados: {len(df_batch):,}")
-       
-       st.subheader("Vista Previa de los Datos Cargados")
-       st.dataframe(df_batch.head(10), use_container_width=True)
+
+       # Validar existencia de columnas requeridas para el modelo
+       required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
+       missing_cols = [col for col in required_cols if col not in df_batch.columns]
+
+       if missing_cols:
+           st.error(f"❌ El archivo CSV no cuenta con las columnas requeridas: {', '.join(missing_cols)}")
+       else:
+           st.success(f"✅ Archivo cargado correctamente. Registros detectados: {len(df_batch):,}")
+           
+           st.subheader("Vista Previa de los Datos Cargados")
+           st.dataframe(df_batch.head(10), use_container_width=True)

        modelo_batch: str = st.selectbox(
            'Selecciona el Modelo para Procesamiento por Lotes',
            ['XGBoost', 'Random Forest'],
        ) or 'XGBoost'
        
        if st.button('🔎 EJECUTAR ANÁLISIS MASIVO DE TRANSACCIONES', type='primary'):
-           if not modelos_cargados:
...
                with st.spinner('Preprocesando lote completo y calculando vectores de características...'):
                    # 1. Ingeniería de características masiva sobre DataFrame
                    df_proc = df_batch.copy()
                    df_proc['error_balance_orig'] = (df_proc['newbalanceOrig'] + df_proc['amount'] - df_proc['oldbalanceOrg'])
                    df_proc['error_balance_dest'] = (df_proc['oldbalanceDest'] + df_proc['amount'] - df_proc['newbalanceDest'])
                    df_proc['orig_queda_cero']    = (df_proc['newbalanceOrig'] == 0).astype(int)
                    df_proc['ratio_monto_saldo']  = df_proc['amount'] / (df_proc['oldbalanceOrg'] + 1)
                    
-                   umbral_95 = df_proc['amount'].quantile(0.95)
                    df_proc['es_extraccion_alta'] = (
                        (df_proc['type'].isin(['TRANSFER','CASH_OUT'])) & 
-                       (df_proc['amount'] > umbral_95)
+                       (df_proc['amount'] > UMBRAL_95_ENTRENAMIENTO)
                    ).astype(int)
                    df_proc['hora_dia'] = df_proc['step'] % 24
                    df_proc['type_TRANSFER'] = (df_proc['type'] == 'TRANSFER').astype(int)
                    
                    features_batch = [
                        'amount', 'oldbalanceOrg', 'newbalanceOrig',
                        'oldbalanceDest', 'newbalanceDest',
                        'error_balance_orig', 'error_balance_dest',
                        'orig_queda_cero', 'ratio_monto_saldo',
                        'es_extraccion_alta', 'hora_dia', 'type_TRANSFER'
                    ]
                    
                    X_batch = df_proc[features_batch].fillna(0)
                    
                    # Cargar modelo seleccionado
                    modelo = modelos[modelo_batch]
                    
                    # 2. Predecir
-                   probas = modelo.predict_proba(X_batch)[:, 1]
+                   # Se pasa .values para evitar el UserWarning de feature names
+                   probas = modelo.predict_proba(X_batch.values)[:, 1]
```
