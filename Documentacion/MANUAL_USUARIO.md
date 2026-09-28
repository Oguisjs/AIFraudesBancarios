# 🏦 Manual de Usuario y Documentación Técnica
## Sistema de Detección de Fraude Transaccional con Inteligencia Artificial

Este manual explica detalladamente la función del sistema de detección de fraudes bancarios, cómo utilizar la aplicación web interactiva, cómo verificar e interpretar las alertas de fraude y cómo funciona a nivel técnico cada uno de los modelos de Inteligencia Artificial implementados.

---

## 📖 1. Función de la Aplicación

La aplicación es un **prototipo de software en tiempo real y batch** desarrollado para la detección y mitigación de fraudes financieros. Está entrenado utilizando el dataset de referencia científica **PaySim1**, que simula un entorno transaccional móvil de gran escala basado en registros reales de dinero móvil en África.

El sistema tiene dos propósitos principales:
1. **Demostración de defensa de tesis/grado:** Muestra visualmente el desempeño comparativo de enfoques supervisados (XGBoost, Random Forest), semi-supervisados (Autoencoders de Keras) y no supervisados (Isolation Forest).
2. **Simulador operativo:** Permite a analistas de prevención de fraudes simular transacciones de forma individual o procesar lotes (lotes CSV) para ver cómo operaría el motor de decisiones de una entidad bancaria frente a intentos de fraude (como el vaciado de cuentas).

---

## 🕹️ 2. Guía de Uso: Cómo Utilizar la Aplicación

La aplicación está dividida en **cuatro pestañas principales** para facilitar la interacción y el análisis científico y operativo:

### Pestaña 1 — 🔍 1. Evaluación en Tiempo Real & XAI
Permite simular y evaluar el riesgo de una transacción en tiempo real ingresando manualmente sus datos contables y de contexto digital.

1. **Configurar Datos de Contexto (Sidebar):**
   * **Frecuencia Transacciones (1h):** Frecuencia atípica que activa la Regla R3 si supera las 5 operaciones/hora.
   * **Score Riesgo Dispositivo/IP (%):** IP sospechosa o reputación de dispositivo. Un valor >90% activa la Regla R2.
   * **Celular del Titular:** Número telefónico utilizado para el envío de alertas SMS simuladas.
2. **Configurar Datos Transaccionales (Formulario Central):**
   * **Tipo de Transacción:** Escoja entre `TRANSFER` (Transferencia) o `CASH_OUT` (Retiro de efectivo). *Nota: En el dataset de simulación, el fraude se concentra exclusivamente en estos dos tipos.*
   * **Monto (RD$):** Escriba el monto transaccionado.
   * **Hora del Día (1-24):** Representa la hora del día. Si se encuentra en la madrugada (1 AM - 4 AM) combinada con alto monto, activa la Regla R1.
   * **Modelo de IA Seleccionado:** Seleccione cuál de los 4 modelos de Inteligencia Artificial ejecutará la predicción de riesgo.
3. **Configurar Saldos Contables:**
   * **Saldo Cuenta Origen (Antes y Después):** El saldo de la cuenta que envía el dinero antes y después del movimiento.
   * **Saldo Cuenta Destino (Antes y Después):** El saldo de la cuenta receptora antes y después del movimiento.
4. **Ejecutar Evaluación:** Haga clic en `🚨 EVALUAR TRANSACCIÓN EN TIEMPO REAL`.
5. **Interpretación y Respuestas del Motor Híbrido:**
   * **Riesgo Alto (>70% o Bloqueo por Regla):** La transacción es **Bloqueada/Suspendida** de inmediato, se genera una vista previa del **Mockup SMS de Alerta** enviado al celular, y se registra automáticamente el caso en la cola del analista (Pestaña 4). Se muestra además el gráfico interactivo de explicabilidad SHAP.
   * **Riesgo Moderado (30% - 70%):** Se requiere **Fricción Adaptativa**. Se solicita al usuario ingresar un código OTP (código demo: `123456`) en un panel interactivo para autorizar la operación.
   * **Riesgo Bajo (<30%):** La transacción es catalogada como **Legítima** y aprobada de forma automática.

### Pestaña 2 — 📂 2. Análisis Masivo por Lotes (CSV)
Permite subir un archivo de transacciones múltiples en lote para su puntuación masiva.

1. **Subir Archivo:** Cargue un archivo CSV (se provee uno de demostración con 500 registros en `data/test_samples.csv`).
2. **Validación Automática:** El sistema verifica la presencia de las columnas obligatorias (`step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`).
3. **Seleccionar Modelo Batch:** Seleccione entre **XGBoost** o **Random Forest** (los modelos recomendados por su alto desempeño).
4. **Ejecutar Scoring:** Haga clic en `🔎 EJECUTAR ANÁLISIS MASIVO DE TRANSACCIONES`.
5. **Revisar Resultados:**
   * Métricas generales (Total del lote, transacciones legítimas, fraudes detectados y tasa de fraude).
   * **Gráficas Separadas (Histogramas de Plotly):** Visualización independiente de la distribución de scores de riesgo para **Transacciones Legítimas (Verde `#10B981`)** y **Transacciones Fraudulentas (Rojo `#EF4444`)**, mostrando conteos y promedios de riesgo por grupo.
   * **Visión Comparativa Superpuesta:** Histograma consolidado superpuesto para comparar ambas clases en una sola vista.

### Pestaña 3 — 📊 3. Dashboard Científico & Métricas
Muestra el respaldo científico del proyecto. Presenta la tabla de métricas consolidadas del conjunto de prueba (`Precision`, `Recall`, `F1-Score`, `AUC-ROC`, `PR-AUC`, `MCC`, y porcentaje de falsos negativos). Además, carga dinámicamente gráficos de entrenamiento clave como las Curvas ROC y las matrices de confusión de los modelos.

### Pestaña 4 — 🛡️ 4. Consola de Analista (Back-Office)
Módulo operativo fundamental que simula el día a día de un analista de prevención de fraudes y ciberseguridad.

1. **Cola de Casos Pendientes:** Muestra las transacciones sospechosas suspendidas por el motor híbrido (Reglas o IA).
2. **Expediente del Caso:** Al seleccionar un caso, se despliega el historial del cliente, montos, la regla o modelo que gatilló la alerta, y los detalles técnicos de los balances del dispositivo e IP en formato JSON.
3. **Acciones Operativas:** El analista puede ejecutar tres tipos de resoluciones en tiempo real:
   * `✅ APROBAR TRANSACCIÓN (Liberar Fondos)`: Aprueba la transacción y la remueve de la cola.
   * `⛔ CONFIRMAR FRAUDE (Bloquear Cuenta)`: Bloquea definitivamente la cuenta del atacante.
   * `📑 SOLICITAR DOCUMENTACIÓN ADICIONAL`: Mantiene el caso retenido solicitando verificación de identidad física.
4. **Historial de Resoluciones:** Tabla inferior que registra cada decisión tomada por el analista durante la sesión actual de forma cronológica.

---

## 🔴 3. Cómo Verificar si es Fraude o No (Criterios de Decisión)

El sistema evalúa el riesgo y devuelve tres tipos de información visual inmediata para verificar el fraude:

### A. Semáforo de Alertas y Estado
* **🟢 LEGÍTIMA (Aprobada):** La transacción presenta un comportamiento contable regular. No hay indicios de manipulación de saldos. El flujo de pago se aprueba de inmediato.
* **🔴 FRAUDE DETECTADO (Suspendida):** El modelo ha detectado patrones anómalos o inconsistencias contables extremas. La transacción es bloqueada en la pasarela de pagos, se activa el protocolo de seguridad y se solicita verificación multifactorial (SMS) al usuario legítimo.

### B. Puntuación de Riesgo (Score de Riesgo)
* Se presenta en un rango de **0.00% a 100.00%**.
* Un score superior al **50.00%** activa la alerta de fraude.
* Los modelos supervisados tienden a ser deterministas (devuelven casi 0% o casi 100%), mientras que el Autoencoder e Isolation Forest devuelven valores continuos de riesgo basados en la magnitud de la anomalía física del comportamiento financiero.

### C. Firmas del Fraude (Ingeniería de Características clave)
Los modelos se basan principalmente en la **inconsistencia contable**, la cual representa la firma del fraude:
1. **`error_balance_orig` (Inconsistencia de Origen):** Es la diferencia matemática de saldo: `newbalanceOrig + amount - oldbalanceOrg`. En una transacción normal, la cuenta de origen disminuye exactamente en la cantidad enviada, por lo que el error es 0. En fraudes de vaciado de cuentas, los atacantes suelen forzar que el balance quede en 0 o alterar montos, creando una discrepancia contable.
2. **`error_balance_dest` (Inconsistencia de Destino):** La discrepancia en el destino: `oldbalanceDest + amount - newbalanceDest`. Si el saldo del receptor no aumenta exactamente en el monto enviado, es una alerta grave de error contable.
3. **`es_extraccion_alta`:** Operaciones de tipo `TRANSFER` o `CASH_OUT` con montos que superan el percentil 95 global del dataset histórico (RD$535,033.57).

---

## 🧠 4. Cómo Funciona cada Modelo de Detección

### 🌲 1. Random Forest (Bosque Aleatorio)
* **Tipo:** Aprendizaje Supervisado.
* **Cómo funciona:** Crea una gran cantidad de árboles de decisión independientes durante el entrenamiento (en este caso, 100 árboles). Cada árbol evalúa de forma independiente las variables de la transacción (saldos, montos, inconsistencias). Para clasificar una transacción, cada árbol vota (fraude o legítimo) y la aplicación asigna el resultado con la mayoría de votos.
* **Desempeño:** Es el modelo con mejor precisión general (**F1-Score: 99.84%** y **Recall: 99.92%**). Detecta casi a la perfección los fraudes gracias a que las reglas contables de vaciado son muy marcadas y constantes.

### 🚀 2. XGBoost (Extreme Gradient Boosting)
* **Tipo:** Aprendizaje Supervisado.
* **Cómo funciona:** Entrena árboles de decisión de forma secuencial, donde cada nuevo árbol aprende y corrige los errores de predicción del árbol anterior. Aplica optimización de gradiente y regularización para no memorizar los datos (evitar sobreajuste).
* **Desempeño:** Similar al Random Forest (**F1-Score: 99.76%**). Es extremadamente rápido para inferencias en tiempo real y muy eficiente en memoria, lo que lo hace el modelo ideal para producción en arquitecturas en la nube.

### 🤖 3. Autoencoder (Red Neuronal Profunda)
* **Tipo:** Aprendizaje Semi-supervisado (Deep Learning).
* **Cómo funciona:** Es una red neuronal artificial entrenada **únicamente con transacciones legítimas**. Su arquitectura consta de dos partes:
  1. *Codificador:* Comprime la transacción de 12 dimensiones a un espacio latente pequeño de 4 neuronas.
  2. *Decodificador:* Intenta reconstruir los datos originales de 12 dimensiones a partir del espacio comprimido.
* **Criterio de Fraude:** Al procesar una transacción de fraude (que tiene patrones de saldos incoherentes), la red no sabe cómo reconstruirla de forma correcta porque nunca vio fraudes durante su entrenamiento. Esto genera un **error de reconstrucción alto**. Si este error supera un umbral crítico (`umbral_config`), la transacción se clasifica como fraude.
* **Desempeño:** F1-Score modesto (**15.20%**), pero es el único modelo capaz de detectar **fraudes de "Día Cero"** (nuevas tipologías de fraudes que nunca se usaron para entrenar los modelos supervisados).

### 📐 4. Isolation Forest (Bosque de Aislamiento)
* **Tipo:** Aprendizaje No Supervisado.
* **Cómo funciona:** En lugar de aprender lo que es normal o fraude, este algoritmo busca aislar físicamente los datos anómalos. Selecciona características de forma aleatoria y corta los datos en rangos aleatorios. Las transacciones fraudulentas (anomalías) se encuentran en zonas dispersas y requieren muy pocos cortes para ser aisladas en comparación con los puntos normales que están agrupados en alta densidad.
* **Desempeño:** Es el modelo más débil en este escenario de datos tabulares estructurados, con un F1-Score muy bajo (**0.77%**) y una alta tasa de falsos negativos (**99.59%**). Esto demuestra que las anomalías contables simuladas son sutiles en variables individuales y requieren la combinación multidimensional de clasificadores supervisados para ser detectadas correctamente.
