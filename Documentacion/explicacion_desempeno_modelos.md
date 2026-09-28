# 📊 Análisis y Explicación Científica del Desempeño de los Modelos

Este documento presenta una explicación técnica y detallada del **Dashboard de Desempeño Científico de los Modelos** implementados para la detección de fraudes bancarios en el proyecto. 

El análisis comparativo evalúa cuatro algoritmos bajo tres paradigmas de aprendizaje (Supervisado, Semi-supervisado y No supervisado) utilizando el subconjunto de prueba del dataset **PaySim1** (~1.9 millones de transacciones de evaluación, representativas del flujo transaccional real).

---

## 📈 Tabla Comparativa de Métricas

La siguiente tabla resume los resultados cuantitativos obtenidos por cada modelo evaluado:

| Modelo de IA / Algoritmo | Paradigma de Aprendizaje | Precision | Recall (Sensibilidad) | F1-Score | AUC-ROC | PR-AUC | MCC | Falsos Negativos (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Supervisado (Bagging) | **0.9976** | **0.9992** | **0.9984** | **1.0000** | **1.0000** | **0.9983** | **0.08%** |
| **XGBoost** | Supervisado (Boosting) | 0.9960 | **0.9992** | 0.9976 | **1.0000** | **1.0000** | 0.9975 | **0.08%** |
| **Autoencoder** | Semi-Supervisado (Deep Learning) | 0.1020 | 0.2979 | 0.1520 | 0.7936 | 0.0847 | 0.1478 | 70.21% |
| **Isolation Forest** | No Supervisado (Aislamiento) | 0.0820 | 0.0041 | 0.0077 | 0.7772 | 0.0712 | 0.0142 | 99.59% |

---

## 🔍 Definición y Relevancia de Métricas en Fraude Financiero

Para interpretar correctamente la tabla en una defensa de grado o reporte técnico, es crucial entender qué mide cada indicador en el contexto de la ciberseguridad bancaria:

1. **Precision (Precisión):** 
   * **¿Qué mide?:** Del total de alertas de fraude que emite el modelo, cuántas son fraudes reales.
   * **Impacto Bancario:** Una precisión baja (como la de *Isolation Forest* u *Autoencoder*) significa una alta tasa de falsos positivos, lo que se traduce en **bloqueos innecesarios a clientes legítimos** (fricción operativa).
2. **Recall / Exhaustividad (Sensibilidad):**
   * **¿Qué mide?:** Del total de fraudes reales que ocurrieron, qué porcentaje fue detectado por el modelo.
   * **Impacto Bancario:** El recall es crítico. Si es bajo, significa que el fraude está pasando sin ser detectado.
3. **F1-Score:**
   * **¿Qué mide?:** Es la media armónica entre Precision y Recall.
   * **Impacto Bancario:** Al ser un dataset altamente desbalanceado (donde el fraude representa solo el $0.13\%$ de los datos), el F1-Score es una métrica mucho más honesta que el *Accuracy* (exactitud global), ya que penaliza los desbalances extremos entre precisión y exhaustividad.
4. **AUC-ROC (Área Bajo la Curva ROC):**
   * **¿Qué mide?:** Capacidad del modelo para ordenar las transacciones de mayor a menor riesgo a lo largo de todos los umbrales de decisión posibles.
   * **Impacto Bancario:** Un valor de $1.00$ representa una separación perfecta de clases; un valor de $0.5$ representa azar.
5. **PR-AUC (Área Bajo la Curva Precision-Recall):**
   * **¿Qué mide?:** Evalúa la relación precisión-exhaustividad específicamente en la clase minoritaria (fraude).
   * **Impacto Bancario:** > [!IMPORTANT]
     > En datos con desbalanceo severo, **PR-AUC es más exigente e informativa que AUC-ROC**, ya que AUC-ROC se ve artificialmente optimista debido al enorme número de Verdaderos Negativos (transacciones legítimas clasificadas correctamente). Un PR-AUC de $1.00$ en Random Forest y XGBoost es un indicador de su excepcional robustez.
6. **MCC (Coeficiente de Correlación de Matthews):**
   * **¿Qué mide?:** Coeficiente de correlación entre las predicciones del modelo y las etiquetas reales. Oscila entre $-1$ (desacuerdo total), $0$ (clasificación al azar) y $+1$ (predicción perfecta).
   * **Impacto Bancario:** Es considerada la métrica más robusta y rigurosa en clasificación binaria sobre clases desbalanceadas porque balancea las cuatro celdas de la matriz de confusión.
7. **Falsos Negativos (%):**
   * **¿Qué mide?:** El porcentaje de fraudes que pasaron el filtro del modelo y se marcaron como legítimos.
   * **Impacto Bancario:** Es la métrica que representa el **riesgo financiero directo** (pérdidas por contracargos y robos no detectados). El objetivo en banca es llevar este valor al mínimo absoluto.

---

## 🧠 Análisis por Modelo y Paradigma

### 1. Modelos Supervisados: Random Forest y XGBoost (Rendimiento Sobresaliente)
* **Comportamiento:** Ambos clasificadores obtuvieron métricas cercanas a la perfección ($F1 > 99.7\%$, $MCC \approx 0.998$, $Falsos\ Negativos = 0.08\%$).
* **Razón Científica:** 
  * El entrenamiento supervisado con balanceo de clases previo (utilizando técnicas como **SMOTE** para poblar sintéticamente la clase fraudulenta) y una minuciosa **ingeniería de características** facilitó la separación de clases.
  * Variables derivadas como `error_balance_orig` e `error_balance_dest` (que capturan inconsistencias contables inmediatas) y `ratio_monto_saldo` actúan como predictores sumamente deterministas que los árboles de decisión segmentan con precisión quirúrgica.
* **Falsos Negativos:** Reducir la tasa de escape de fraudes al **$0.08\%$** (solo $8$ de cada $10,000$ fraudes escapan al control) cumple con los estándares más estrictos de tolerancia al riesgo de la banca moderna.

### 2. Modelo Semi-Supervisado: Autoencoder (Rendimiento Moderado - Valor Conceptual)
* **Comportamiento:** Obtiene un $F1\text{-Score}$ de $15.20\%$ y una tasa de Falsos Negativos alta ($70.21\%$).
* **Razón Científica:** 
  * Los Autoencoders se entrenan **únicamente con transacciones legítimas** para aprender la distribución de la normalidad. La detección se realiza midiendo el Error Cuadrático Medio de Reconstrucción ($MSE$).
  * Dado que no conocen los patrones específicos del fraude durante el entrenamiento, su precisión numérica es mucho menor que la de los modelos supervisados.
* **Justificación de su Uso (Detección de Día Cero):** 
  * > [!TIP]
    > A pesar de sus métricas numéricas más bajas en el dataset de prueba, el Autoencoder es de gran valor operativo. Los modelos supervisados (RF y XGBoost) sufren de obsolescencia ante nuevos esquemas de fraude inéditos en el set de entrenamiento. El Autoencoder, al buscar discrepancias con la "normalidad", es el **único modelo capaz de detectar anomalías inéditas o de "día cero"**.

### 3. Modelo No Supervisado: Isolation Forest (Rendimiento Insuficiente)
* **Comportamiento:** Desempeño extremadamente pobre ($F1 = 0.77\%$, $Recall = 0.41\%$ y $Falsos\ Negativos = 99.59\%$).
* **Razón Científica:** 
  * Isolation Forest aísla anomalías basándose en cortes aleatorios en el espacio de características, asumiendo que los valores anómalos están "aislados" y alejados geométricamente en variables individuales.
  * Sin embargo, en el fraude bancario representativo de PaySim, los fraudes no son necesariamente transacciones con montos infinitamente grandes o valores atípicos sencillos. Los fraudes ocurren dentro de **relaciones lógicas específicas** entre variables (por ejemplo, vaciar exactamente una cuenta y dejar el balance de destino inconsistente). Isolation Forest no logra capturar estas interacciones lógicas complejas sin supervisión, lo que lo descarta como modelo de producción primario para esta tarea.

---

## 🎓 Tips clave para la Defensa de la Tesis

Si los jurados te cuestionan sobre esta tabla, enfoca tus argumentos en:
1. **La justificación del desbalanceo:** Destaca que se usó el **F1-Score** y el **MCC** como métricas rectoras en lugar del *Accuracy* clásico para evitar el sesgo de la clase mayoritaria.
2. **La complementariedad del motor híbrido:** Los modelos supervisados sirven como primera línea de defensa ultra-precisa para fraudes conocidos. El Autoencoder se incluye como una segunda capa orientada a la adaptabilidad y anomalías emergentes no mapeadas.
3. **El impacto operativo de la ingeniería de características:** Explica que el excelente desempeño de XGBoost y RF no es fruto del azar, sino de la incorporación de variables contables calculadas a nivel de negocio (`error_balance_orig`), las cuales facilitan la tarea de clasificación del algoritmo.
