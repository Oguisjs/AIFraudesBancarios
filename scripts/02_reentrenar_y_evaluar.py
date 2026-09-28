# scripts/02_reentrenar_y_evaluar.py
import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

try:
    import psutil
    def medir_recursos():
        return psutil.virtual_memory().used / (1024 ** 3)
except ImportError:
    def medir_recursos():
        return 0.0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras

# Métricas
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
EVIDENCIA_DIR = ROOT_DIR / 'evidencia'

EVIDENCIA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("FASE 3 Y 4: REENTRENAMIENTO MULTIMODELO Y EVALUACIÓN CIENTÍFICA")
    print("             (SIN FUGA DE INFORMACIÓN - 10 FEATURES PRE-AUTORIZACIÓN)")
    print("=" * 80)

    # 1. Cargar datos procesados
    print("\n1. Cargando datos desde data/processed/...")
    X_train = np.load(str(DATA_DIR / 'X_train.npy'))
    X_train_sc = np.load(str(DATA_DIR / 'X_train_sc.npy'))
    y_train = np.load(str(DATA_DIR / 'y_train.npy'))

    X_val = np.load(str(DATA_DIR / 'X_val.npy'))
    X_val_sc = np.load(str(DATA_DIR / 'X_val_sc.npy'))
    y_val = np.load(str(DATA_DIR / 'y_val.npy'))

    X_test = np.load(str(DATA_DIR / 'X_test.npy'))
    X_test_sc = np.load(str(DATA_DIR / 'X_test_sc.npy'))
    y_test = np.load(str(DATA_DIR / 'y_test.npy'))

    print(f"Train: {X_train.shape} | Fraudes: {int(y_train.sum())} ({y_train.mean()*100:.2f}%)")
    print(f"Val:   {X_val.shape} | Fraudes: {int(y_val.sum())} ({y_val.mean()*100:.2f}%)")
    print(f"Test:  {X_test.shape} | Fraudes: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)")

    registros_rendimiento = []
    modelos = {}
    predicciones = {}
    scores = {}

    # -------------------------------------------------------------
    # MODELO 0: REGRESIÓN LOGÍSTICA (LÍNEA BASE / BASELINE)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">> Entrenando Modelo Base: Regresión Logística...")
    m0_ram_i = medir_recursos()
    t0 = time.perf_counter()
    modelo_lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    modelo_lr.fit(X_train_sc, y_train)
    t_lr = time.perf_counter() - t0
    m0_ram_f = medir_recursos()

    y_pred_lr = modelo_lr.predict(X_test_sc)
    score_lr = modelo_lr.predict_proba(X_test_sc)[:, 1]

    modelos['Regresión Logística'] = modelo_lr
    predicciones['Regresión Logística'] = y_pred_lr
    scores['Regresión Logística'] = score_lr
    registros_rendimiento.append({
        'Modelo': 'Regresión Logística',
        'Tiempo_Entrenamiento_s': round(t_lr, 2),
        'RAM_Pico_GB': round(m0_ram_f, 2)
    })
    joblib.dump(modelo_lr, str(MODELS_DIR / 'logistic_regression.pkl'))
    print(f"Regresión Logística completada en {t_lr:.2f} s. Modelo guardado.")

    # -------------------------------------------------------------
    # MODELO 1: RANDOM FOREST
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">> Entrenando Random Forest...")
    m1_ram_i = medir_recursos()
    t0 = time.perf_counter()
    modelo_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    modelo_rf.fit(X_train, y_train)
    t_rf = time.perf_counter() - t0
    m1_ram_f = medir_recursos()

    y_pred_rf = modelo_rf.predict(X_test)
    score_rf = modelo_rf.predict_proba(X_test)[:, 1]

    modelos['Random Forest'] = modelo_rf
    predicciones['Random Forest'] = y_pred_rf
    scores['Random Forest'] = score_rf
    registros_rendimiento.append({
        'Modelo': 'Random Forest',
        'Tiempo_Entrenamiento_s': round(t_rf, 2),
        'RAM_Pico_GB': round(m1_ram_f, 2)
    })
    joblib.dump(modelo_rf, str(MODELS_DIR / 'random_forest.pkl'))
    print(f"Random Forest completado en {t_rf:.2f} s. Modelo guardado.")

    # -------------------------------------------------------------
    # MODELO 2: XGBOOST
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">> Entrenando XGBoost...")
    m2_ram_i = medir_recursos()
    t0 = time.perf_counter()
    modelo_xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=10.0,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    modelo_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    t_xgb = time.perf_counter() - t0
    m2_ram_f = medir_recursos()

    y_pred_xgb = modelo_xgb.predict(X_test)
    score_xgb = modelo_xgb.predict_proba(X_test)[:, 1]

    modelos['XGBoost'] = modelo_xgb
    predicciones['XGBoost'] = y_pred_xgb
    scores['XGBoost'] = score_xgb
    registros_rendimiento.append({
        'Modelo': 'XGBoost',
        'Tiempo_Entrenamiento_s': round(t_xgb, 2),
        'RAM_Pico_GB': round(m2_ram_f, 2)
    })
    joblib.dump(modelo_xgb, str(MODELS_DIR / 'xgboost.pkl'))
    print(f"XGBoost completado en {t_xgb:.2f} s. Modelo guardado.")

    # -------------------------------------------------------------
    # MODELO 3: ISOLATION FOREST
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">> Entrenando Isolation Forest...")
    m3_ram_i = medir_recursos()
    t0 = time.perf_counter()
    modelo_if = IsolationForest(
        n_estimators=200,
        contamination=0.0187,
        max_samples=0.8,
        random_state=42,
        n_jobs=-1
    )
    # Se entrena con los datos no balanceados o con train
    modelo_if.fit(X_train)
    t_if = time.perf_counter() - t0
    m3_ram_f = medir_recursos()

    pred_raw_if = modelo_if.predict(X_test)
    y_pred_if = np.where(pred_raw_if == -1, 1, 0)
    raw_scores_if = -modelo_if.score_samples(X_test)
    # Normalizar score a [0, 1]
    s_min, s_max = raw_scores_if.min(), raw_scores_if.max()
    score_if = (raw_scores_if - s_min) / (s_max - s_min + 1e-8)

    modelos['Isolation Forest'] = modelo_if
    predicciones['Isolation Forest'] = y_pred_if
    scores['Isolation Forest'] = score_if
    registros_rendimiento.append({
        'Modelo': 'Isolation Forest',
        'Tiempo_Entrenamiento_s': round(t_if, 2),
        'RAM_Pico_GB': round(m3_ram_f, 2)
    })
    joblib.dump(modelo_if, str(MODELS_DIR / 'isolation_forest.pkl'))
    print(f"Isolation Forest completado en {t_if:.2f} s. Modelo guardado.")

    # -------------------------------------------------------------
    # MODELO 4: AUTOENCODER PROFUNDO
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">> Entrenando Autoencoder Profundo (10 dimensiones)...")
    m4_ram_i = medir_recursos()
    t0 = time.perf_counter()

    # Entrenar solo con transacciones legítimas
    X_train_legitimas = X_train_sc[y_train == 0]
    n_features = X_train_sc.shape[1]

    tf.random.set_seed(42)
    entrada = keras.Input(shape=(n_features,))
    cod = keras.layers.Dense(24, activation='relu')(entrada)
    cod = keras.layers.Dense(12, activation='relu')(cod)
    cod = keras.layers.Dense(6,  activation='relu')(cod)  # Bottleneck
    dec = keras.layers.Dense(12, activation='relu')(cod)
    dec = keras.layers.Dense(24, activation='relu')(dec)
    salida = keras.layers.Dense(n_features, activation='linear')(dec)

    autoencoder = keras.Model(inputs=entrada, outputs=salida)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True
    )
    autoencoder.fit(
        X_train_legitimas, X_train_legitimas,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    t_ae = time.perf_counter() - t0
    m4_ram_f = medir_recursos()

    # Calcular umbral sobre validación normal
    X_val_legitimas = X_val_sc[y_val == 0]
    recon_val = autoencoder.predict(X_val_legitimas, verbose=0)
    err_val = np.mean(np.power(X_val_legitimas - recon_val, 2), axis=1)
    umbral_ae = float(np.percentile(err_val, 95))

    # Inferencia en test
    recon_test = autoencoder.predict(X_test_sc, verbose=0)
    errores_test = np.mean(np.power(X_test_sc - recon_test, 2), axis=1)
    y_pred_ae = (errores_test > umbral_ae).astype(int)
    # Normalizar error continuo sin clipping para preservar orden de ranking
    ae_min, ae_max = errores_test.min(), errores_test.max()
    score_ae = (errores_test - ae_min) / (ae_max - ae_min + 1e-8)

    modelos['Autoencoder'] = autoencoder
    predicciones['Autoencoder'] = y_pred_ae
    scores['Autoencoder'] = score_ae
    registros_rendimiento.append({
        'Modelo': 'Autoencoder',
        'Tiempo_Entrenamiento_s': round(t_ae, 2),
        'RAM_Pico_GB': round(m4_ram_f, 2)
    })

    autoencoder.save(str(MODELS_DIR / 'autoencoder.h5'))
    joblib.dump({'umbral': umbral_ae}, str(MODELS_DIR / 'autoencoder_config.pkl'))
    print(f"Autoencoder completado en {t_ae:.2f} s. Umbral: {umbral_ae:.6f}. Guardado.")

    # -------------------------------------------------------------
    # 5. CÁLCULO DE MÉTRICAS CIENTÍFICAS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. EVALUACIÓN COMPARATIVA DEFINITIVA (CONJUNTO DE PRUEBA: 65,824 REGISTROS)")
    print("=" * 80)

    filas_metricas = []
    filas_detalladas = []

    for nombre in ['Regresión Logística', 'Random Forest', 'XGBoost', 'Isolation Forest', 'Autoencoder']:
        yp = predicciones[nombre]
        sc = scores[nombre]

        cm = confusion_matrix(y_test, yp)
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_test, yp, zero_division=0)
        rec = recall_score(y_test, yp, zero_division=0)
        f1 = f1_score(y_test, yp, zero_division=0)
        auc = roc_auc_score(y_test, sc)
        prauc = average_precision_score(y_test, sc)
        mcc = matthews_corrcoef(y_test, yp)
        brier = brier_score_loss(y_test, sc)

        fn_pct = (fn / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0

        filas_metricas.append({
            'Modelo': nombre,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'AUC-ROC': round(auc, 4),
            'PR-AUC': round(prauc, 4),
            'MCC': round(mcc, 4),
            'Brier_Score': round(brier, 4),
            'Falsos_Negativos': int(fn),
            'Falsos_Neg_Pct': round(fn_pct, 2)
        })

        filas_detalladas.append({
            'Modelo': nombre,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn),
            'Total_Fraudes': int(tp + fn),
            'Total_Legitimas': int(tn + fp)
        })

    df_metricas = pd.DataFrame(filas_metricas)
    df_detalles = pd.DataFrame(filas_detalladas)
    df_rendimiento = pd.DataFrame(registros_rendimiento)

    # Unir rendimiento
    df_resumen = df_metricas.merge(df_rendimiento, on='Modelo').merge(df_detalles[['Modelo', 'TP', 'FP', 'TN', 'FN']], on='Modelo')

    print("\nTABLA DE RESULTADOS DEFINITIVOS (SIN FUGA DE INFORMACIÓN):")
    print(df_resumen[['Modelo', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'PR-AUC', 'MCC', 'Falsos_Negativos', 'Falsos_Neg_Pct', 'Tiempo_Entrenamiento_s']].to_string(index=False))

    print("\nMATRICES DE CONFUSIÓN (VALORES ABSOLUTOS):")
    print(df_detalles.to_string(index=False))

    # Guardar en CSV
    df_resumen.to_csv(str(DATA_DIR / 'comparacion_modelos.csv'), index=False)
    df_detalles.to_csv(str(DATA_DIR / 'matrices_confusion_absolutas.csv'), index=False)
    print(f"\n[OK] Métricas guardadas en {DATA_DIR / 'comparacion_modelos.csv'}")

    # -------------------------------------------------------------
    # 6. GENERACIÓN DE GRÁFICOS CIENTÍFICOS PARA TESIS
    # -------------------------------------------------------------
    print("\n6. Generando gráficos de alta resolución en evidencia/...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # A) Curvas ROC
    plt.figure(figsize=(9, 7), dpi=300)
    for nombre in ['Regresión Logística', 'Random Forest', 'XGBoost', 'Isolation Forest', 'Autoencoder']:
        fpr, tpr, _ = roc_curve(y_test, scores[nombre])
        auc_val = roc_auc_score(y_test, scores[nombre])
        plt.plot(fpr, tpr, lw=2, label=f'{nombre} (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Azar (AUC = 0.5000)')
    plt.title('Curvas ROC Comparativas (Sin Fuga de Información)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(EVIDENCIA_DIR / 'curvas_roc.png'))
    plt.close()

    # B) Curvas Precision-Recall
    plt.figure(figsize=(9, 7), dpi=300)
    prevalencia = float(y_test.mean())
    plt.axhline(y=prevalencia, color='gray', linestyle='--', lw=1.5, label=f'Azar / Prevalencia ({prevalencia:.4f})')

    estilos_pr = {
        'Regresión Logística': {'lw': 2, 'ls': '-'},
        'Random Forest': {'lw': 2.5, 'ls': '--'},
        'XGBoost': {'lw': 2, 'ls': '-'},
        'Isolation Forest': {'lw': 2, 'ls': '-'},
        'Autoencoder': {'lw': 2, 'ls': '-'}
    }

    for nombre in ['Regresión Logística', 'Random Forest', 'XGBoost', 'Isolation Forest', 'Autoencoder']:
        p_vals, r_vals, _ = precision_recall_curve(y_test, scores[nombre])
        pr_val = average_precision_score(y_test, scores[nombre])
        st = estilos_pr.get(nombre, {'lw': 2, 'ls': '-'})
        plt.plot(r_vals, p_vals, lw=st['lw'], linestyle=st['ls'], label=f'{nombre} (PR-AUC = {pr_val:.4f})')
    plt.title('Curvas Precision-Recall Comparativas (Sin Fuga de Información)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Recall (Sensibilidad)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(EVIDENCIA_DIR / 'curvas_precision_recall.png'))
    plt.close()

    # C) Matrices de Confusión en Cuadrícula
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()
    nombres_ord = ['Regresión Logística', 'Random Forest', 'XGBoost', 'Isolation Forest', 'Autoencoder']
    for idx, nombre in enumerate(nombres_ord):
        cm = confusion_matrix(y_test, predicciones[nombre])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Legítima', 'Fraude'], yticklabels=['Legítima', 'Fraude'],
                    annot_kws={'size': 12, 'weight': 'bold'})
        axes[idx].set_title(f'Matriz de Confusión: {nombre}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicción', fontsize=10)
        axes[idx].set_ylabel('Real', fontsize=10)
    # Quitar el 6to subplot vacío
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(str(EVIDENCIA_DIR / 'matrices_confusion_grid.png'))
    plt.close()

    # D) Curvas de Calibración (Reliability Diagram)
    plt.figure(figsize=(9, 7), dpi=300)
    for nombre in ['Regresión Logística', 'Random Forest', 'XGBoost']:
        prob_true, prob_pred = calibration_curve(y_test, scores[nombre], n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', lw=2, label=f'{nombre} (Brier: {brier_score_loss(y_test, scores[nombre]):.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectamente Calibrado')
    plt.title('Curvas de Calibración de Probabilidades (Reliability Diagrams)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Probabilidad Media Predicha', fontsize=12)
    plt.ylabel('Fracción de Fraudes Reales', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(EVIDENCIA_DIR / 'curvas_calibracion.png'))
    plt.close()

    # -------------------------------------------------------------
    # 7. ESTUDIO DE VARIABILIDAD (3 SEMILLAS EN XGBOOST Y RF)
    # -------------------------------------------------------------
    print("\n7. Evaluando variabilidad estocástica con semillas (42, 123, 999)...")
    semillas = [42, 123, 999]
    res_semillas = []
    for sem in semillas:
        # XGBoost
        xgb_s = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=10.0, eval_metric='auc', random_state=sem, n_jobs=-1
        )
        xgb_s.fit(X_train, y_train)
        pred_s = xgb_s.predict(X_test)
        sc_s = xgb_s.predict_proba(X_test)[:, 1]
        res_semillas.append({
            'Modelo': 'XGBoost',
            'Semilla': sem,
            'F1-Score': f1_score(y_test, pred_s),
            'AUC-ROC': roc_auc_score(y_test, sc_s),
            'Recall': recall_score(y_test, pred_s)
        })

    df_semillas = pd.DataFrame(res_semillas)
    df_semillas.to_csv(str(DATA_DIR / 'variabilidad_semillas.csv'), index=False)
    print(df_semillas.to_string(index=False))

    print("\n" + "=" * 80)
    print("PROCESO DE REENTRENAMIENTO Y EVALUACIÓN FINALIZADO CON ÉXITO")
    print("=" * 80)

if __name__ == '__main__':
    main()
