# scripts/regenerar_importancia_y_correlacion_limpias.py
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
EVIDENCIA_DIR = ROOT_DIR / 'evidencia'
ANEXOS_DIR = ROOT_DIR / 'anexos_imagenes'
PLAN_ANEXOS_DIR = ROOT_DIR / 'plancorreccion' / 'imagenes_anexos'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

features_cols = [
    'amount', 'oldbalanceOrg', 'oldbalanceDest',
    'ratio_monto_saldo', 'dif_saldo_monto', 'dest_saldo_cero',
    'es_extraccion_alta', 'hora_dia', 'es_madrugada', 'type_TRANSFER'
]

features_nombres_legibles = {
    'amount': 'Monto Solicitado (amount)',
    'oldbalanceOrg': 'Saldo Previo Emisor (oldbalanceOrg)',
    'oldbalanceDest': 'Saldo Previo Receptor (oldbalanceDest)',
    'ratio_monto_saldo': 'Ratio Monto / Saldo Emisor',
    'dif_saldo_monto': 'Diferencia Saldo - Monto',
    'dest_saldo_cero': 'Cuenta Destino Saldo Cero',
    'es_extraccion_alta': 'Extracción Alto Monto (>P95)',
    'hora_dia': 'Hora del Día (step % 24)',
    'es_madrugada': 'Horario Madrugada (1-4 AM)',
    'type_TRANSFER': 'Tipo Transferencia (TRANSFER)'
}

def regenerar_importancia_features():
    print("1. Regenerando fig_importancia_features.png con las 10 variables pre-autorización...")
    rf = joblib.load(str(MODELS_DIR / 'random_forest.pkl'))
    xgb = joblib.load(str(MODELS_DIR / 'xgboost.pkl'))

    df_rf = pd.DataFrame({
        'Característica': [features_nombres_legibles[c] for c in features_cols],
        'Importancia': rf.feature_importances_
    }).sort_values(by='Importancia', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(df_rf['Característica'], df_rf['Importancia'], color='#3b82f6', alpha=0.85, height=0.6)
    ax.set_xlabel('Importancia Relativa (Gini Importance / MDI)', fontsize=12, fontweight='bold')
    ax.set_title('Importancia de Características Pre-Autorización — Random Forest\n(Sin Fuga de Información)', fontsize=13, fontweight='bold', pad=15)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.4f} ({w*100:.1f}%)',
                va='center', fontsize=10, fontweight='bold', color='#1e293b')

    ax.set_xlim(0, max(df_rf['Importancia']) * 1.25)
    plt.tight_layout()

    rutas = [
        EVIDENCIA_DIR / 'graficos_shap' / 'fig_importancia_features.png',
        ANEXOS_DIR / 'fig_importancia_features.png',
        PLAN_ANEXOS_DIR / 'fig_importancia_features.png'
    ]
    for r in rutas:
        r.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(r))
        print(f"  [OK] Guardado en {r}")
    plt.close()

def regenerar_matriz_correlacion():
    print("\n2. Regenerando fig_correlacion.png con las 10 variables pre-autorización...")
    X_train = np.load(str(DATA_DIR / 'X_train.npy'))
    y_train = np.load(str(DATA_DIR / 'y_train.npy'))

    # Submuestra representativa para correlación rápida y exacta
    np.random.seed(42)
    idx_sample = np.random.choice(len(X_train), min(100000, len(X_train)), replace=False)
    
    df_corr = pd.DataFrame(X_train[idx_sample], columns=features_cols)
    df_corr['isFraud'] = y_train[idx_sample]

    nombres_cortos = [
        'amount', 'oldbalOrg', 'oldbalDest',
        'ratio_monto', 'dif_saldo', 'dest_cero',
        'extr_alta', 'hora_dia', 'madrugada', 'TRANSFER', 'isFraud'
    ]
    df_corr.columns = nombres_cortos

    corr_matrix = df_corr.corr(method='spearman')

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax,
        annot_kws={'size': 9, 'weight': 'bold'}
    )
    ax.set_title('Mapa de Correlación (Spearman) — 10 Variables Pre-Autorización y Fraude\n(Sin Fuga de Información)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10, fontweight='bold')
    plt.tight_layout()

    rutas = [
        EVIDENCIA_DIR / 'graficos_shap' / 'fig_correlacion.png',
        ANEXOS_DIR / 'fig_correlacion.png',
        PLAN_ANEXOS_DIR / 'fig_correlacion.png'
    ]
    for r in rutas:
        r.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(r))
        print(f"  [OK] Guardado en {r}")
    plt.close()

if __name__ == '__main__':
    regenerar_importancia_features()
    regenerar_matriz_correlacion()
    print("\n[ÉXITO] Ambas figuras han sido regeneradas con las características limpias.")
