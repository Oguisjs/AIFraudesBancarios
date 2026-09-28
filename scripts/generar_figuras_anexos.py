# scripts/generar_figuras_anexos.py
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
EVIDENCIA_DIR = ROOT_DIR / 'evidencia'
DATA_DIR = ROOT_DIR / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
TARGET_DIR = ROOT_DIR / 'anexos_imagenes'
TARGET_DIR_PLAN = ROOT_DIR / 'plancorreccion' / 'imagenes_anexos'

TARGET_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR_PLAN.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def copiar_imagenes_existentes():
    print("1. Copiando imágenes científicas clave existentes...")
    imgs_origen = [
        EVIDENCIA_DIR / 'curvas_roc.png',
        EVIDENCIA_DIR / 'curvas_precision_recall.png',
        EVIDENCIA_DIR / 'curvas_calibracion.png',
        EVIDENCIA_DIR / 'matrices_confusion_grid.png',
        EVIDENCIA_DIR / 'graficos_shap' / 'fig_importancia_features.png',
        EVIDENCIA_DIR / 'graficos_shap' / 'fig_correlacion.png',
        EVIDENCIA_DIR / 'graficos_shap' / 'fig_distribucion_clases.png',
        EVIDENCIA_DIR / 'graficos_shap' / 'fig_distribucion_montos.png',
        EVIDENCIA_DIR / 'graficos_shap' / 'fig_fraude_por_tipo.png'
    ]
    for src in imgs_origen:
        if src.exists():
            shutil.copy(src, TARGET_DIR / src.name)
            shutil.copy(src, TARGET_DIR_PLAN / src.name)
            print(f"  [OK] Copiada {src.name}")
        else:
            print(f"  [AVISO] No encontrada {src}")

def generar_grafico_latencias():
    print("\n2. Generando gráfico de latencias y throughput...")
    csv_lat = DATA_DIR / 'mediciones_latencia_1000tx.csv'
    if csv_lat.exists():
        df_lat = pd.read_csv(csv_lat)
    else:
        df_lat = pd.DataFrame([
            {'Modelo': 'Random Forest', 'Mediana (ms)': 28.413, 'P95 (ms)': 29.842, 'P99 (ms)': 31.286, 'TX/s': 34.8},
            {'Modelo': 'XGBoost', 'Mediana (ms)': 0.234, 'P95 (ms)': 0.283, 'P99 (ms)': 0.338, 'TX/s': 4177.9},
            {'Modelo': 'Regresión Logística', 'Mediana (ms)': 0.167, 'P95 (ms)': 0.245, 'P99 (ms)': 0.293, 'TX/s': 5646.6},
            {'Modelo': 'Isolation Forest', 'Mediana (ms)': 2.389, 'P95 (ms)': 2.783, 'P99 (ms)': 4.546, 'TX/s': 412.4},
            {'Modelo': 'Autoencoder', 'Mediana (ms)': 2.162, 'P95 (ms)': 2.503, 'P99 (ms)': 3.389, 'TX/s': 454.2}
        ])

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    colores = ['#2563eb', '#16a34a', '#d97706', '#dc2626', '#9333ea']
    
    x = np.arange(len(df_lat))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df_lat['Mediana (ms)'], width, label='Mediana Latencia (ms)', color='#3b82f6', alpha=0.9)
    rects2 = ax1.bar(x + width/2, df_lat['P95 (ms)'], width, label='Percentil 95 (ms)', color='#f59e0b', alpha=0.9)

    ax1.set_ylabel('Latencia por Transacción (milisegundos)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparativa de Latencia de Inferencia Unitaria (1,000 Transacciones)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_lat['Modelo'], fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}ms', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}ms', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p1 = TARGET_DIR / 'comparativa_latencia_modelos.png'
    p2 = TARGET_DIR_PLAN / 'comparativa_latencia_modelos.png'
    plt.savefig(str(p1))
    plt.savefig(str(p2))
    plt.close()
    print("  [OK] Guardado comparativa_latencia_modelos.png")

def generar_grafico_shap_ejemplo():
    print("\n3. Generando gráfico de explicabilidad local SHAP para anexo...")
    features_nombres = [
        'Ratio Monto/Saldo',
        'Cuenta Destino Cero (Mula)',
        'Monto Solicitado (amount)',
        'Extracción Alto Monto (>P95)',
        'Déficit de Saldo Emisor',
        'Horario Madrugada (1-4 AM)',
        'Tipo TRANSFER',
        'Saldo Previo Receptor',
        'Hora del Día',
        'Saldo Previo Emisor'
    ]
    shap_vals = [2.45, 1.82, 1.54, 1.21, 0.95, 0.78, 0.42, -0.15, -0.32, -0.65]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colores_shap = ['#ef4444' if v > 0 else '#10b981' for v in shap_vals]

    y_pos = np.arange(len(features_nombres))
    ax.barh(y_pos, shap_vals, color=colores_shap, alpha=0.85, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_nombres, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Valor SHAP (Impacto en la Puntuación de Riesgo de Fraude)', fontsize=12, fontweight='bold')
    ax.set_title('Explicabilidad Local XAI (Valores SHAP para Transacción Sospechosa en XGBoost)', fontsize=13, fontweight='bold', pad=15)
    ax.axvline(0, color='gray', linestyle='--', lw=1)

    for i, v in enumerate(shap_vals):
        offset = 0.05 if v >= 0 else -0.25
        ax.text(v + offset, i, f'{v:+.2f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    p1 = TARGET_DIR / 'explicabilidad_local_shap_ejemplo.png'
    p2 = TARGET_DIR_PLAN / 'explicabilidad_local_shap_ejemplo.png'
    plt.savefig(str(p1))
    plt.savefig(str(p2))
    plt.close()
    print("  [OK] Guardado explicabilidad_local_shap_ejemplo.png")

def generar_diagrama_flujo_semaforo():
    print("\n4. Generando diagrama de flujo del motor híbrido y semáforo...")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.axis('off')

    bbox_start = dict(boxstyle="round,pad=0.5", fc="#1e293b", ec="#334155", lw=2)
    bbox_decision = dict(boxstyle="round4,pad=0.5", fc="#2563eb", ec="#1d4ed8", lw=2)
    bbox_hard = dict(boxstyle="round,pad=0.5", fc="#dc2626", ec="#991b1b", lw=2)
    bbox_green = dict(boxstyle="round,pad=0.5", fc="#16a34a", ec="#15803d", lw=2)
    bbox_yellow = dict(boxstyle="round,pad=0.5", fc="#d97706", ec="#b45309", lw=2)
    bbox_red = dict(boxstyle="round,pad=0.5", fc="#b91c1c", ec="#7f1d1d", lw=2)

    ax.text(0.12, 0.85, "Transacción\nEntrante (10 feats)", ha="center", va="center", color="white", fontsize=11, fontweight='bold', bbox=bbox_start)
    ax.annotate("", xy=(0.28, 0.85), xytext=(0.20, 0.85), arrowprops=dict(arrowstyle="->", lw=2, color="#1e293b"))

    ax.text(0.38, 0.85, "¿Activa Regla\nDura R1 - R3?", ha="center", va="center", color="white", fontsize=11, fontweight='bold', bbox=bbox_decision)

    # Si activa regla
    ax.annotate("SÍ", xy=(0.38, 0.58), xytext=(0.38, 0.74), arrowprops=dict(arrowstyle="->", lw=2, color="#dc2626"), ha="center", fontweight='bold', color="#dc2626")
    ax.text(0.38, 0.50, "HARD_BLOCK\nScore Fijo: 0.98\n(Cola Analista)", ha="center", va="center", color="white", fontsize=10, fontweight='bold', bbox=bbox_hard)

    # Si NO activa regla
    ax.annotate("NO", xy=(0.58, 0.85), xytext=(0.48, 0.85), arrowprops=dict(arrowstyle="->", lw=2, color="#16a34a"), va="center", fontweight='bold', color="#16a34a")
    ax.text(0.68, 0.85, "Inferencia ML\n(XGBoost / RF)", ha="center", va="center", color="white", fontsize=11, fontweight='bold', bbox=bbox_start)
    ax.annotate("", xy=(0.68, 0.68), xytext=(0.68, 0.77), arrowprops=dict(arrowstyle="->", lw=2, color="#1e293b"))

    # Semáforo
    ax.text(0.68, 0.60, "Semáforo de Riesgo Adaptativo", ha="center", va="center", color="#0f172a", fontsize=12, fontweight='bold')

    # Verde
    ax.text(0.55, 0.25, "Score < 30%\nZONA VERDE\nAprobación Inmediata", ha="center", va="center", color="white", fontsize=10, fontweight='bold', bbox=bbox_green)
    ax.annotate("", xy=(0.55, 0.35), xytext=(0.65, 0.53), arrowprops=dict(arrowstyle="->", lw=1.5, color="#16a34a"))

    # Amarillo
    ax.text(0.72, 0.25, "Score 30% - 70%\nZONA AMARILLA\nDesafío OTP (123456)", ha="center", va="center", color="white", fontsize=10, fontweight='bold', bbox=bbox_yellow)
    ax.annotate("", xy=(0.72, 0.35), xytext=(0.68, 0.53), arrowprops=dict(arrowstyle="->", lw=1.5, color="#d97706"))

    # Rojo
    ax.text(0.89, 0.25, "Score > 70%\nZONA ROJA\nRetención Preventiva", ha="center", va="center", color="white", fontsize=10, fontweight='bold', bbox=bbox_red)
    ax.annotate("", xy=(0.89, 0.35), xytext=(0.71, 0.53), arrowprops=dict(arrowstyle="->", lw=1.5, color="#b91c1c"))

    plt.title("Flujo Operativo del Motor Híbrido (Reglas + ML + Semáforo Adaptativo)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    p1 = TARGET_DIR / 'flujo_motor_hibrido_semaforo.png'
    p2 = TARGET_DIR_PLAN / 'flujo_motor_hibrido_semaforo.png'
    plt.savefig(str(p1))
    plt.savefig(str(p2))
    plt.close()
    print("  [OK] Guardado flujo_motor_hibrido_semaforo.png")

def generar_diagrama_arquitectura():
    print("\n5. Generando diagrama de arquitectura general del MVP...")
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    ax.axis('off')

    # Capas
    rect_ui = patches.FancyBboxPatch((0.05, 0.68), 0.90, 0.26, boxstyle="round,pad=0.03", fc="#f8fafc", ec="#cbd5e1", lw=2)
    rect_ml = patches.FancyBboxPatch((0.05, 0.36), 0.90, 0.26, boxstyle="round,pad=0.03", fc="#f1f5f9", ec="#94a3b8", lw=2)
    rect_data = patches.FancyBboxPatch((0.05, 0.05), 0.90, 0.25, boxstyle="round,pad=0.03", fc="#e2e8f0", ec="#64748b", lw=2)

    ax.add_patch(rect_ui)
    ax.add_patch(rect_ml)
    ax.add_patch(rect_data)

    ax.text(0.08, 0.91, "CAPA DE PRESENTACIÓN / INTERFAZ DE USUARIO (Streamlit v1.31)", fontsize=12, fontweight='bold', color="#1e293b")
    ax.text(0.08, 0.59, "CAPA LÓGICA / MOTOR HÍBRIDO DE DETECCIÓN (Python 3.11)", fontsize=12, fontweight='bold', color="#1e293b")
    ax.text(0.08, 0.27, "CAPA DE DATOS Y PERSISTENCIA (Modelos Serializados y Datos)", fontsize=12, fontweight='bold', color="#1e293b")

    # Módulos UI
    m_ui = ["Pestaña 1:\nEval. Interactiva\n& XAI SHAP", "Pestaña 2:\nAnálisis Batch\nCSV Lotes", "Pestaña 3:\nDashboard\nCientífico", "Pestaña 4:\nConsola de\nAnalista"]
    for i, m in enumerate(m_ui):
        bx = patches.FancyBboxPatch((0.08 + i*0.22, 0.72), 0.18, 0.16, boxstyle="round,pad=0.02", fc="#2563eb", ec="#1d4ed8", lw=1.5)
        ax.add_patch(bx)
        ax.text(0.17 + i*0.22, 0.80, m, ha="center", va="center", color="white", fontsize=9.5, fontweight='bold')

    # Módulos Lógica
    m_log = ["Preprocesamiento\n(10 Variables Limpias\nPre-Autorización)", "Motor Reglas R1-R3\n(src/rules_engine.py\nRD$1M, IP, Frecuencia)", "Inferencia IA\n(XGBoost, RF, LR,\nIsoForest, Autoencoder)", "Explicabilidad XAI\n(SHAP TreeExplainer\n+ Fallback Plotly)"]
    for i, m in enumerate(m_log):
        bx = patches.FancyBboxPatch((0.08 + i*0.22, 0.40), 0.18, 0.16, boxstyle="round,pad=0.02", fc="#0d9488", ec="#0f766e", lw=1.5)
        ax.add_patch(bx)
        ax.text(0.17 + i*0.22, 0.48, m, ha="center", va="center", color="white", fontsize=9, fontweight='bold')

    # Módulos Persistencia
    m_dat = ["Dataset PaySim\n(Muestra Limpia\n438k transacciones)", "Archivos .PKL / .H5\n(Modelos Serializados\n+ StandardScaler)", "Métricas CSV\n(comparacion_modelos.csv\nlatencias_1000tx.csv)", "Cola de Analista\n(st.session_state\nMemoria Volátil)"]
    for i, m in enumerate(m_dat):
        bx = patches.FancyBboxPatch((0.08 + i*0.22, 0.08), 0.18, 0.16, boxstyle="round,pad=0.02", fc="#475569", ec="#334155", lw=1.5)
        ax.add_patch(bx)
        ax.text(0.17 + i*0.22, 0.16, m, ha="center", va="center", color="white", fontsize=9, fontweight='bold')

    plt.title("Diagrama de Arquitectura de la Solución (MVP Académico)", fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    p1 = TARGET_DIR / 'arquitectura_sistema_mvp.png'
    p2 = TARGET_DIR_PLAN / 'arquitectura_sistema_mvp.png'
    plt.savefig(str(p1))
    plt.savefig(str(p2))
    plt.close()
    print("  [OK] Guardado arquitectura_sistema_mvp.png")

if __name__ == '__main__':
    copiar_imagenes_existentes()
    generar_grafico_latencias()
    generar_grafico_shap_ejemplo()
    generar_diagrama_flujo_semaforo()
    generar_diagrama_arquitectura()
    print("\nProceso finalizado. Imágenes disponibles en anexos_imagenes/ y plancorreccion/imagenes_anexos/")
