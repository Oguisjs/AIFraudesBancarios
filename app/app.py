# app/app.py
# Ejecutar con: streamlit run app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import datetime as _dt
import plotly.express as px
import tensorflow as tf  # type: ignore[import-untyped]
from pathlib import Path
from typing import Any
import sys

# ── Rutas absolutas calculadas a partir de la ubicación del script ──
APP_DIR   = Path(__file__).parent.resolve()
ROOT_DIR  = APP_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'src'))

from rules_engine import RulesEngine

MODELS_DIR    = ROOT_DIR / 'models'
NOTEBOOKS_DIR = ROOT_DIR / 'notebooks'
EVIDENCIA_DIR = ROOT_DIR / 'evidencia' / 'graficos_shap'
DATA_DIR      = ROOT_DIR / 'data' / 'processed'

# ── Configuración de la página ──
st.set_page_config(
    page_title='Sistema de Detección de Fraudes Bancarios - MVP Tesis',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded'
)

UMBRAL_95_ENTRENAMIENTO = 535033.57

# ── Inicializar Estado Global de la Sesión (Cola del Analista y Historial) ──
if 'analyst_queue' not in st.session_state:
    st.session_state.analyst_queue = [
        {
            'id': 'TX-8941',
            'timestamp': '2026-07-23 02:15:10',
            'cliente': 'CLI-4892 (Juan Pérez)',
            'tipo': 'TRANSFER',
            'monto': 1250000.0,
            'score': 0.96,
            'decision_inicial': 'HARD_BLOCK',
            'regla_activada': 'R1: Transferencia nocturna de alto monto (>RD$1M entre 1 AM y 4 AM)',
            'estado': 'PENDIENTE_REVISION',
            'detalles': {
                'oldbalanceOrg': 1300000.0,
                'newbalanceOrig': 50000.0,
                'oldbalanceDest': 0.0,
                'newbalanceDest': 1250000.0,
                'velocidad': 2,
                'riesgo_ip': 85.0
            }
        },
        {
            'id': 'TX-8942',
            'timestamp': '2026-07-23 14:10:05',
            'cliente': 'CLI-1044 (María Gómez)',
            'tipo': 'CASH_OUT',
            'monto': 480000.0,
            'score': 0.88,
            'decision_inicial': 'SUSPENDIDA_ML',
            'regla_activada': 'R2: Dispositivo con reputación de riesgo extremadamente alta (>90%)',
            'estado': 'PENDIENTE_REVISION',
            'detalles': {
                'oldbalanceOrg': 480000.0,
                'newbalanceOrig': 0.0,
                'oldbalanceDest': 10000.0,
                'newbalanceDest': 490000.0,
                'velocidad': 6,
                'riesgo_ip': 92.0
            }
        }
    ]

if 'resolution_history' not in st.session_state:
    st.session_state.resolution_history = []

if 'current_eval' not in st.session_state:
    st.session_state.current_eval = None

rules_engine = RulesEngine()

# ── Estilos CSS Personalizados (Diseño Premium) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        color: white;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    .glass-card-danger {
        background: rgba(239, 68, 68, 0.1);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(239, 68, 68, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(239, 68, 68, 0.1);
        margin-bottom: 20px;
    }

    .glass-card-success {
        background: rgba(16, 185, 129, 0.1);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(16, 185, 129, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.1);
        margin-bottom: 20px;
    }

    .glass-card-warning {
        background: rgba(251, 191, 36, 0.1);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(251, 191, 36, 0.4);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(251, 191, 36, 0.1);
        margin-bottom: 20px;
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #60A5FA, #3B82F6, #1D4ED8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 8px;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 24px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }

    .phone-mockup {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border-radius: 32px;
        padding: 20px 16px;
        border: 3px solid #334155;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        max-width: 340px;
        margin: 16px auto;
        position: relative;
    }
    .phone-mockup::before {
        content: '';
        display: block;
        width: 80px;
        height: 6px;
        background: #334155;
        border-radius: 3px;
        margin: 0 auto 18px auto;
    }
    .sms-bubble {
        background: linear-gradient(135deg, #EF4444, #B91C1C);
        border-radius: 18px 18px 18px 4px;
        padding: 14px 16px;
        color: white;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    .sms-meta {
        color: #64748B;
        font-size: 0.72rem;
        text-align: left;
        margin-top: 6px;
    }
    .phone-status-bar {
        display: flex;
        justify-content: space-between;
        color: #94A3B8;
        font-size: 0.70rem;
        margin-bottom: 14px;
        padding: 0 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelos (se cachean para mejorar el rendimiento) ──
@st.cache_resource
def cargar_modelos():
    modelos = {
        'Random Forest':       joblib.load(str(MODELS_DIR / 'random_forest.pkl')),
        'XGBoost':             joblib.load(str(MODELS_DIR / 'xgboost.pkl')),
        'Regresión Logística': joblib.load(str(MODELS_DIR / 'logistic_regression.pkl')),
        'Isolation Forest':    joblib.load(str(MODELS_DIR / 'isolation_forest.pkl')),
    }
    scaler  = joblib.load(str(MODELS_DIR / 'scaler.pkl'))
    ae      = tf.keras.models.load_model(str(MODELS_DIR / 'autoencoder.h5'), compile=False)
    ae_cfg  = joblib.load(str(MODELS_DIR / 'autoencoder_config.pkl'))
    return modelos, scaler, ae, ae_cfg

modelos: dict = {}
scaler: Any = None
autoencoder: Any = None
ae_cfg: dict = {}
error_msg: str = ""
modelos_cargados: bool = False

try:
    modelos, scaler, autoencoder, ae_cfg = cargar_modelos()
    modelos_cargados = True
except Exception as e:
    modelos_cargados = False
    error_msg = str(e)

# ── Funciones de soporte ──
def preprocesar_transaccion(tipo, monto, saldo_orig_antes, saldo_dest_antes, step=1):
    """Calcula las 10 variables derivadas pre-autorización requeridas por los modelos limpios"""
    ratio = monto / (saldo_orig_antes + 1.0)
    dif_saldo = saldo_orig_antes - monto
    dest_cero = int(saldo_dest_antes == 0.0)
    extraccion = int(tipo in ['TRANSFER', 'CASH_OUT'] and monto > UMBRAL_95_ENTRENAMIENTO)
    hora = step % 24
    es_madrugada = int(1 <= hora <= 4)
    es_transfer = int(tipo == 'TRANSFER')

    features = pd.DataFrame([[
        monto, saldo_orig_antes, saldo_dest_antes,
        ratio, dif_saldo, dest_cero,
        extraccion, hora, es_madrugada, es_transfer
    ]], columns=[
        'amount', 'oldbalanceOrg', 'oldbalanceDest',
        'ratio_monto_saldo', 'dif_saldo_monto', 'dest_saldo_cero',
        'es_extraccion_alta', 'hora_dia', 'es_madrugada', 'type_TRANSFER'
    ])
    return features

def predecir(modelo_nombre, features):
    """Obtiene la predicción y el score de riesgo según el modelo seleccionado"""
    if modelo_nombre == 'Autoencoder':
        features_sc = scaler.transform(features)
        recon = autoencoder.predict(features_sc, verbose=0)
        error = float(np.mean(np.power(features_sc - recon, 2), axis=1)[0])
        umbral = ae_cfg['umbral']
        pred = int(error > umbral)
        score = min(1.0, error / (umbral * 2.0))
    elif modelo_nombre == 'Isolation Forest':
        modelo = modelos[modelo_nombre]
        raw = modelo.predict(features.values)
        pred = int(raw[0] == -1)
        score_raw = float(-modelo.score_samples(features.values)[0])
        score = min(1.0, max(0.0, (score_raw - 0.35) / 0.3))
    elif modelo_nombre == 'Regresión Logística':
        modelo = modelos[modelo_nombre]
        features_sc = scaler.transform(features)
        pred = int(modelo.predict(features_sc)[0])
        score = float(modelo.predict_proba(features_sc)[0][1])
    else:
        modelo = modelos[modelo_nombre]
        pred = int(modelo.predict(features.values)[0])
        score = float(modelo.predict_proba(features.values)[0][1])
    return pred, score

def generar_grafico_shap(modelo_nombre, features_df):
    """Genera la gráfica explicativa SHAP de explicabilidad local XAI"""
    try:
        import shap
        if modelo_nombre in ['XGBoost', 'Random Forest'] and modelo_nombre in modelos:
            modelo = modelos[modelo_nombre]
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(features_df)

            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            elif len(np.array(shap_values).shape) == 3:
                vals = shap_values[0, 0, 1]
            else:
                vals = shap_values[0]

            df_shap = pd.DataFrame({
                'Característica': features_df.columns,
                'Contribución SHAP': vals
            }).sort_values(by='Contribución SHAP', key=abs, ascending=False).head(6)

            fig = px.bar(
                df_shap,
                x='Contribución SHAP',
                y='Característica',
                orientation='h',
                title=f'Explicabilidad Local XAI (Valores SHAP - {modelo_nombre})',
                color='Contribución SHAP',
                color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
                template='plotly_dark'
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            return fig
    except Exception:
        pass

    # Heurística explicativa visual limpia como fallback
    f_vals = features_df.iloc[0].to_dict()
    contribs = {
        'Monto Solicitado': min(1.0, float(f_vals.get('amount', 0)) / 500000.0),
        'Ratio Monto/Saldo': min(1.0, float(f_vals.get('ratio_monto_saldo', 0))),
        'Déficit de Saldo (Sin fondos)': 1.0 if f_vals.get('dif_saldo_monto', 0) < 0 else 0.0,
        'Cuenta Destino Cero (Mula)': float(f_vals.get('dest_saldo_cero', 0)),
        'Extracción Alto Monto (>P95)': float(f_vals.get('es_extraccion_alta', 0)),
        'Horario Madrugada (1-4 AM)': float(f_vals.get('es_madrugada', 0))
    }
    df_fb = pd.DataFrame({
        'Característica': list(contribs.keys()),
        'Contribución Estimada': list(contribs.values())
    }).sort_values(by='Contribución Estimada', ascending=True)

    fig = px.bar(
        df_fb,
        x='Contribución Estimada',
        y='Característica',
        orientation='h',
        title=f'Explicabilidad Local de Factores de Riesgo Pre-Autorización ({modelo_nombre})',
        color='Contribución Estimada',
        color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
        template='plotly_dark'
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_sms_alert(celular: str, monto: float, tipo: str, score: float) -> None:
    """Renderiza mockup visual de SMS de alerta"""
    hora_actual = _dt.datetime.now().strftime('%H:%M')
    fecha_actual = _dt.datetime.now().strftime('%d/%m/%Y')
    st.markdown(f"""
    <div class='phone-mockup'>
        <div class='phone-status-bar'>
            <span>{hora_actual}</span>
            <span>📶 ▐▐▐▐ 🔋</span>
        </div>
        <div style='text-align:center; color:#94A3B8; font-size:0.72rem; margin-bottom:12px;'>
            📱 {celular}
        </div>
        <div class='sms-bubble'>
            <strong>🏦 BancoSeguro RD &mdash; ALERTA DE SEGURIDAD</strong><br><br>
            Se ha detectado una transacción sospechosa con un nivel de riesgo del
            <strong>{score*100:.1f}%</strong>.<br><br>
            Tipo: <strong>{tipo}</strong> | Monto: <strong>RD${monto:,.2f}</strong><br><br>
            Operación <strong>SUSPENDIDA PREVENTIVAMENTE</strong> y enviada al analista de fraudes.
        </div>
        <div class='sms-meta'>✔✔ Entregado · {hora_actual} · {fecha_actual}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Interfaz Sidebar ──
with st.sidebar:
    st.image('https://cdn-icons-png.flaticon.com/512/2830/2830284.png', width=90)
    st.markdown("## Control de IA & Reglas")
    st.info("Sistema híbrido: Reglas Deterministas + 4 Algoritmos de Machine Learning.")
    if modelos_cargados:
        st.success("✅ Modelos en disco listos")
    else:
        st.error(f"❌ Error al cargar modelos: {error_msg}")

    st.divider()
    st.markdown("### Contexto Digital & Dispositivo")
    velocidad_trans_1h = st.slider('Frecuencia Transacciones (1h)', 1, 10, 1,
                                   help="Frecuencia atípica activa Regla R3 si es > 5 ops/hora.")
    riesgo_dispositivo_ip = st.slider('Score Riesgo Dispositivo/IP (%)', 0.0, 100.0, 15.0,
                                      help="Riesgo de IP o reputación de dispositivo. >90% activa Regla R2.")
    
    st.divider()
    celular_titular = st.text_input('Celular del Titular', value='809-555-0199')

# ── Contenido Principal ──
st.markdown('<div class="title-gradient">MVP Académico: IA en Detección de Fraudes Bancarios</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sustentación de Tesis &mdash; Motor Híbrido, Explicabilidad XAI (SHAP) y Consola de Analista</div>', unsafe_allow_html=True)

# Pestañas del Sistema
tab1, tab2, tab3, tab4 = st.tabs([
    '1. Evaluación en Tiempo Real & XAI',
    '2. Análisis Masivo por Lotes (CSV)',
    '3. Dashboard Científico & Métricas',
    '4. Consola de Analista (Back-Office)'
])

# ══════════════════════════════════════════
# PESTAÑA 1: EVALUACIÓN EN TIEMPO REAL & XAI
# ══════════════════════════════════════════
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Simulador de Transacciones Bancarias (Motor Híbrido)")
    st.write("Ingrese las variables transaccionales y de contexto para ejecutar el motor de reglas e inferencia de IA.")
    
    col1, col2 = st.columns(2)
    with col1:
        tipo = st.selectbox('Tipo de Transacción', ['TRANSFER', 'CASH_OUT'])
        monto = st.number_input('Monto de Transacción (RD$)', min_value=0.0, value=1200000.0, step=50000.0)
        step = st.number_input('Hora del Día (1 - 24)', min_value=1, max_value=24, value=2,
                               help="Madrugada (1 AM - 4 AM) combinada con alto monto activa la Regla R1.")
        modelo_sel = st.selectbox('Modelo de IA Seleccionado', ['XGBoost', 'Random Forest', 'Regresión Logística', 'Isolation Forest', 'Autoencoder'])
        
    with col2:
        sb_orig_a = st.number_input('Saldo Cuenta Origen (ANTES de la transacción)', min_value=0.0, value=1250000.0, step=10000.0)
        sb_dest_a = st.number_input('Saldo Cuenta Destino (ANTES de la transacción)', min_value=0.0, value=0.0, step=10000.0,
                                    help="Saldo cero en destino es un indicador común de cuentas receptoras/mulas.")
        st.info("ℹ️ **Metodología Pre-Autorización:** Los saldos posteriores han sido eliminados de la evaluación para prevenir fuga de información (*data leakage*).")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button('EVALUAR TRANSACCIÓN EN TIEMPO REAL', use_container_width=True):
        if not modelos_cargados:
            st.error("Modelos no disponibles.")
        else:
            tipo_str = tipo or 'TRANSFER'
            with st.spinner('Evaluando Motor Híbrido (Reglas + IA + SHAP)...'):
                time.sleep(0.3)
                
                # 1. Evaluación del Motor de Reglas Deterministas
                rule_eval = rules_engine.evaluate(
                    tipo=tipo_str,
                    monto=monto,
                    hora_dia=int(step) % 24,
                    riesgo_dispositivo_ip=riesgo_dispositivo_ip,
                    velocidad_trans_1h=velocidad_trans_1h
                )
                
                features = preprocesar_transaccion(
                    tipo_str, monto, sb_orig_a, sb_dest_a, step
                )

                if rule_eval['decision'] == 'HARD_BLOCK':
                    pred = 1
                    score = 0.98
                    block_by_rule = True
                    rule_reason = rule_eval['reason']
                else:
                    block_by_rule = False
                    rule_reason = ""
                    if tipo_str not in ['TRANSFER', 'CASH_OUT']:
                        pred = 0
                        score = 0.0
                    else:
                        pred, score = predecir(modelo_sel, features)

                # Registrar caso en la Cola de Analista si es Riesgo Moderado o Alto
                if score >= 0.30:
                    tx_id = f"TX-{len(st.session_state.analyst_queue)+8943}"
                    st.session_state.analyst_queue.append({
                        'id': tx_id,
                        'timestamp': _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'cliente': f"CLI-{abs(hash(str(monto)))%8999+1000}",
                        'tipo': tipo_str,
                        'monto': monto,
                        'score': score,
                        'decision_inicial': 'HARD_BLOCK' if block_by_rule else ('SUSPENDIDA_ML' if score > 0.70 else 'PENDIENTE_OTP'),
                        'regla_activada': rule_reason if block_by_rule else f"Riesgo ML Elevado ({score*100:.1f}%)",
                        'estado': 'PENDIENTE_REVISION',
                        'detalles': {
                            'oldbalanceOrg': sb_orig_a,
                            'oldbalanceDest': sb_dest_a,
                            'dif_saldo_monto': sb_orig_a - monto,
                            'ratio_monto_saldo': monto / (sb_orig_a + 1.0),
                            'dest_saldo_cero': int(sb_dest_a == 0.0),
                            'velocidad': velocidad_trans_1h,
                            'riesgo_ip': riesgo_dispositivo_ip
                        }
                    })

                st.session_state.current_eval = {
                    'pred': pred,
                    'score': score,
                    'block_by_rule': block_by_rule,
                    'rule_reason': rule_reason,
                    'features': features,
                    'monto': monto,
                    'tipo': tipo_str,
                    'modelo_sel': modelo_sel
                }

    # Renderizado de Resultados si hay una evaluación en sesión
    current_eval = st.session_state.get('current_eval')
    if current_eval is not None:
        ev = current_eval
        score = ev['score']
        block_by_rule = ev['block_by_rule']
        rule_reason = ev['rule_reason']
        features = ev['features']
        modelo_sel = ev['modelo_sel']
        
        st.subheader("Resultado de la Evaluación de Riesgo Híbrida")
        res_col1, res_col2, res_col3 = st.columns(3)

        if block_by_rule or score > 0.70:
            with res_col1:
                st.markdown('<div class="glass-card-danger">', unsafe_allow_html=True)
                st.metric("DECISIÓN DE SEGURIDAD", "🔴 ALERTA DE SUSPENSIÓN", delta_color="inverse")
                st.markdown('</div>', unsafe_allow_html=True)
            with res_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("PUNTUACIÓN DE RIESGO", f"{score * 100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with res_col3:
                st.markdown('<div class="glass-card-danger">', unsafe_allow_html=True)
                st.metric("ACCIÓN AUTOMÁTICA", "COLA DE ANALISTA")
                st.markdown('</div>', unsafe_allow_html=True)

            if block_by_rule:
                st.error(f"**BLOQUEO POR MOTOR DE REGLAS DE NEGOCIO:** {rule_reason}. Transacción enviada inmediatamente a la Consola de Analista (Pestaña 4).")
            else:
                st.error(f"**NIVEL DE RIESGO ALTO (ML > 70%):** Transacción Suspendida Preventivamente por la IA ({modelo_sel}).")

            col_xai, col_sms = st.columns([1.3, 1.0])
            with col_xai:
                fig_shap = generar_grafico_shap(modelo_sel, features)
                st.plotly_chart(fig_shap, use_container_width=True)
            with col_sms:
                render_sms_alert(celular_titular, ev['monto'], ev['tipo'], score)

        elif 0.30 <= score <= 0.70:
            with res_col1:
                st.markdown('<div class="glass-card-warning">', unsafe_allow_html=True)
                st.metric("DECISIÓN DE SEGURIDAD", "🟡 RIESGO MODERADO")
                st.markdown('</div>', unsafe_allow_html=True)
            with res_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("PUNTUACIÓN DE RIESGO", f"{score * 100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with res_col3:
                st.markdown('<div class="glass-card-warning">', unsafe_allow_html=True)
                st.metric("FRICCIÓN ADAPTATIVA", "VERIFICACIÓN OTP")
                st.markdown('</div>', unsafe_allow_html=True)

            st.warning("**FRICCIÓN ADAPTATIVA REQUERIDA (Riesgo 30% - 70%):** Ingrese el código OTP de 6 dígitos enviado por SMS para completar la operación.")
            
            # Simulador de OTP / MFA Interactivo
            st.markdown('<div class="glass-card-warning">', unsafe_allow_html=True)
            st.markdown("#### Simulador Interactivo de Verificación OTP / MFA")
            otp_code = st.text_input("Ingrese el Código de Verificación OTP (Para demostración use: 123456):", type="password")
            if st.button("Validar Código OTP y Finalizar"):
                if otp_code == "123456":
                    st.success("Verificación OTP Exitosa. La transacción ha sido AUTORIZADA.")
                else:
                    st.error("Código OTP incorrecto. Por razones de seguridad la operación se mantiene retenida.")
            st.markdown('</div>', unsafe_allow_html=True)

            fig_shap = generar_grafico_shap(modelo_sel, features)
            st.plotly_chart(fig_shap, use_container_width=True)

        else:
            with res_col1:
                st.markdown('<div class="glass-card-success">', unsafe_allow_html=True)
                st.metric("DECISIÓN DE SEGURIDAD", "🟢 TRANSACCIÓN LEGÍTIMA")
                st.markdown('</div>', unsafe_allow_html=True)
            with res_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("PUNTUACIÓN DE RIESGO", f"{score * 100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with res_col3:
                st.markdown('<div class="glass-card-success">', unsafe_allow_html=True)
                st.metric("ESTADO TRANSACCIÓN", "APROBADA")
                st.markdown('</div>', unsafe_allow_html=True)

            st.success("Transacción confiable y aprobada de inmediato. Cumple con patrones financieros normales.")

# ══════════════════════════════════════════
# PESTAÑA 2: ANÁLISIS BATCH CSV
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Análisis Masivo por Lotes (Batch Scoring)")
    st.write("Carga un archivo CSV con transacciones masivas para clasificarlas y tabular alertas.")
    
    archivo = st.file_uploader('Selecciona un archivo CSV', type=['csv'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if archivo:
        df_batch = pd.DataFrame(pd.read_csv(archivo))
        required_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']
        missing_cols = [col for col in required_cols if col not in df_batch.columns]

        if missing_cols:
            st.error(f" Columnas faltantes para evaluación pre-autorización: {', '.join(missing_cols)}")
        else:
            st.success(f" Registros cargados: {len(df_batch):,}")
            st.dataframe(df_batch.head(5), use_container_width=True)
            
            modelo_batch = st.selectbox('Selecciona Modelo para Lotes', ['XGBoost', 'Random Forest', 'Regresión Logística']) or 'XGBoost'
            
            if st.button('EJECUTAR ANÁLISIS MASIVO DE TRANSACCIONES', type='primary'):
                with st.spinner('Procesando lote...'):
                    df_proc = df_batch.copy()
                    df_proc['ratio_monto_saldo']  = df_proc['amount'] / (df_proc['oldbalanceOrg'] + 1.0)
                    df_proc['dif_saldo_monto']    = df_proc['oldbalanceOrg'] - df_proc['amount']
                    df_proc['dest_saldo_cero']    = (df_proc['oldbalanceDest'] == 0.0).astype(int)
                    df_proc['es_extraccion_alta'] = ((df_proc['type'].isin(['TRANSFER','CASH_OUT'])) & (df_proc['amount'] > UMBRAL_95_ENTRENAMIENTO)).astype(int)
                    df_proc['hora_dia']           = df_proc['step'] % 24
                    df_proc['es_madrugada']       = ((df_proc['hora_dia'] >= 1) & (df_proc['hora_dia'] <= 4)).astype(int)
                    df_proc['type_TRANSFER']      = (df_proc['type'] == 'TRANSFER').astype(int)
                    
                    features_batch = [
                        'amount', 'oldbalanceOrg', 'oldbalanceDest',
                        'ratio_monto_saldo', 'dif_saldo_monto', 'dest_saldo_cero',
                        'es_extraccion_alta', 'hora_dia', 'es_madrugada', 'type_TRANSFER'
                    ]
                    
                    X_batch = df_proc[features_batch].fillna(0)
                    modelo = modelos[modelo_batch]
                    if modelo_batch == 'Regresión Logística':
                        X_b_sc = scaler.transform(X_batch)
                        probas = modelo.predict_proba(X_b_sc)[:, 1]
                    else:
                        probas = modelo.predict_proba(X_batch.values)[:, 1]

                    tipo_mask: Any = df_proc['type'].isin(['TRANSFER', 'CASH_OUT']).to_numpy()
                    probas = np.where(tipo_mask, probas, 0.0)
                    preds = (probas > 0.5).astype(int)
                    
                    df_batch['Score Riesgo (%)'] = np.round(probas * 100, 2).tolist()
                    df_batch['Predicción'] = np.where(preds == 1, '🔴 Fraude', '🟢 Legítimo').tolist()
                    fraudes_det = int(np.sum(preds))
                    legitimos_det = len(df_batch) - fraudes_det
                    
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total Lote", f"{len(df_batch):,}")
                with col_m2:
                    st.metric("Transacciones Legítimas", f"{legitimos_det:,}")
                with col_m3:
                    st.metric("Fraudes Detectados", f"{fraudes_det:,}")
                with col_m4:
                    st.metric("Tasa de Fraude", f"{fraudes_det/len(df_batch)*100:.3f}%")

                st.divider()

                # Separar transacciones por categoría
                df_legit = df_batch[df_batch['Predicción'] == '🟢 Legítimo']
                df_fraud = df_batch[df_batch['Predicción'] == '🔴 Fraude']

                st.markdown("### Gráficas Separadas de Score de Riesgo (Legítimas vs Fraudulentas)")

                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 🟢 Transacciones Legítimas")
                    if len(df_legit) > 0:
                        prom_legit = float(df_legit['Score Riesgo (%)'].mean())
                        st.caption(f"Total: **{len(df_legit):,}** | Promedio Riesgo: **{prom_legit:.2f}%**")
                        fig_legit = px.histogram(
                            df_legit,
                            x='Score Riesgo (%)',
                            nbins=40,
                            title='Distribución de Score: Legítimas',
                            color_discrete_sequence=['#10B981'],
                            template='plotly_dark'
                        )
                        fig_legit.update_layout(
                            height=350,
                            xaxis_title='Score de Riesgo (%)',
                            yaxis_title='Cantidad de Transacciones',
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_legit, use_container_width=True)
                    else:
                        st.info("No hay transacciones legítimas en este lote.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_g2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("#### 🔴 Transacciones Fraudulentas")
                    if len(df_fraud) > 0:
                        prom_fraud = float(df_fraud['Score Riesgo (%)'].mean())
                        st.caption(f"Total: **{len(df_fraud):,}** | Promedio Riesgo: **{prom_fraud:.2f}%**")
                        fig_fraud = px.histogram(
                            df_fraud,
                            x='Score Riesgo (%)',
                            nbins=40,
                            title='Distribución de Score: Fraudulentas',
                            color_discrete_sequence=['#EF4444'],
                            template='plotly_dark'
                        )
                        fig_fraud.update_layout(
                            height=350,
                            xaxis_title='Score de Riesgo (%)',
                            yaxis_title='Cantidad de Transacciones',
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_fraud, use_container_width=True)
                    else:
                        st.info("No se detectaron transacciones fraudulentas en este lote.")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("### Visión Comparativa Superpuesta")
                fig_comp = px.histogram(
                    df_batch,
                    x='Score Riesgo (%)',
                    color='Predicción',
                    barmode='overlay',
                    nbins=50,
                    title='Distribución Comparativa Legítimas vs. Fraudulentas',
                    color_discrete_map={'🟢 Legítimo': '#10B981', '🔴 Fraude': '#EF4444'},
                    template='plotly_dark'
                )
                fig_comp.update_layout(
                    height=380,
                    xaxis_title='Score de Riesgo (%)',
                    yaxis_title='Cantidad de Transacciones',
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_comp, use_container_width=True)

# ══════════════════════════════════════════
# PESTAÑA 3: DASHBOARD CIENTÍFICO & MÉTRICAS
# ══════════════════════════════════════════
with tab3:
    st.subheader("Dashboard del Desempeño Científico de los Modelos (Sin Fuga de Información)")
    st.markdown("Resultados empíricos obtenidos sobre el conjunto de prueba independiente (**65,824 transacciones, 1,232 fraudes**) con **10 características pre-autorización**.")

    res_csv_path = DATA_DIR / 'comparacion_modelos.csv'
    if res_csv_path.exists():
        df_metricas_loaded = pd.read_csv(res_csv_path)
        cols_mostrar = ['Modelo', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'PR-AUC', 'MCC', 'Falsos_Negativos', 'Falsos_Neg_Pct', 'Tiempo_Entrenamiento_s']
        st.dataframe(df_metricas_loaded[[c for c in cols_mostrar if c in df_metricas_loaded.columns]], use_container_width=True)
    else:
        st.info("Cargando métricas científicas...")

    def mostrar_imagen(path_or_name, caption):
        target_path = Path(path_or_name)
        if not target_path.exists():
            filename = target_path.name
            candidatos = [
                EVIDENCIA_DIR / filename,
                NOTEBOOKS_DIR / filename,
                ROOT_DIR / 'evidencia' / filename,
                ROOT_DIR / 'evidencia' / 'graficos_shap' / filename
            ]
            for cand in candidatos:
                if cand.exists():
                    target_path = cand
                    break

        if target_path.exists():
            try:
                from PIL import Image
                img = Image.open(target_path)
                st.image(img, caption=caption, use_column_width=True)
            except Exception as e:
                st.error(f"Error al abrir la imagen {target_path.name}: {str(e)}")
        else:
            st.warning(f"No se encontró la imagen en la ruta: {target_path}")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        mostrar_imagen(EVIDENCIA_DIR / 'curvas_roc.png', 'Curvas ROC Comparativas (Línea Base + 4 Modelos IA)')
        mostrar_imagen(EVIDENCIA_DIR / 'curvas_calibracion.png', 'Curvas de Calibración / Reliability Diagrams')
    with col_img2:
        mostrar_imagen(EVIDENCIA_DIR / 'curvas_precision_recall.png', 'Curvas Precision-Recall Comparativas')
        mostrar_imagen(EVIDENCIA_DIR / 'matrices_confusion_grid.png', 'Matrices de Confusión en Valores Absolutos')

    with st.expander("📊 Ver más evidencia gráfica (Importancia de Variables, Correlación, Distribuciones)", expanded=False):
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            mostrar_imagen(EVIDENCIA_DIR / 'fig_importancia_features.png', 'Importancia de Variables (Random Forest / XGBoost)')
            mostrar_imagen(EVIDENCIA_DIR / 'fig_distribucion_clases.png', 'Distribución de Clases (Legítimas vs Fraudes)')
            mostrar_imagen(EVIDENCIA_DIR / 'fig_fraude_por_tipo.png', 'Proporción de Fraude por Tipo de Transacción')
        with col_e2:
            mostrar_imagen(EVIDENCIA_DIR / 'fig_correlacion.png', 'Matriz de Correlación de Variables')
            mostrar_imagen(EVIDENCIA_DIR / 'fig_distribucion_montos.png', 'Distribución de Montos por Categoría')

# ══════════════════════════════════════════
# PESTAÑA 4: CONSOLA DE GESTIÓN DE ANALISTA (BACK-OFFICE)
# ══════════════════════════════════════════
with tab4:
    st.subheader("Consola de Gestión de Analista de Fraudes (Back-Office)")
    st.write("Módulo operativo para que el analista de prevención de fraudes inspeccione, libere o confirme fraudes en cola.")

    queue = st.session_state.get('analyst_queue', [])
    pendientes = [c for c in queue if c.get('estado') == 'PENDIENTE_REVISION']
    history = st.session_state.get('resolution_history', [])
    resueltos = len(history)

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Casos Pendientes en Cola", f"{len(pendientes)}")
    with col_m2:
        st.metric("Casos Resueltos en Sesión", f"{resueltos}")
    with col_m3:
        st.metric("Total en Expediente", f"{len(queue)}")

    st.divider()

    if not pendientes:
        st.success(" No hay casos pendientes de revisión en la cola del analista.")
    else:
        opciones_casos = [f"{c['id']} | {c['cliente']} | RD${c['monto']:,.2f} | Score: {c['score']*100:.1f}%" for c in pendientes]
        caso_sel_str = st.selectbox("Seleccione un Caso de la Cola para Evaluación:", opciones_casos)
        caso_id = caso_sel_str.split(" | ")[0] if caso_sel_str else pendientes[0]['id']
        caso = next((c for c in pendientes if c['id'] == caso_id), pendientes[0])

        col_c1, col_c2 = st.columns([1.2, 1.0])
        with col_c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"### Expediente del Caso: `{caso['id']}`")
            st.write(f"**Titular de Cuenta:** {caso['cliente']}")
            st.write(f"**Marca Temporal:** {caso['timestamp']}")
            st.write(f"**Operación:** `{caso['tipo']}` | **Monto:** `RD${caso['monto']:,.2f}`")
            st.write(f"**Score de Riesgo:** `{caso['score']*100:.1f}%`")
            st.write(f"**Motivo de Suspensión:** `{caso['regla_activada']}`")

            st.markdown("#### Métricas de Contexto y Balances:")
            st.json(caso.get('detalles', {}))
            st.markdown('</div>', unsafe_allow_html=True)

        with col_c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Decisiones Operativas del Analista")
            st.caption("Seleccione la resolución tras validar la legitimidad con el usuario:")

            if st.button("APROBAR TRANSACCIÓN (Liberar Fondos)", use_container_width=True):
                caso['estado'] = 'APROBADA_POR_ANALISTA'
                st.session_state.resolution_history.append({
                    'id': caso['id'],
                    'accion': 'APROBADA (LIBERADA)',
                    'timestamp': _dt.datetime.now().strftime('%H:%M:%S'),
                    'analista': 'Analista_Fraudes_01'
                })
                st.success(f"Caso {caso['id']} liberado y aprobado.")
                st.rerun()

            if st.button("CONFIRMAR FRAUDE (Bloquear Cuenta)", use_container_width=True):
                caso['estado'] = 'FRAUDE_CONFIRMADO'
                st.session_state.resolution_history.append({
                    'id': caso['id'],
                    'accion': 'FRAUDE CONFIRMADO (BLOQUEO)',
                    'timestamp': _dt.datetime.now().strftime('%H:%M:%S'),
                    'analista': 'Analista_Fraudes_01'
                })
                st.error(f"Fraude confirmado para {caso['id']}. Cuenta bloqueada.")
                st.rerun()

            if st.button("SOLICITAR DOCUMENTACIÓN ADICIONAL", use_container_width=True):
                caso['estado'] = 'EN_ESPERA_DOCS'
                st.info(f"Solicitud de documentos enviada para el caso {caso['id']}.")
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    history = st.session_state.get('resolution_history', [])
    if history:
        st.divider()
        st.subheader("Historial de Resoluciones de la Sesión Actual")
        st.dataframe(pd.DataFrame(history), use_container_width=True)
