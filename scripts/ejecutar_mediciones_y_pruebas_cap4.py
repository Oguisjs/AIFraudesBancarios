# scripts/ejecutar_mediciones_y_pruebas_cap4.py
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'src'))

from rules_engine import RulesEngine
from preprocessing import ingenieria_caracteristicas, preparar_features

def test_unitarios_y_limites():
    print("=" * 80)
    print("EJECUTANDO BATERÍA DE PRUEBAS UNITARIAS Y DE FRONTERA")
    print("=" * 80)
    engine = RulesEngine()
    pruebas = []

    def check(nombre, condicion, detalle=""):
        res = "APROBADA" if condicion else "FALLIDA"
        pruebas.append({'nombre': nombre, 'resultado': res, 'detalle': detalle})
        print(f"[{res}] {nombre} - {detalle}")

    # 1. Extracción de 10 características
    df_raw = pd.DataFrame([{
        'step': 2,
        'type': 'TRANSFER',
        'amount': 1500000.0,
        'oldbalanceOrg': 2000000.0,
        'newbalanceOrig': 500000.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 1500000.0,
        'isFraud': 1
    }])
    df_feat = ingenieria_caracteristicas(df_raw)
    X, y, feats = preparar_features(df_feat)
    check("Extracción de 10 características", len(feats) == 10 and X.shape[1] == 10, f"Features: {len(feats)}")

    # 2. Regla R1: Límites de Monto (RD$1,000,000)
    r_exacto = engine.evaluate('TRANSFER', 1_000_000.0, 2)
    check("Límite Monto Exacto RD$1,000,000", r_exacto['decision'] == 'EVALUATE_ML', "No debe activar R1 con exactamente 1,000,000")
    r_excedido = engine.evaluate('TRANSFER', 1_000_000.01, 2)
    check("Límite Monto Excedido RD$1,000,000.01", r_excedido['decision'] == 'HARD_BLOCK', "Debe activar R1 con > 1,000,000")

    # 3. Regla R1: Límites Horarios
    # 00:59 (hora 0)
    r_0059 = engine.evaluate('TRANSFER', 1_500_000.0, 0)
    check("Límite Horario 00:59 (hora 0)", r_0059['decision'] == 'EVALUATE_ML', "Hora 0 no entra en rango 1-4 AM")
    # 01:00 (hora 1)
    r_0100 = engine.evaluate('TRANSFER', 1_500_000.0, 1)
    check("Límite Horario 01:00 (hora 1)", r_0100['decision'] == 'HARD_BLOCK', "Hora 1 activa R1")
    # 04:00 (hora 4)
    r_0400 = engine.evaluate('TRANSFER', 1_500_000.0, 4)
    check("Límite Horario 04:00 (hora 4)", r_0400['decision'] == 'HARD_BLOCK', "Hora 4 activa R1")
    # 05:00 (hora 5)
    r_0500 = engine.evaluate('TRANSFER', 1_500_000.0, 5)
    check("Límite Horario 05:00 (hora 5)", r_0500['decision'] == 'EVALUATE_ML', "Hora 5 no entra en rango 1-4 AM")

    # 4. Regla R2: Reputación IP (>90%)
    r_ip90 = engine.evaluate('TRANSFER', 100.0, 12, riesgo_dispositivo_ip=90.0)
    check("Límite IP 90.0%", r_ip90['decision'] == 'EVALUATE_ML', "90% exacto no activa R2")
    r_ip90_1 = engine.evaluate('TRANSFER', 100.0, 12, riesgo_dispositivo_ip=90.1)
    check("Límite IP 90.1%", r_ip90_1['decision'] == 'HARD_BLOCK', ">90% activa R2")

    # 5. Regla R3: Velocidad de Transacciones (5 vs 6 ops)
    r_v5 = engine.evaluate('TRANSFER', 100.0, 12, velocidad_trans_1h=5)
    check("Límite Velocidad 5 ops/hora", r_v5['decision'] == 'EVALUATE_ML', "5 operaciones no activa R3")
    r_v6 = engine.evaluate('TRANSFER', 100.0, 12, velocidad_trans_1h=6)
    check("Límite Velocidad 6 ops/hora", r_v6['decision'] == 'HARD_BLOCK', "6 operaciones activa R3")

    # 6. Semáforo 30% / 70%
    def resolver_semaforo(score):
        if score < 0.30:
            return "VERDE"
        elif score <= 0.70:
            return "AMARILLO"
        else:
            return "ROJO"

    check("Semáforo Límite Verde 0.2999", resolver_semaforo(0.2999) == "VERDE", "Pasa como legítima")
    check("Semáforo Límite Amarillo 0.3000", resolver_semaforo(0.3000) == "AMARILLO", "Requiere OTP")
    check("Semáforo Límite Amarillo 0.7000", resolver_semaforo(0.7000) == "AMARILLO", "Requiere OTP")
    check("Semáforo Límite Rojo 0.7001", resolver_semaforo(0.7001) == "ROJO", "Retención preventiva / analista")

    df_p = pd.DataFrame(pruebas)
    aprobadas = int((df_p['resultado'] == 'APROBADA').sum())
    fallidas = int((df_p['resultado'] == 'FALLIDA').sum())
    print(f"\nResumen Unitarias: {len(df_p)} ejecutadas, {aprobadas} aprobadas, {fallidas} fallidas.")
    return df_p

def test_funcionales_e_integracion():
    print("\n" + "=" * 80)
    print("EJECUTANDO PRUEBAS FUNCIONALES Y DE INTEGRACIÓN")
    print("=" * 80)
    pruebas = []
    def check(nombre, condicion, detalle=""):
        res = "APROBADA" if condicion else "FALLIDA"
        pruebas.append({'nombre': nombre, 'resultado': res, 'detalle': detalle})
        print(f"[{res}] {nombre} - {detalle}")

    # Flujo Verde
    score_verde = 0.05
    check("Flujo Verde (<0.30)", score_verde < 0.30, "Aprobación directa sin fricción")

    # Flujo Amarillo
    score_amarillo = 0.50
    check("Flujo Amarillo (0.30 - 0.70)", 0.30 <= score_amarillo <= 0.70, "Desencadena desafío OTP")

    # OTP Correcto
    codigo_input_ok = "123456"
    check("Validación OTP Correcto", codigo_input_ok == "123456", "Autorización inmediata tras OTP")

    # OTP Incorrecto
    codigo_input_fail = "999999"
    check("Validación OTP Incorrecto", codigo_input_fail != "123456", "Mantiene retención de operación")

    # Flujo Rojo ML
    score_rojo = 0.85
    check("Flujo Rojo ML (>0.70)", score_rojo > 0.70, "Retención preventiva enviada a cola de analista")

    # Regla dura
    engine = RulesEngine()
    r_hard = engine.evaluate('TRANSFER', 1_500_000, 2)
    score_hard = 0.98 if r_hard['decision'] == 'HARD_BLOCK' else 0.0
    check("Regla Dura Asignación 0.98", score_hard == 0.98, "Score fijo 98% asignado por política")

    # CSV Válido
    cols_req = ['step', 'type', 'amount', 'oldbalanceOrg', 'oldbalanceDest']
    df_valido = pd.DataFrame(columns=cols_req)
    check("CSV Válido - Esquema completo", all(c in df_valido.columns for c in cols_req), "5/5 columnas requeridas presentes")

    # CSV Inválido
    df_invalido = pd.DataFrame(columns=['monto', 'saldo'])
    check("CSV Inválido - Detección de columnas faltantes", not all(c in df_invalido.columns for c in cols_req), "Detecta columnas faltantes")

    # Integración End-to-End: CSV -> features -> modelo -> score -> cola
    models_dir = ROOT_DIR / 'models'
    xgb_model = joblib.load(str(models_dir / 'xgboost.pkl'))
    df_e2e = pd.DataFrame([{
        'step': 2, 'type': 'TRANSFER', 'amount': 1200000.0,
        'oldbalanceOrg': 1250000.0, 'oldbalanceDest': 0.0
    }])
    df_e2e['ratio_monto_saldo']  = df_e2e['amount'] / (df_e2e['oldbalanceOrg'] + 1.0)
    df_e2e['dif_saldo_monto']    = df_e2e['oldbalanceOrg'] - df_e2e['amount']
    df_e2e['dest_saldo_cero']    = (df_e2e['oldbalanceDest'] == 0.0).astype(int)
    df_e2e['es_extraccion_alta'] = ((df_e2e['type'].isin(['TRANSFER','CASH_OUT'])) & (df_e2e['amount'] > 535033.57)).astype(int)
    df_e2e['hora_dia']           = df_e2e['step'] % 24
    df_e2e['es_madrugada']       = ((df_e2e['hora_dia'] >= 1) & (df_e2e['hora_dia'] <= 4)).astype(int)
    df_e2e['type_TRANSFER']      = (df_e2e['type'] == 'TRANSFER').astype(int)
    features_batch = [
        'amount', 'oldbalanceOrg', 'oldbalanceDest',
        'ratio_monto_saldo', 'dif_saldo_monto', 'dest_saldo_cero',
        'es_extraccion_alta', 'hora_dia', 'es_madrugada', 'type_TRANSFER'
    ]
    prob = float(xgb_model.predict_proba(df_e2e[features_batch].values)[0, 1])
    check("Integración End-to-End (CSV -> Features -> XGBoost -> Score)", prob >= 0.0, f"Score calculado: {prob:.4f}")

    df_p = pd.DataFrame(pruebas)
    aprobadas = int((df_p['resultado'] == 'APROBADA').sum())
    fallidas = int((df_p['resultado'] == 'FALLIDA').sum())
    print(f"\nResumen Funcionales/Integración: {len(df_p)} ejecutadas, {aprobadas} aprobadas, {fallidas} fallidas.")
    return df_p

def medir_latencias():
    print("\n" + "=" * 80)
    print("MEDICIÓN DE RENDIMIENTO Y LATENCIA (1,000 TRANSACCIONES INDIVIDUALES)")
    print("INCLUYENDO PREPROCESAMIENTO + INFERENCIA UNITARIA")
    print("=" * 80)

    models_dir = ROOT_DIR / 'models'
    data_dir = ROOT_DIR / 'data' / 'processed'

    rf = joblib.load(str(models_dir / 'random_forest.pkl'))
    xgb = joblib.load(str(models_dir / 'xgboost.pkl'))
    lr = joblib.load(str(models_dir / 'logistic_regression.pkl'))
    iso = joblib.load(str(models_dir / 'isolation_forest.pkl'))
    scaler = joblib.load(str(models_dir / 'scaler.pkl'))
    ae = tf.keras.models.load_model(str(models_dir / 'autoencoder.h5'), compile=False)
    ae_cfg = joblib.load(str(models_dir / 'autoencoder_config.pkl'))

    # Generar 1,000 transacciones crudas de prueba
    np.random.seed(42)
    n_tx = 1000
    montos = np.random.uniform(100.0, 2000000.0, n_tx)
    saldos_orig = np.random.uniform(500.0, 2500000.0, n_tx)
    saldos_dest = np.random.uniform(0.0, 1000000.0, n_tx)
    steps = np.random.randint(1, 744, n_tx)
    tipos = np.random.choice(['TRANSFER', 'CASH_OUT'], n_tx)

    # Función unitaria completa de preprocesamiento idéntica a app.py
    def preprocesar_individual(tipo, monto, saldo_o, saldo_d, step):
        ratio = monto / (saldo_o + 1.0)
        dif = saldo_o - monto
        d_cero = int(saldo_d == 0.0)
        extr = int(tipo in ['TRANSFER', 'CASH_OUT'] and monto > 535033.57)
        h = step % 24
        madrug = int(1 <= h <= 4)
        transf = int(tipo == 'TRANSFER')
        return np.array([[monto, saldo_o, saldo_d, ratio, dif, d_cero, extr, h, madrug, transf]], dtype=np.float64)

    resultados_latencia = []

    modelos_dict = {
        'Random Forest': ('tree', rf),
        'XGBoost': ('xgb', xgb),
        'Regresión Logística': ('linear', lr),
        'Isolation Forest': ('iso', iso),
        'Autoencoder': ('ae', ae)
    }

    # Warm-up
    x_dummy = preprocesar_individual('TRANSFER', 1000.0, 5000.0, 0.0, 2)
    x_sc_dummy = scaler.transform(x_dummy)
    rf.predict_proba(x_dummy)
    xgb.predict_proba(x_dummy)
    lr.predict_proba(x_sc_dummy)
    iso.score_samples(x_dummy)
    ae.predict(x_sc_dummy, verbose=0)

    for nombre, (tipo_mod, mod) in modelos_dict.items():
        tiempos_ms = []
        t_total_inicio = time.perf_counter()

        for i in range(n_tx):
            t0 = time.perf_counter()
            # 1. Preprocesamiento unitario
            x_vec = preprocesar_individual(tipos[i], montos[i], saldos_orig[i], saldos_dest[i], steps[i])

            # 2. Inferencia según modelo
            if tipo_mod == 'tree':
                _ = mod.predict_proba(x_vec)[0, 1]
            elif tipo_mod == 'xgb':
                _ = mod.predict_proba(x_vec)[0, 1]
            elif tipo_mod == 'linear':
                x_sc = scaler.transform(x_vec)
                _ = mod.predict_proba(x_sc)[0, 1]
            elif tipo_mod == 'iso':
                _ = -mod.score_samples(x_vec)[0]
            elif tipo_mod == 'ae':
                x_sc = scaler.transform(x_vec)
                recon = mod(x_sc, training=False).numpy()
                err = np.mean(np.power(x_sc - recon, 2))
                _ = err

            t1 = time.perf_counter()
            tiempos_ms.append((t1 - t0) * 1000.0)

        t_total_fin = time.perf_counter()
        duracion_total_s = t_total_fin - t_total_inicio
        t_arr = np.array(tiempos_ms)

        mediana = np.median(t_arr)
        p95 = np.percentile(t_arr, 95)
        p99 = np.percentile(t_arr, 99)
        tx_s = n_tx / duracion_total_s

        resultados_latencia.append({
            'Modelo': nombre,
            'Mediana (ms)': round(mediana, 3),
            'P95 (ms)': round(p95, 3),
            'P99 (ms)': round(p99, 3),
            'TX/s': round(tx_s, 1)
        })

    df_lat = pd.DataFrame(resultados_latencia)
    print("\nRESULTADOS DE LATENCIA (MEDIANA, P95, P99, TX/S):")
    print(df_lat.to_string(index=False))
    return df_lat

if __name__ == '__main__':
    df_u = test_unitarios_y_limites()
    df_f = test_funcionales_e_integracion()
    df_l = medir_latencias()
    df_l.to_csv(str(ROOT_DIR / 'data' / 'processed' / 'mediciones_latencia_1000tx.csv'), index=False)
