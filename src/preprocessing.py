# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

UMBRAL_95_POBLACIONAL = 535033.57

def cargar_datos(ruta, muestra=1_000_000, seed=42):
    """Carga una muestra estratificada del dataset manteniendo tasa de fraude"""
    df = pd.read_csv(ruta)
    fraudes = df[df['isFraud'] == 1]
    legitimas = df[df['isFraud'] == 0].sample(
        n=muestra - len(fraudes), random_state=seed
    )
    df_muestra = pd.concat([fraudes, legitimas]).sample(frac=1, random_state=seed)
    print(f'Dataset cargado: {len(df_muestra):,} filas')
    print(f'Fraudes: {df_muestra.isFraud.sum():,} ({df_muestra.isFraud.mean()*100:.3f}%)')
    return df_muestra

def ingenieria_caracteristicas(df):
    """
    Crea variables derivadas estrictamente PRE-AUTORIZACIÓN.
    Elimina cualquier fuga de información (newbalanceOrig, newbalanceDest y derivadas).
    """
    df = df.copy()

    # 1. Ratio del monto respecto al saldo disponible del emisor antes de la operación
    df['ratio_monto_saldo'] = df['amount'] / (df['oldbalanceOrg'] + 1.0)

    # 2. Diferencia entre saldo origen previo y monto solicitado (capacidad de cobertura)
    df['dif_saldo_monto'] = df['oldbalanceOrg'] - df['amount']

    # 3. ¿La cuenta destino tiene saldo cero antes de la transferencia? (Patrón típico de cuenta mula / nueva)
    df['dest_saldo_cero'] = (df['oldbalanceDest'] == 0.0).astype(int)

    # 4. Operación de extracción de alto monto (> percentil 95 poblacional)
    df['es_extraccion_alta'] = (
        (df['type'].isin(['TRANSFER', 'CASH_OUT'])) &
        (df['amount'] > UMBRAL_95_POBLACIONAL)
    ).astype(int)

    # 5. Hora del día (step % 24)
    df['hora_dia'] = df['step'] % 24

    # 6. Indicador de horario nocturno de alto riesgo (1 AM a 4 AM)
    df['es_madrugada'] = ((df['hora_dia'] >= 1) & (df['hora_dia'] <= 4)).astype(int)

    # 7. Filtrar solo TRANSFER y CASH_OUT (donde se concentra el fraude operativo)
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

    print(f'Filas tras filtrar por tipo (TRANSFER / CASH_OUT): {len(df):,}')
    return df

def preparar_features(df):
    """
    Codifica variables categóricas y estructura el vector limpio de características pre-autorización.
    """
    df = df.copy()
    # Codificación dummy para tipo de operación (1 si TRANSFER, 0 si CASH_OUT)
    if 'type_TRANSFER' not in df.columns:
        df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)

    # Lista definitiva de 10 características sin fuga de información
    features = [
        'amount',
        'oldbalanceOrg',
        'oldbalanceDest',
        'ratio_monto_saldo',
        'dif_saldo_monto',
        'dest_saldo_cero',
        'es_extraccion_alta',
        'hora_dia',
        'es_madrugada',
        'type_TRANSFER'
    ]

    X = df[features].copy()
    y = df['isFraud'].astype(int)
    return X, y, features

def dividir_y_balancear(X, y, seed=42):
    """
    Divide en train/val/test estratificado y aplica SMOTE estrictamente en train.
    """
    # División 70% train | 15% val | 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.17647058823529413,  # 0.15 / 0.85
        stratify=y_temp, random_state=seed
    )

    print(f'Train original: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}')
    print(f'Fraudes en train (antes SMOTE): {y_train.sum():,} ({y_train.mean()*100:.3f}%)')

    # SMOTE SOLO sobre training para evitar fuga hacia validación y prueba
    smote = SMOTE(sampling_strategy=0.1, random_state=seed, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f'Train tras SMOTE: {len(X_train_bal):,}')
    print(f'Fraudes tras SMOTE: {y_train_bal.sum():,} ({y_train_bal.mean()*100:.3f}%)')
    return X_train_bal, X_val, X_test, y_train_bal, y_val, y_test

def escalar(X_train, X_val, X_test):
    """
    Normaliza las variables ajustando el StandardScaler únicamente con datos de entrenamiento.
    Conserva los nombres de columnas para compatibilidad perfecta con Scikit-learn y SHAP.
    """
    scaler = StandardScaler()
    feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None

    # fit SOLO en train
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # Convertir a DataFrame manteniendo nombres de columnas
    if feature_names is not None:
        X_train_sc = pd.DataFrame(X_train_sc, columns=feature_names)
        X_val_sc   = pd.DataFrame(X_val_sc, columns=feature_names)
        X_test_sc  = pd.DataFrame(X_test_sc, columns=feature_names)

    # Guardar scaler para uso en inferencia
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f'Scaler guardado exitosamente en: {scaler_path}')
    return X_train_sc, X_val_sc, X_test_sc, scaler
