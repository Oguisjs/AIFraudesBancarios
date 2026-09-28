# scripts/01_regenerar_datos.py
import sys
import os
import time
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / 'src'))

from preprocessing import (
    cargar_datos,
    ingenieria_caracteristicas,
    preparar_features,
    dividir_y_balancear,
    escalar
)

def main():
    print("=" * 60)
    print("FASE 2: REGENERACIÓN DE DATOS PREPROCESADOS (SIN DATA LEAKAGE)")
    print("=" * 60)

    t0 = time.time()
    ruta_raw = ROOT_DIR / 'data' / 'raw' / 'PS_20174392719_1491204439457_log.csv'
    if not ruta_raw.exists():
        raise FileNotFoundError(f"No se encontró el archivo raw en {ruta_raw}")

    # 1. Cargar muestra estratificada de 1,000,000 de registros
    print("\n1. Cargando datos...")
    df = cargar_datos(str(ruta_raw), muestra=1_000_000, seed=42)

    # 2. Ingeniería de características limpia (pre-autorización)
    print("\n2. Aplicando ingeniería de características pre-autorización...")
    df_feat = ingenieria_caracteristicas(df)

    # 3. Preparar matriz de características (10 features)
    print("\n3. Preparando características...")
    X, y, features = preparar_features(df_feat)
    print(f"Características definitivas ({len(features)}): {features}")

    # 4. Dividir y balancear con SMOTE únicamente en train
    print("\n4. Dividiendo en Train (70%), Val (15%), Test (15%) y aplicando SMOTE...")
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_y_balancear(X, y, seed=42)

    # 5. Escalar datos ajustando StandardScaler sólo en train
    print("\n5. Ajustando StandardScaler y normalizando...")
    X_train_sc, X_val_sc, X_test_sc, scaler = escalar(X_train, X_val, X_test)

    # 6. Guardar en data/processed
    out_dir = ROOT_DIR / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n6. Guardando arrays en {out_dir}...")
    np.save(str(out_dir / 'X_train.npy'), X_train.values.astype(np.float64))
    np.save(str(out_dir / 'X_train_sc.npy'), X_train_sc.values.astype(np.float64))
    np.save(str(out_dir / 'y_train.npy'), y_train.values.astype(np.int64))

    np.save(str(out_dir / 'X_val.npy'), X_val.values.astype(np.float64))
    np.save(str(out_dir / 'X_val_sc.npy'), X_val_sc.values.astype(np.float64))
    np.save(str(out_dir / 'y_val.npy'), y_val.values.astype(np.int64))

    np.save(str(out_dir / 'X_test.npy'), X_test.values.astype(np.float64))
    np.save(str(out_dir / 'X_test_sc.npy'), X_test_sc.values.astype(np.float64))
    np.save(str(out_dir / 'y_test.npy'), y_test.values.astype(np.int64))

    elapsed = time.time() - t0
    print(f"\n[OK] Datos regenerados y guardados con éxito en {elapsed:.2f} segundos.")
    print("=" * 60)

if __name__ == '__main__':
    main()
