# src/rules_engine.py

class RulesEngine:
    """
    Motor Híbrido de Reglas Deterministas de Negocio.
    Evalúa transacciones previa inferencia de los modelos de Machine Learning.
    """
    def __init__(self):
        pass

    def evaluate(self, tipo: str, monto: float, hora_dia: int,
                 riesgo_dispositivo_ip: float = 0.0,
                 velocidad_trans_1h: int = 1) -> dict:
        """
        Evalúa si la transacción activa alguna regla dura de bloqueo.

        Args:
            tipo (str): Tipo de transacción ('TRANSFER', 'CASH_OUT', etc.)
            monto (float): Monto de la operación en RD$
            hora_dia (int): Hora de la transacción (0 - 23)
            riesgo_dispositivo_ip (float): Score de riesgo del dispositivo/IP (0 - 100)
            velocidad_trans_1h (int): Transacciones registradas en la última hora

        Returns:
            dict con keys: 'decision' ('HARD_BLOCK' | 'EVALUATE_ML'),
                           'rules_triggered' (list),
                           'reason' (str)
        """
        rules_triggered = []

        # Regla 1: Intento de transferencia / retiro > RD$1,000,000 en horario nocturno (1 AM - 4 AM)
        if tipo in ['TRANSFER', 'CASH_OUT'] and monto > 1_000_000 and (1 <= hora_dia <= 4):
            rules_triggered.append("R1: Transferencia nocturna de alto monto (>RD$1,000,000 entre 1 AM y 4 AM)")

        # Regla 2: Dispositivo con reputación de riesgo extremadamente alta (> 90%)
        if riesgo_dispositivo_ip > 90.0:
            rules_triggered.append(f"R2: Dispositivo/IP con reputación de alto riesgo ({riesgo_dispositivo_ip:.1f}%)")

        # Regla 3: Frecuencia atípica (> 5 transferencias en la última hora)
        if velocidad_trans_1h > 5:
            rules_triggered.append(f"R3: Frecuencia atípica de transacciones ({velocidad_trans_1h} ops/hora)")

        if rules_triggered:
            return {
                'decision': 'HARD_BLOCK',
                'rules_triggered': rules_triggered,
                'reason': " | ".join(rules_triggered)
            }
        else:
            return {
                'decision': 'EVALUATE_ML',
                'rules_triggered': [],
                'reason': 'Sin activación de reglas deterministas duras.'
            }
