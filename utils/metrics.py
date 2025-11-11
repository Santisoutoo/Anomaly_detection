import numpy as np
import pandas as pd

def calculate_anomaly_metrics(
    df, rul_col='RUL',
    anomaly_col='is_anomaly',
    unit_col='unit_id',
    time_col='time_cycles'
):
    """
    Métricas estandarizadas para todos los métodos.
    
    Returns:
        dict con métricas clave
    """
    results = {
        'method': '',
        'dataset': '',
        'total_units': df[unit_col].nunique(),
        'units_detected': 0,
        'detection_rate': 0.0,
        'avg_first_detection_rul': 0.0,
        'avg_detection_cycle_pct': 0.0,
        'early_detections': 0,  # RUL > 50
        'false_positive_rate': 0.0,  # anomalías con RUL > 150
        'anomaly_percentage': 0.0
    }
    
    first_detections = []
    cycle_percentages = []
    false_positives = 0
    
    for unit in df[unit_col].unique():
        unit_data = df[df[unit_col] == unit].sort_values(time_col)
        anomalies = unit_data[unit_data[anomaly_col] == 1]
        
        if len(anomalies) > 0:
            results['units_detected'] += 1
            
            # Primera detección
            first_anom = anomalies.iloc[0]
            first_rul = first_anom[rul_col]
            first_detections.append(first_rul)
            
            # % del ciclo
            total_cycles = unit_data[time_col].max()
            first_cycle = first_anom[time_col]
            cycle_percentages.append((first_cycle / total_cycles) * 100)
            
            # Contadores
            if first_rul > 50:
                results['early_detections'] += 1
            if first_rul > 150:
                false_positives += 1
    
    # Promedios
    if first_detections:
        results['avg_first_detection_rul'] = np.mean(first_detections)
        results['avg_detection_cycle_pct'] = np.mean(cycle_percentages)
        results['detection_rate'] = (results['units_detected'] / results['total_units']) * 100
        results['false_positive_rate'] = (false_positives / results['units_detected']) * 100
    
    results['anomaly_percentage'] = (df[anomaly_col].sum() / len(df)) * 100
    
    return results

def format_results(metrics):
    """Imprime resultados formateados"""
    print("="*60)
    print(f"📊 {metrics['method']} - {metrics['dataset']}")
    print("="*60)
    print(f"Cobertura: {metrics['detection_rate']:.1f}% ({metrics['units_detected']}/{metrics['total_units']} unidades)")
    print(f"RUL promedio 1ª detección: {metrics['avg_first_detection_rul']:.1f} ciclos")
    print(f"Detección al {metrics['avg_detection_cycle_pct']:.1f}% del ciclo")
    print(f"Detecciones tempranas (RUL>50): {metrics['early_detections']}")
    print(f"Falsos positivos (RUL>150): {metrics['false_positive_rate']:.1f}%")
    print(f"% anomalías totales: {metrics['anomaly_percentage']:.1f}%")
    print("="*60)