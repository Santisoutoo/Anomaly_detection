# 🔧 Laboratorio Unidad III — Detección de Anomalías en NASA C‑MAPSS

## 📌 Descripción general
Proyecto para comparar 7 técnicas de detección de anomalías aplicadas al dataset NASA C‑MAPSS (simulación de degradación de motores turbofan). El objetivo es evaluar qué método detecta más tempranamente la degradación y proponer una arquitectura de mantenimiento predictivo basada en gemelos digitales.

## 🎯 Objetivos
- Implementar y comparar múltiples algoritmos de detección de anomalías.
- Evaluar la capacidad de detección temprana de cada método.
- Identificar variables críticas que indiquen degradación.
- Diseñar una propuesta de gemelo digital para mantenimiento predictivo.

## 📊 Dataset
**NASA C‑MAPSS (Turbofan Engine Degradation Simulation)**  
- Registros de sensores desde operación normal hasta fallo.
- Varias unidades y condiciones operativas.
- 21 sensores que monitorean temperatura, presión, velocidad, etc.
- Formato: tablas por unidad/episodio (csv/parquet), con marcas temporales y ciclos de vida.

## 🛠️ Métodos implementados

### 1) Métodos clásicos
- **Z‑scores**: detección por desviación estándar (umbral sobre el score).  
- **PCA**: reducción dimensional y detección de outliers mediante reconstrucción o score de Mahalanobis.

### 2) Machine Learning (no supervisado)
- **Isolation Forest**: aislamiento por particionamiento aleatorio.  
- **One‑Class SVM**: frontera de decisión para la clase “normal”.

### 3) Deep Learning
- **Autoencoder**: reconstrucción y umbral sobre error de reconstrucción.  
- **LSTM‑Autoencoder**: captura dependencias temporales en las series.  
- **TCN‑VAE**: convoluciones temporales causales + variational autoencoder para modelado probabilístico.


## ✅ Notas rápidas
- Versionar datasets y checkpoints grandes fuera del repositorio (p. ej., DVC / almacenamiento externo).
- Definir métricas de detección temprana (e.g., tiempo de advertencia antes del fallo, tasa de falsos positivos).
- Automatizar evaluación con pipelines reproducibles (scripts / CI).

