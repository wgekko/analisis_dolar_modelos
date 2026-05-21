
## Plataforma de Análisis Macroeconómico y Forecasting Financiero Avanzado

Esta plataforma es un ecosistema completo de aplicaciones interactivas construidas con **Streamlit**, diseñadas para el análisis, simulación y proyección de series temporales financieras y macroeconómicas (con un enfoque especial en la dinámica del Dólar y el IPC). 

El proyecto combina econometría clásica, algoritmos de Machine Learning, arquitecturas de Deep Learning y modelos físicos aplicados a las finanzas (Econofísica).

## Módulos Disponibles

La plataforma está dividida en 6 consolas independientes que abordan el análisis de datos desde diferentes enfoques metodológicos:

1. **`1-modeloSarimaHoltWintersLstm.py` (Forecast Integral)**
   Compara simultáneamente proyecciones usando estadística clásica (SARIMA, Holt-Winters) y Redes Neuronales Recurrentes (LSTM).

2. **`2-modeloGRU-TCN-BILstm.py` (Deep Learning Projector)**
   Permite experimentar con arquitecturas avanzadas de Deep Learning para series temporales, incluyendo redes GRU, BiLSTM y Redes Convolucionales Temporales (TCN), con hiperparámetros ajustables.

3. **`3-modeloRandomForest.py` (Panel Unificado Random Forest)**
   Un panel automatizado que proyecta valores futuros mediante modelos ensamblados (Random Forest), calculando métricas de entropía, volatilidad y realizando descomposición estacional de la tendencia.

4. **`4-modeloVarGarch.py` (Analytics Macro: VAR & GARCH)**
   Analiza la relación e interdependencia vectorial entre el Dólar y la Inflación (IPC) usando Vectores Autorregresivos (VAR), betas móviles (Rolling Betas) y pronostica el riesgo/volatilidad mediante modelos GARCH(1,1).

5. **`5-modeloPass-Through.py` (Simulador Bilateral Pass-Through AI)**
   Simulador multivariable iterativo impulsado por Random Forest. Rompe la asunción del dólar como variable exógena y permite modelar un bucle de retroalimentación endógeno entre el Tipo de Cambio y la Inflación ante diferentes escenarios de shock.

6. **`6-modeloTermodinámica.py` (Econofísica Avanzada)**
   Traduce los movimientos históricos de las cotizaciones, spreads y brechas cambiarias en vectores de fuerzas cinemáticas (velocidad, aceleración/impulso) y analiza el mercado bajo leyes de la termodinámica (sincronización de fuerza y caos/entropía).

## Estructura de Datos Requerida

Para que los módulos funcionen correctamente en modo automático, debes colocar tus datasets en formato Excel (`.xlsx`) dentro de una carpeta llamada `data` en la raíz del proyecto.


## -------------------------------------------------------------------
Tecnologías Utilizadas o librerias usadas

Framework Web: Streamlit
Procesamiento de Datos: Pandas, NumPy
Visualización: Plotly Graph Objects
Machine Learning Clásico: Scikit-Learn
Estadística y Econometría: Statsmodels, Arch (Volatilidad)
Deep Learning: PyTorch, TensorFlow/Keras

## -------------------------------------------------------------------
La estructura esperada es:

├── data/
│   ├── Dolar.xlsx        # Histórico de cotizaciones (Compra, Venta, Promedio)
│   ├── ipc.xlsx          # Serie histórica del IPC General
│   └── dolar-ipc.xlsx    # Dataset consolidado para el simulador Pass-Through
├── 1-modeloSarimaHoltWintersLstm.py
├── ...
└── requirements.txt

# instalar el archivo
pip install -r requirements.txt

# para correr el proyecto 
streamlit run main.py


git clone (https://github.com/wgekko/analisis_dolar_modelos.git)
cd tu_repositorio
