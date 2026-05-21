import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Modelos Estadísticos
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Machine Learning (LSTM)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Forecast Dolar: SARIMA, HW & LSTM", page_icon=":material/analytics:")

# --- 1. PROCESAMIENTO DE DATOS ---
@st.cache_data
def load_data_optimized(file_source):
    try:
        df = pd.read_excel(file_source, engine="openpyxl")
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        if 'FECHA' not in df.columns:
            return None
            
        if not pd.api.types.is_datetime64_any_dtype(df['FECHA']):
            fecha_num = pd.to_numeric(df['FECHA'], errors='coerce')
            if fecha_num.notna().sum() > 0 and (fecha_num.dropna().iloc[0] < 100000):
                df['FECHA'] = pd.to_datetime(fecha_num, unit='D', origin='1899-12-30')
            else:
                df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
                
        df = df.dropna(subset=['FECHA']).sort_values('FECHA').set_index('FECHA')
        
        # Filtrar solo columnas numéricas que nos interesan
        cols_interes = [c for c in df.columns if c in ['COMPRA', 'VENTA', 'PROMEDIO', 'BRECHA']]
        for c in cols_interes:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        return df[cols_interes].dropna()
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return None

# --- 2. FUNCIONES DE MODELOS (Cacheadas para velocidad) ---

@st.cache_data(show_spinner=False)
def run_sarima(ts_data, steps):
    model = SARIMAX(ts_data, order=(1, 1, 1)).fit(disp=False)
    forecast = model.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()

@st.cache_data(show_spinner=False)
def run_holtwinters(ts_data, steps):
    model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=7).fit()
    return model.forecast(steps)

@st.cache_data(show_spinner=False)
def run_lstm(ts_data, steps, lookback=10):
    # Preprocesamiento LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Construcción de la Red
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=16, epochs=20, verbose=0)
    
    # Predicción iterativa (Unistep)
    last_sequence = scaled_data[-lookback:].reshape((1, lookback, 1))
    predictions_scaled = []
    
    for _ in range(steps):
        next_pred = model.predict(last_sequence, verbose=0)[0][0]
        predictions_scaled.append(next_pred)
        next_pred_reshaped = np.array([[[next_pred]]])
        last_sequence = np.append(last_sequence[:, 1:, :], next_pred_reshaped, axis=1)
        
    return scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

# --- 3. INTERFAZ DE USUARIO ---
st.subheader(":material/finance_mode: Forecast Integral Dólar: Estadística Clásica vs Redes Neuronales")
st.markdown("Comparativa simultánea: **SARIMA, Holt-Winters y LSTM**")

# Carga de archivo
file_upload = st.file_uploader("Sube tu archivo Dolar.xlsx", type=["xlsx"])
path_local = os.path.join("data", "Dolar.xlsx")
file_source = file_upload if file_upload is not None else (path_local if os.path.exists(path_local) else None)

if file_source:
    df = load_data_optimized(file_source)
    if df is not None and not df.empty:
        st.success(":material/check_circle: Datos cargados con éxito.")
        st.subheader(":material/settings_applications: Configuración " )
        # Controles
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Selecciona la serie a predecir:", df.columns)
        with col2:
            steps = st.slider("Días a proyectar:", min_value=1, max_value=30, value=7)
            
        ts_data = df[selected_col].dropna()
        f_dates = pd.date_range(ts_data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        
        with st.spinner(':material/memory: Ejecutando motores de predicción (SARIMA, Holt-Winters, LSTM)...'):
            # Ejecutar modelos
            mean_sarima, conf_sarima = run_sarima(ts_data, steps)
            preds_hw = run_holtwinters(ts_data, steps)
            preds_lstm = run_lstm(ts_data, steps)
            
            # --- GRÁFICO COMPARATIVO ---
            st.subheader(f":material/open_in_new: Proyección a {steps} días: {selected_col}")
            fig = go.Figure()
            
            # Histórico (últimos 90 días para claridad)
            hist = ts_data.tail(90)
            fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Histórico", line=dict(color='white', width=2)))
            
            # SARIMA + Intervalos
            fig.add_trace(go.Scatter(x=f_dates, y=mean_sarima, name="SARIMA", line=dict(dash='dot', color='#00d1b2')))
            fig.add_trace(go.Scatter(
                x=f_dates.append(f_dates[::-1]),
                y=np.concatenate([conf_sarima.iloc[:, 1], conf_sarima.iloc[:, 0][::-1]]),
                fill='toself', fillcolor='rgba(0, 209, 178, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                showlegend=True, name="Rango Confianza (SARIMA)"
            ))
            
            # Holt-Winters
            fig.add_trace(go.Scatter(x=f_dates, y=preds_hw, name="Holt-Winters", line=dict(dash='dash', color='#ffdd57')))
            
            # LSTM
            fig.add_trace(go.Scatter(x=f_dates, y=preds_lstm, name="Red LSTM (IA)", line=dict(color='#ff3860', width=2)))
            
            fig.update_layout(template="plotly_dark", height=600, hovermode="x unified", yaxis_title="Precio en $")
            st.plotly_chart(fig, width='stretch')
            
            # --- TABLA UNIFICADA ---
            with st.expander(":material/table_chart: Ver tabla comparativa de valores proyectados"):
                df_tabla = pd.DataFrame({
                    "Fecha": f_dates.strftime('%d/%m/%Y'),
                    "Holt-Winters": preds_hw.values,
                    "SARIMA (Base)": mean_sarima.values,
                    "SARIMA (Mín)": conf_sarima.iloc[:, 0].values,
                    "SARIMA (Máx)": conf_sarima.iloc[:, 1].values,
                    "Red LSTM": preds_lstm
                }).set_index("Fecha")
                
                # Resaltamos el modelo IA (LSTM) y el estadístico principal (SARIMA)
                st.dataframe(
                    df_tabla.style.format("${:.2f}")
                    .highlight_max(subset=["Red LSTM", "SARIMA (Base)", "Holt-Winters"], color="#3d0000")
                    .highlight_min(subset=["Red LSTM", "SARIMA (Base)", "Holt-Winters"], color="#002b1a"),
                    width='stretch'
                )
                
                # Exportar
                csv = df_tabla.to_csv().encode('utf-8')
                st.download_button(
                    label=":material/download: Descargar Proyecciones Combinadas (CSV)",
                    data=csv,
                    file_name=f'forecast_integral_{selected_col.lower()}_{steps}d.csv',
                    mime='text/csv'
                )
    else:
        st.warning(":material/warning: El archivo no contiene datos válidos o faltan columnas.")
else:
    st.info(":material/upload: Sube un archivo Excel para comenzar.")