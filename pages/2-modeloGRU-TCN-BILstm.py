import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(
    layout="wide", 
    page_title="Deep Learning: Proyección Dólar (Custom)", 
    page_icon=":material/graph_3:"
)

DATA_PATH = "data"

# =========================================================
# LÓGICA DE CARGA Y TRANSFORMACIÓN
# =========================================================
@st.cache_data
def load_dolar(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
            
        df.columns = [str(c).strip().upper() for c in df.columns]
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
        
        posibles = [c for c in df.columns if ('VENTA' in c or 'PRECIO' in c or 'PROMEDIO' in c or 'BLUE' in c)]
        if not posibles:
            posibles = [df.columns[1]]
            
        df = df.rename(columns={posibles[0]: 'PRECIO'})
        #df['PRECIO'] = pd.to_numeric(df['PRECIO'], errors='coerce')
        #df = df.dropna(subset=['FECHA', 'PRECIO']).sort_values('FECHA')        
        #df = df.set_index('FECHA').resample('D').ffill().reset_index()
        df['PRECIO'] = pd.to_numeric(df['PRECIO'], errors='coerce')
        df = df.dropna(subset=['FECHA', 'PRECIO']).sort_values('FECHA')    
        
        df = df.drop_duplicates(subset=['FECHA'], keep='last')
        
        df = df.set_index('FECHA').resample('D').ffill().reset_index()

        return df
    except Exception as e:
        st.error(f"Error cargando Dolar: {e}")
        return None

def prepare_data(data, lookback):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    # Usamos lookback para crear secuencias
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler

# =========================================================
# CLASES DE MODELOS
# =========================================================
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1) 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TCNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super(TCNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.relu(out)
        out = out[:, :, -1] 
        return self.fc(out)

# =========================================================
# INTERFAZ PRINCIPAL
# =========================================================
st.subheader(":material/integration_instructions: Analytics Deep Learning: Dinámica del Dólar")

# --- SIDEBAR: Configuración Dinámica ---
with st.sidebar:
    st.header(":material/settings: Configuración del Experimento")
    
    model_type = st.radio("Arquitectura:",["GRU", "BiLSTM", "TCN"])
    
    # 1. Filtro de Año (Sesgo macroeconómico)
    start_year = st.slider("Año de inicio del entrenamiento", 2019, 2026, 2020)
    
    # 2. Ventana de memoria (Lookback)
    lookback = st.slider("Ventana de memoria (Días hacia atrás)", 30, 365, 90)
    
    # 3. Horizonte de predicción
    horizonte = st.slider("Días a proyectar", 1, 60, 15)
    
    st.divider()
    ejecutar = st.button(":material/engineering: Ejecutar Predicción IA", type="primary", width='stretch')

# --- FLUJO ---
file_path = os.path.join(DATA_PATH, "Dolar.xlsx")
df_full = load_dolar(file_path)

if df_full is not None:
    # Filtramos la data antes de preparar
    df_filtered = df_full[df_full['FECHA'].dt.year >= start_year].copy()
    ts_data = df_filtered.set_index("FECHA")["PRECIO"]
    
    if len(ts_data) <= lookback:
        st.error(f":material/warning: Insuficientes datos desde {start_year} para una ventana de {lookback} días. Reduce el lookback o amplía el año de inicio.")
    else:
        if ejecutar:
            X, y, scaler = prepare_data(ts_data, lookback)
            
            with st.spinner(f"Entrenando con historia desde {start_year} y ventana de {lookback} días..."):
                if model_type == "GRU": model = GRUModel()
                elif model_type == "BiLSTM": model = BiLSTMModel()
                else: model = TCNModel()
                    
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                criterion = nn.MSELoss()
                
                for epoch in range(50):
                    model.train()
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()

                # Inferencia
                model.eval()
                current_seq = scaler.transform(ts_data.values[-lookback:].reshape(-1, 1))
                current_seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
                
                predicciones_escaladas = []
                with torch.no_grad():
                    for _ in range(horizonte):
                        pred_scaled = model(current_seq_tensor)
                        predicciones_escaladas.append(pred_scaled.item())
                        next_input = pred_scaled.unsqueeze(1)
                        current_seq_tensor = torch.cat((current_seq_tensor[:, 1:, :], next_input), dim=1)
                
                predicciones = scaler.inverse_transform(np.array(predicciones_escaladas).reshape(-1, 1)).flatten()
                
                # Gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_data.index[-90:], y=ts_data.values[-90:], name="Histórico (90d)", line=dict(color='gray')))
                
                dias_futuros = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq='D')
                fig.add_trace(go.Scatter(x=dias_futuros, y=predicciones, name=f"Proyección {model_type}", line=dict(color='#00d1b2', width=3, dash='dash')))
                
                fig.update_layout(title=f"Predicción (Modelo {model_type})", template="plotly_dark")
                st.plotly_chart(fig, width='stretch')
                
                # Tabla
                tabla_datos = pd.DataFrame({
                    "Fecha": dias_futuros.strftime('%d-%m-%Y'),
                    "Precio": [f"$ {p:.2f}" for p in predicciones]
                })
                st.dataframe(tabla_datos, width='stretch', hide_index=True)
        else:
            st.info(":material/transition_push: Configura los parámetros y presiona **Ejecutar**.")

st.write("   ")

st.divider()
