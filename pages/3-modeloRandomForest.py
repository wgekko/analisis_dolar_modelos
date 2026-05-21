import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from datetime import datetime, timedelta

# Modelos Estadísticos y ML
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    layout="wide", 
    page_title="Fintech Analytics - Proyección Dólar Blue", 
    page_icon=":material/bar_chart_4_bars:"
)

# --- ESTILOS PERSONALIZADOS ADAPTADOS A TU CONFIG.TOML ---
st.markdown("""
    <style>
    .main-title {
        font-size: 28pt;
        font-weight: bold;
        color: #FF8C00; /* Naranja principal de tu config */
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 12pt;
        color: #E1C16E; /* Mostaza claro */
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-box {
        background-color: #1B263B; /* Tu secondaryBackgroundColor */
        border: 1px solid #CCCCCC;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Panel Unificado de Proyección del Dólar</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modelos Aplicado a Series Financieras Diarias (Estadísticos y ML)</div>', unsafe_allow_html=True)

# --- 1. PROCESAMIENTO Y CARGA DE DATOS DESDE CARPETA "data" ---
@st.cache_data
def load_and_transform_dolar(file_path):
    """Carga, limpia y transforma el archivo de Dólar Blue mapeando las fechas numéricas de Excel."""
    if not os.path.exists(file_path):
        return None
    try:
        xl = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_name = "u$s BLUE" if "u$s BLUE" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet_name)
        
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        if 'FECHA' not in df.columns:
            st.error(":material/warning: No se encontró la columna 'FECHA' en el archivo.")
            return None
            
        def parse_excel_date(x):
            try:
                serial = float(x)
                return pd.to_datetime('1899-12-30') + pd.to_timedelta(serial, 'D')
            except:
                return pd.to_datetime(x, errors='coerce')
                
        df['FECHA'] = df['FECHA'].apply(parse_excel_date)
        df = df[df['FECHA'].notna()]
        
        df = df.set_index('FECHA')
        df = df.sort_index()
        
        valid_cols = [c for c in ['COMPRA', 'VENTA', 'PROMEDIO', 'BRECHA'] if c in df.columns]
        df_clean = df[valid_cols].apply(pd.to_numeric, errors='coerce')
        
        df_clean = df_clean.ffill().bfill()
        return df_clean
    except Exception as e:
        st.error(f":material/warning: Error al procesar el archivo Excel: {e}")
        return None

default_path = os.path.join("data", "Dolar.xlsx")
df_ts = None

st.sidebar.header(":material/settings_alert: Configuración del Análisis")
if os.path.exists(default_path):
    st.sidebar.success(f":material/folder_code: Archivo detectado automáticamente en: `{default_path}`")
    df_ts = load_and_transform_dolar(default_path)
else:
    uploaded_file = st.sidebar.file_uploader("O cargar archivo Excel manualmente:", type=["xlsx"])
    if uploaded_file is not None:
        df_ts = load_and_transform_dolar(uploaded_file)

# --- 2. MODELO DE MACHINE LEARNING ---
def train_ml_forecast(df_series, forecast_steps=7, look_back=15):
    df_ml = df_series.copy()
    col_name = df_ml.columns[0]
    
    for i in range(1, look_back + 1):
        df_ml[f'lag_{i}'] = df_ml[col_name].shift(i)
        
    df_ml = df_ml.dropna()
    X = df_ml.drop(columns=[col_name]).values
    y = df_ml[col_name].values
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    predictions = []
    last_known = list(df_series.values[-look_back:].flatten())
    
    for _ in range(forecast_steps):
        input_features = np.array(last_known[-look_back:][::-1]).reshape(1, -1)
        pred = rf.predict(input_features)[0]
        predictions.append(pred)
        last_known.append(pred)
        
    return np.array(predictions)

# --- 3. PROCESAMIENTO PRINCIPAL DE INTERFAZ ---
if df_ts is not None:
    selected_col = st.sidebar.selectbox("Seleccione la Variable a Modelar:", options=df_ts.columns)
    horizonte = st.sidebar.slider("Horizonte de Predicción (Días Corridos):", min_value=1, max_value=30, value=7)
    
    ts_data = df_ts[selected_col].dropna()
    
    # --- CÁLCULO DE MÉTRICAS CLAVE ---
    hist_mean = ts_data.mean()
    volatilidad = ts_data.std()
    
    # NUEVO: Filtro dinámico para obtener el promedio de los últimos 6 meses (180 días)
    ult_fecha = ts_data.index[-1]
    fecha_6m_atras = ult_fecha - pd.Timedelta(days=180)
    mean_6m = ts_data.loc[fecha_6m_atras:ult_fecha].mean()
    
    counts, _ = np.histogram(ts_data, bins=15)
    prob = counts / sum(counts)
    ent_val = entropy(prob, base=2) if sum(counts) > 0 else 0
    
    # CAMBIO AQUÍ: Ahora se distribuyen en 5 columnas de igual tamaño
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.markdown(f'<div class="metric-box"><strong style="color:#FFA500;">Último Cierre Real</strong><br><span style="font-size:18pt; color:#FFF; font-weight:bold;">${ts_data.values[-1]:.2f}</span><br><small style="color:#CCCCCC;">{ts_data.index[-1].strftime("%d/%m/%Y")}</small></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-box"><strong style="color:#FFA500;">Promedio Histórico</strong><br><span style="font-size:18pt; color:#FFF; font-weight:bold;">${hist_mean:.2f}</span><br><small style="color:#CCCCCC;">Toda la serie analizada</small></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-box"><strong style="color:#40C4FF;">Promedio 6 Meses</strong><br><span style="font-size:18pt; color:#FFF; font-weight:bold;">${mean_6m:.2f}</span><br><small style="color:#CCCCCC;">Últimos 180 días directos</small></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-box"><strong style="color:#FFA500;">Volatilidad (σ)</strong><br><span style="font-size:18pt; color:#FFF; font-weight:bold;">{volatilidad:.2f}</span><br><small style="color:#CCCCCC;">Desviación estándar</small></div>', unsafe_allow_html=True)
    with col_m5:
        st.markdown(f'<div class="metric-box"><strong style="color:#FFA500;">Entropía</strong><br><span style="font-size:18pt; color:#FFF; font-weight:bold;">{ent_val:.2f}</span><br><small style="color:#CCCCCC;">Incertidumbre/Ruido</small></div>', unsafe_allow_html=True)

    st.write("---")

    tab_grafica, tab_modelos, tab_descomp = st.tabs([":material/finance_mode: Gráfico de Proyecciones Dinámicas", ":material/schema: Comparativa de Modelos", ":material/chart_data: Descomposición Financiera"])

    f_dates = pd.date_range(ts_data.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq='D')
    
    with st.spinner(':material/clock_loader_40: Procesando algoritmos predictivos...'):
        try:
            model_sarima = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,0,0,7), enforce_stationarity=False).fit(disp=False)
            pred_sarima = model_sarima.forecast(steps=horizonte)
        except:
            pred_sarima = np.full(horizonte, ts_data.values[-1])
            
        try:
            model_hw = ExponentialSmoothing(ts_data, trend='add', seasonal=None).fit()
            pred_hw = model_hw.forecast(steps=horizonte)
        except:
            pred_hw = np.full(horizonte, ts_data.values[-1])
            
        try:
            pred_rf = train_ml_forecast(ts_data.to_frame(), forecast_steps=horizonte)
        except:
            pred_rf = np.full(horizonte, ts_data.values[-1])

    with tab_grafica:
        st.subheader("Simulación Multimodelo de Tendencias de Mercado")
        
        dias_visibles = st.slider("Días históricos a visualizar en pantalla:", min_value=30, max_value=365, value=90)
        hist_reciente = ts_data.tail(dias_visibles)
        
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Scatter(
            x=hist_reciente.index, y=hist_reciente.values,
            name="Cotización Real", line=dict(color="#40C4FF", width=3), mode="lines"
        ))
        
        x_fut = [hist_reciente.index[-1]] + list(f_dates)
        
        fig_comp.add_trace(go.Scatter(x=x_fut, y=[hist_reciente.values[-1]] + list(pred_sarima), name="Proyección SARIMA", line=dict(dash='dot', color='#FFD700', width=2.5)))
        fig_comp.add_trace(go.Scatter(x=x_fut, y=[hist_reciente.values[-1]] + list(pred_hw), name="Proyección Holt-Winters", line=dict(dash='dash', color='#FFA500', width=2.5)))
        fig_comp.add_trace(go.Scatter(x=x_fut, y=[hist_reciente.values[-1]] + list(pred_rf), name="Proyección Random Forest", line=dict(dash='dashdot', color='#FF5252', width=3)))
        
        fig_comp.update_layout(
            title=f"Historial Cercano y Escenarios de Predicción para: Dólar ({selected_col})",
            xaxis_title="Fecha",
            yaxis_title="Precio ($)",
            hovermode="x unified",
            template="plotly_dark",
            plot_bgcolor="#1B263B",
            paper_bgcolor="#0D1B2A",
            height=550,
            xaxis=dict(showgrid=True, gridcolor="#2C3E50"),
            yaxis=dict(showgrid=True, gridcolor="#2C3E50")
        )
        st.plotly_chart(fig_comp, width='stretch')

    with tab_modelos:
        st.subheader(":material/qr_code_2: Matrices Comparativas de Proyección")
        
        df_proyecciones = pd.DataFrame({
            "Fecha": f_dates.strftime('%d/%m/%Y'),
            "Estadístico (SARIMA)": pred_sarima,
            "Suavizado (Holt-Winters)": pred_hw,
            "Machine Learning (RF)": pred_rf
        }).set_index("Fecha")
        
        col_t1, col_t2 = st.columns([2, 1])
        
        with col_t1:
            st.markdown("**Valores Líquidos Estimados ($):**")
            st.dataframe(df_proyecciones, width='stretch')
            
            csv_data = df_proyecciones.to_csv().encode('utf-8')
            st.download_button(
                label=":material/system_update_alt: Exportar Proyecciones a CSV",
                data=csv_data,
                file_name=f"proyecciones_blue_{selected_col.lower()}.csv",
                mime="text/csv"
            )
            
        with col_t2:
            st.markdown("**Consenso de Mercado Calculado:**")
            consenso_medio = df_proyecciones.mean(axis=1)
            dispersion = df_proyecciones.std(axis=1)
            
            df_consenso = pd.DataFrame({
                "Precio Consenso Promedio": consenso_medio,
                "Dispersión ($)": dispersion
            })
            st.dataframe(df_consenso.style.format("${:.2f}"), width='stretch')

    with tab_descomp:
        st.subheader(":material/stacked_line_chart: Descomposición de Tendencias y Ciclos Semanales")
        if len(ts_data) >= 30:
            try:
                res = seasonal_decompose(ts_data, model='additive', period=7)
                
                fig_desc = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Tendencia Primaria", "Efecto Estacional Semanal", "Residuos Macroeconómicos / Ruido"])
                
                fig_desc.add_trace(go.Scatter(x=res.trend.index, y=res.trend.values, name="Tendencia", line=dict(color="#FF8C00"), mode='lines'), row=1, col=1)
                fig_desc.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal.values, name="Estacionalidad", line=dict(color="#C8E25D"), mode='lines'), row=2, col=1)
                fig_desc.add_trace(go.Scatter(x=res.resid.index, y=res.resid.values, name="Residuo", line=dict(color="#FF5252"), mode='markers'), row=3, col=1)
                
                fig_desc.update_layout(
                    height=600, 
                    template="plotly_dark",
                    plot_bgcolor="#1B263B",
                    paper_bgcolor="#0D1B2A",
                    showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor="#2C3E50"),
                    yaxis=dict(showgrid=True, gridcolor="#2C3E50"),
                    xaxis2=dict(showgrid=True, gridcolor="#2C3E50"),
                    yaxis2=dict(showgrid=True, gridcolor="#2C3E50"),
                    xaxis3=dict(showgrid=True, gridcolor="#2C3E50"),
                    yaxis3=dict(showgrid=True, gridcolor="#2C3E50")
                )
                st.plotly_chart(fig_desc, width='stretch')
            except Exception as e:
                st.error(f":material/error: No se pudo realizar la descomposición analítica: {e}")
        else:
            st.warning(":material/warning: Se requieren registros más extensos para computar descomposiciones analíticas.")
else:
    st.error(":material/error: Error crítico: Verifique que exista el archivo en la ruta 'data/Dolar.xlsx' o cárguelo de forma manual.")