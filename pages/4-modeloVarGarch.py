import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
import os
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Importación para el modelo GARCH
try:
    from arch import arch_model
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    layout="wide",
    page_title="Dolar-IPC-Vector Autoregression",
    page_icon=":material/monitoring:"
)

DATA_PATH = "data"

# =========================================================
# FUNCIONES AUXILIARES Y DE CARGA
# =========================================================

def get_file(filename):
    path = os.path.join(DATA_PATH, filename)
    return path if os.path.exists(path) else None

@st.cache_data
def load_dolar(file_source):
    df = pd.read_excel(file_source, engine="openpyxl")
    df.columns = [str(c).strip().upper() for c in df.columns]
    df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
    
    posibles = [c for c in df.columns if ('VENTA' in c or 'PRECIO' in c or 'PROMEDIO' in c)]
    if not posibles:
        raise ValueError("No se encontró columna de precio en Dolar.xlsx")
    
    df = df.rename(columns={posibles[0]: 'PRECIO'})
    df['PRECIO'] = pd.to_numeric(df['PRECIO'], errors='coerce')
    df = df.dropna(subset=['FECHA', 'PRECIO']).sort_values('FECHA')
    return df

@st.cache_data
def load_ipc(file_source):
    try:
        df_raw = pd.read_excel(file_source, engine="openpyxl", header=None)
        mask = df_raw[0].astype(str).str.contains('TotalNacional-Nivelgeneral', case=False, na=False)
        if not mask.any():
            raise ValueError("No se encontró 'TotalNacional-Nivelgeneral'")
            
        row_idx = mask[mask].index[0]
        fechas = pd.to_datetime(df_raw.iloc[row_idx - 1, 1:], errors='coerce')
        valores = pd.to_numeric(df_raw.iloc[row_idx, 1:], errors='coerce')
        
        series = pd.Series(valores.values, index=fechas)
        series = series[series.index.notna()]
        series.index = pd.to_datetime(series.index).to_period('M').to_timestamp()
        series = series[~series.index.duplicated(keep='first')].sort_index().dropna()
        series.name = 'IPC'
        return series
    except Exception as e:
        st.error(f"Error cargando IPC: {e}")
        return None

# =========================================================
# MODELOS Y CÁLCULOS ESTADÍSTICOS
# =========================================================

def run_var_forecast(df_dolar, series_ipc):
    dolar_m = df_dolar.set_index('FECHA')['PRECIO'].resample('MS').mean()
    dolar_m.index = dolar_m.index.to_period('M').to_timestamp()
    dolar_m.name = 'Dolar'
    
    data = pd.concat([dolar_m, series_ipc], axis=1)
    data.columns = ['Dolar', 'IPC']
    data = data.dropna()
    data = data[(data['Dolar'] > 0) & (data['IPC'] > 0)]
    
    if data.empty or len(data) < 12:
        raise ValueError("Datos insuficientes para VAR")
        
    data_diff = np.log(data).diff().replace([np.inf, -np.inf], np.nan).dropna()
    
    model = VAR(data_diff)
    results = model.fit(ic='aic', maxlags=6)
    lag_order = results.k_ar
    
    forecast_diff = results.forecast(data_diff.values[-lag_order:], steps=3)
    
    last_vals = data.iloc[-1].values.copy()
    proyecciones = []
    curr = last_vals.copy()
    
    for f in forecast_diff:
        curr = curr * np.exp(f)
        proyecciones.append(curr.copy())
        
    idx = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq='MS')
    return pd.DataFrame(proyecciones, columns=['Dolar', 'IPC'], index=idx), data

def calcular_beta(x, y):
    df = pd.concat([x, y], axis=1).dropna()
    df.columns = ['X', 'Y']
    X = sm.add_constant(df['X'])
    model = sm.OLS(df['Y'], X).fit()
    return model, model.params['X'], model.params['const'], model.rsquared, df

def calcular_rolling_beta(x, y, window):
    """Calcula el coeficiente Beta de forma móvil usando una ventana de tiempo fija."""
    df = pd.concat([x, y], axis=1).dropna()
    df.columns = ['X', 'Y']
    betas = []
    fechas = []
    
    for i in range(window, len(df) + 1):
        sub_df = df.iloc[i - window:i]
        X = sm.add_constant(sub_df['X'])
        model = sm.OLS(sub_df['Y'], X).fit()
        betas.append(model.params['X'])
        fechas.append(sub_df.index[-1]) # Asigna el valor al final de la ventana
        
    return pd.Series(betas, index=fechas)

# =========================================================
# INTERFAZ
# =========================================================

st.subheader("Analytics Macro: VAR & GARCH Volatility")
st.markdown("---")

path_dolar = get_file("Dolar.xlsx")
path_ipc = get_file("ipc.xlsx")

col1, col2 = st.columns(2)
file_dolar = col1.file_uploader("Subir Dolar.xlsx", type=["xlsx"])
file_ipc = col2.file_uploader("Subir ipc.xlsx", type=["xlsx"])

source_d = path_dolar if path_dolar else file_dolar
source_i = path_ipc if path_ipc else file_ipc

if source_d and source_i:
    try:
        df_dolar = load_dolar(source_d)
        series_ipc = load_ipc(source_i)
        
        if series_ipc is None:
            st.stop()
            
        st.success(":material/done_all: Archivos cargados correctamente")
        
        # --- SECCIÓN VAR ---
        st.subheader(":material/line_axis: 1. Proyección de Precios (Modelo VAR)")
        forecast_df, data_hist = run_var_forecast(df_dolar, series_ipc)
        
        st.dataframe(forecast_df.style.format("{:.2f}"))
        
        c1, c2 = st.columns(2)
        with c1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=data_hist.index, y=data_hist['Dolar'], mode='lines', name='Histórico Dólar', line=dict(color='white')))
            fig1.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Dolar'], mode='lines+markers', name='Forecast VAR', line=dict(color='#00d1b2', dash='dash')))
            fig1.update_layout(title="Proyección Dólar", template="plotly_dark")
            st.plotly_chart(fig1, width='stretch')
            
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data_hist.index, y=data_hist['IPC'], mode='lines', name='Histórico IPC', line=dict(color='white')))
            fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['IPC'], mode='lines+markers', name='Forecast VAR', line=dict(color='#ff3860', dash='dash')))
            fig2.update_layout(title="Proyección IPC", template="plotly_dark")
            st.plotly_chart(fig2, width='stretch')

        st.markdown("---")
        
        # --- SECCIÓN REGRESIÓN (BETA HISTÓRICO) ---
        st.subheader(":material/stacked_bar_chart: 2. Sensibilidad Estructural Histórica (Regresión OLS)")
        dolar_ret = np.log(data_hist['Dolar']).diff().dropna()
        ipc_ret = np.log(data_hist['IPC']).diff().dropna()

        model1, beta1, alpha1, r21, df1 = calcular_beta(dolar_ret, ipc_ret)
        model2, beta2, alpha2, r22, df2 = calcular_beta(ipc_ret, dolar_ret)

        cb1, cb2 = st.columns(2)
        with cb1:
            with st.container(border=True):
                st.subheader("Beta Dólar → IPC (Toda la Serie)")
                st.metric("Beta (Impacto)", f"{beta1:.4f}")
                st.metric("R² (Explicabilidad)", f"{r21:.4f}")
                
            fig_beta1 = go.Figure()
            fig_beta1.add_trace(go.Scatter(x=df1['X'], y=df1['Y'], mode='markers', name='Datos', marker=dict(color='#00d1b2', opacity=0.6)))
            fig_beta1.add_trace(go.Scatter(x=df1['X'], y=model1.predict(sm.add_constant(df1['X'])), mode='lines', name='Regresión', line=dict(color='yellow')))
            fig_beta1.update_layout(title='Beta Dólar → IPC (Histórico)', xaxis_title='Variación Dólar', yaxis_title='Variación IPC', template="plotly_dark")
            st.plotly_chart(fig_beta1, width='stretch')

        with cb2:
            with st.container(border=True):
                st.subheader("Beta IPC → Dólar (Toda la Serie)")
                st.metric("Beta (Impacto)", f"{beta2:.4f}")
                st.metric("R² (Explicabilidad)", f"{r22:.4f}")

            fig_beta2 = go.Figure()
            fig_beta2.add_trace(go.Scatter(x=df2['X'], y=df2['Y'], mode='markers', name='Datos', marker=dict(color='#ff3860', opacity=0.6)))
            fig_beta2.add_trace(go.Scatter(x=df2['X'], y=model2.predict(sm.add_constant(df2['X'])), mode='lines', name='Regresión', line=dict(color='yellow')))
            fig_beta2.update_layout(title='Beta IPC → Dólar (Histórico)', xaxis_title='Variación IPC', yaxis_title='Variación Dólar', template="plotly_dark")
            st.plotly_chart(fig_beta2, width='stretch')

        st.markdown("---")

        # --- SECCIÓN REGRESIÓN RECIENTE (VENTANA FIJA CORTA) ---
        st.subheader(":material/bar_chart: 2b. Sensibilidad Reciente (Ventana Temporal Corta)")
        
        ventana_meses = st.selectbox(
            "Seleccionar ventana de meses históricos recientes:",
            options=[24, 12],
            index=0,
            help="24 meses ofrece estabilidad estadística elemental. 12 meses captura el régimen inmediato pero es muy sensible al ruido muestral."
        )
        
        dolar_ret_rec = dolar_ret.tail(ventana_meses)
        ipc_ret_rec = ipc_ret.tail(ventana_meses)

        model1_rec, beta1_rec, alpha1_rec, r21_rec, df1_rec = calcular_beta(dolar_ret_rec, ipc_ret_rec)
        model2_rec, beta2_rec, alpha2_rec, r22_rec, df2_rec = calcular_beta(ipc_ret_rec, dolar_ret_rec)

        cbr1, cbr2 = st.columns(2)
        with cbr1:
            with st.container(border=True):
                st.subheader(f"Beta Dólar → IPC (Últimos {ventana_meses} meses)")
                st.metric("Beta (Impacto Reciente)", f"{beta1_rec:.4f}")
                st.metric("R² (Explicabilidad Reciente)", f"{r21_rec:.4f}")
                
            fig_beta1_rec = go.Figure()
            fig_beta1_rec.add_trace(go.Scatter(x=df1_rec['X'], y=df1_rec['Y'], mode='markers', name='Datos Recientes', marker=dict(color='#00d1b2', opacity=0.8)))
            fig_beta1_rec.add_trace(go.Scatter(x=df1_rec['X'], y=model1_rec.predict(sm.add_constant(df1_rec['X'])), mode='lines', name='Regresión Reciente', line=dict(color='#ffdd57')))
            fig_beta1_rec.update_layout(title=f'Beta Dólar → IPC (Últimos {ventana_meses} Meses)', xaxis_title='Variación Dólar', yaxis_title='Variación IPC', template="plotly_dark")
            st.plotly_chart(fig_beta1_rec, width='stretch')

        with cbr2:
            with st.container(border=True):
                st.subheader(f"Beta IPC → Dólar (Últimos {ventana_meses} meses)")
                st.metric("Beta (Impacto Reciente)", f"{beta2_rec:.4f}")
                st.metric("R² (Explicabilidad Reciente)", f"{r22_rec:.4f}")

            fig_beta2_rec = go.Figure()
            fig_beta2_rec.add_trace(go.Scatter(x=df2_rec['X'], y=df2_rec['Y'], mode='markers', name='Datos Recientes', marker=dict(color='#ff3860', opacity=0.8)))
            fig_beta2_rec.add_trace(go.Scatter(x=df2_rec['X'], y=model2_rec.predict(sm.add_constant(df2_rec['X'])), mode='lines', name='Regresión Reciente', line=dict(color='#ffdd57')))
            fig_beta2_rec.update_layout(title=f'Beta IPC → Dólar (Últimos {ventana_meses} Meses)', xaxis_title='Variación IPC', yaxis_title='Variación Dólar', template="plotly_dark")
            st.plotly_chart(fig_beta2_rec, width='stretch')

        st.markdown("---")

        # --- NUEVA SECCIÓN 2c: ANÁLISIS DINÁMICO DE BETAS MÓVILES ---
        st.header(":material/grouped_bar_chart: 2c. Evolución Dinámica de la Sensibilidad (Rolling Betas)")
        st.write("Visualiza cómo muta la correlación estructural mes a mes. Cada punto en la línea representa el coeficiente Beta estimado usando únicamente los N meses previos a esa fecha.")
        
        ventana_rolling = st.slider(
            "Seleccionar tamaño de ventana móvil (meses de estimación):", 
            min_value=6, 
            max_value=36, 
            value=24, 
            step=2,
            help="Ventanas más cortas (ej. 12m) reaccionan rápido a crisis pero oscilan mucho. Ventanas más largas (ej. 24m) suavizan tendencias estructurales."
        )
        
        if len(dolar_ret) >= ventana_rolling:
            rolling_beta1 = calcular_rolling_beta(dolar_ret, ipc_ret, ventana_rolling)
            rolling_beta2 = calcular_rolling_beta(ipc_ret, dolar_ret, ventana_rolling)
            
            cro1, cro2 = st.columns(2)
            with cro1:
                fig_roll1 = go.Figure()
                fig_roll1.add_trace(go.Scatter(x=rolling_beta1.index, y=rolling_beta1.values, mode='lines', name='Beta Móvil', line=dict(color='#00d1b2', width=2.5)))
                # Línea de referencia horizontal con el Beta histórico de toda la muestra
                fig_roll1.add_hline(y=beta1, line_dash="dash", line_color="#ffdd57", annotation_text="Beta Histórico Global", annotation_position="top left")
                fig_roll1.update_layout(title=f'Evolución del Coeficiente: Dólar → IPC (Muestra de {ventana_rolling}m)', xaxis_title='Fecha de Evaluación', yaxis_title='Magnitud del Beta', template="plotly_dark")
                st.plotly_chart(fig_roll1, width='stretch')
                
            with cro2:
                fig_roll2 = go.Figure()
                fig_roll2.add_trace(go.Scatter(x=rolling_beta2.index, y=rolling_beta2.values, mode='lines', name='Beta Móvil', line=dict(color='#ff3860', width=2.5)))
                # Línea de referencia horizontal con el Beta histórico de toda la muestra
                fig_roll2.add_hline(y=beta2, line_dash="dash", line_color="#ffdd57", annotation_text="Beta Histórico Global", annotation_position="top left")
                fig_roll2.update_layout(title=f'Evolución del Coeficiente: IPC → Dólar (Muestra de {ventana_rolling}m)', xaxis_title='Fecha de Evaluación', yaxis_title='Magnitud del Beta', template="plotly_dark")
                st.plotly_chart(fig_roll2, width='stretch')
        else:
            st.warning(f":material/warning: La muestra consolidada posee únicamente {len(dolar_ret)} observaciones, insuficientes para computar ventanas rolling de {ventana_rolling} meses.")

        st.markdown("---")

        # --- SECCIÓN GARCH (VOLATILIDAD) ---
        st.subheader(":material/insert_chart: 3. Predicción de Riesgo: Modelo GARCH(1,1)")
        st.write("Análisis de clústeres de volatilidad utilizando rendimientos diarios del Dólar.")
        
        if GARCH_AVAILABLE:
            with st.spinner(':material/settings_b_roll: Ajustando modelo GARCH de volatilidad...'):
                ts_dolar_diario = df_dolar.set_index('FECHA')['PRECIO'].asfreq('D').ffill()
                retornos_diarios = 100 * np.log(ts_dolar_diario).diff().dropna()
                
                am = arch_model(retornos_diarios, vol='Garch', p=1, o=0, q=1, dist='Normal')
                res_garch = am.fit(disp='off')
                
                horizonte = 30
                forecasts = res_garch.forecast(horizon=horizonte)
                
                pred_volatilidad = np.sqrt(forecasts.variance.values[-1, :])
                fechas_futuras = pd.date_range(start=ts_dolar_diario.index[-1] + pd.Timedelta(days=1), periods=horizonte, freq='D')
                
                cg1, cg2, cg3 = st.columns(3)
                vol_actual = res_garch.conditional_volatility.iloc[-1]
                vol_futura_promedio = pred_volatilidad.mean()
                tendencia = vol_futura_promedio - vol_actual
                
                with st.container(border=True):
                    cg1.metric("Volatilidad Diaria Actual", f"{vol_actual:.2f}%")
                    cg2.metric("Volatilidad Proyectada (30d)", f"{vol_futura_promedio:.2f}%", delta=f"{tendencia:.2f}%", delta_color="inverse")
                    cg3.metric("Medio Término (Omega)", f"{res_garch.params['omega']:.4f}")
                
                fig_garch = go.Figure()
                hist_vol = res_garch.conditional_volatility.tail(180)
                fig_garch.add_trace(go.Scatter(x=hist_vol.index, y=hist_vol, mode='lines', name='Volatilidad Histórica', line=dict(color='#ffdd57')))
                fig_garch.add_trace(go.Scatter(x=fechas_futuras, y=pred_volatilidad, mode='lines', name='Forecast GARCH', line=dict(color='#ff3860', dash='dot', width=3)))
                
                fig_garch.update_layout(
                    title=":material/query_stats: Evolución y Pronóstico de la Volatilidad Condicional",
                    yaxis_title="Volatilidad (%)",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_garch, width='stretch')
                
                with st.expander(":material/monitoring: Ver Resumen Estadístico GARCH"):
                    st.text(res_garch.summary().as_text())
        else:
            st.warning(":material/warning: La librería 'arch' no está instalada. Ejecuta `pip install arch` en tu terminal para ver el análisis de volatilidad.")

    except Exception as e:
        st.error(f":material/error: Error procesando datos: {e}")
else:
    st.info(":material/upload_file: Subí los archivos Dolar.xlsx e ipc.xlsx para comenzar el análisis macroeconómico.")