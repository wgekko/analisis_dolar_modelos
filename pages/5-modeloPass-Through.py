import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import os

# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="Simulador Bilateral Pass-Through AI",
    page_icon=":material/chart_data:"
)

DATA_PATH = "data"
FILE_NAME = "dolar-ipc.xlsx"

# =========================================================
# CARGA Y PREPARACIÓN DE DATOS OPTIMIZADOS
# =========================================================
@st.cache_data
def load_and_prepare_data(filepath):
    try:
        if filepath.endswith('.csv'):
            df_raw = pd.read_csv(filepath)
        else:
            df_raw = pd.read_excel(filepath, engine="openpyxl")
        
        # Mapeo estricto del nuevo dataset optimizado
        columnas_necesarias = {
            'fecha': 'FECHA',
            'Dolar-venta': 'PCT_DOLAR',
            'TotalNacional-Nivelgeneral': 'PCT_IPC'
        }
        
        # Filtrar y renombrar columnas clave
        df = df_raw[list(columnas_necesarias.keys())].rename(columns=columnas_necesarias)
        
        # Tratamiento del índice temporal
        df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
        df = df.dropna(subset=['FECHA']).sort_values('FECHA').set_index('FECHA')
        
        # Casteo numérico de seguridad
        df['PCT_DOLAR'] = pd.to_numeric(df['PCT_DOLAR'], errors='coerce')
        df['PCT_IPC'] = pd.to_numeric(df['PCT_IPC'], errors='coerce')
        df = df.dropna()
        
        return df
        
    except Exception as e:
        st.error(f":material/error: Error al procesar el archivo unificado: {e}")
        return None

# =========================================================
# FEATURE ENGINEERING: MATRIZ DE REZAGOS CRUZADOS
# =========================================================
def build_lagged_features(df, max_lags=4):
    df_features = df.copy()
    feature_cols = []
    
    for lag in range(1, max_lags + 1):
        df_features[f'DOLAR_LAG_{lag}'] = df_features['PCT_DOLAR'].shift(lag)
        df_features[f'IPC_LAG_{lag}'] = df_features['PCT_IPC'].shift(lag)
        feature_cols.extend([f'DOLAR_LAG_{lag}', f'IPC_LAG_{lag}'])
        
    return df_features.dropna(), feature_cols

# =========================================================
# MOTOR DE SIMULACIÓN MULTIVARIABLE ITERATIVO
# =========================================================
def ejecutar_simulacion(model_ipc, model_dolar, df_historico, predictores, lags, meses, modo, parametro_cambiario):
    """
    Simula de forma recursiva el comportamiento del IPC y el Dólar,
    retroalimentando los rezagos con las propias predicciones del sistema.
    """
    ultima_fila = df_historico.iloc[-1].copy()
    
    # Clonamos el estado final conocido de la economía
    estado_actual = {col: ultima_fila[col] for col in df_historico.columns}
    
    proyecciones_ipc = []
    proyecciones_dolar = []
    
    for m in range(meses):
        # Crear vector de entrada X para los modelos
        x_pred = pd.DataFrame([[estado_actual[p] for p in predictores]], columns=predictores)
        
        # 1. Predecir Inflación del mes corriente
        ipc_pred = model_ipc.predict(x_pred)[0]
        
        # 2. Determinar Dólar del mes corriente según la regla seleccionada
        if modo == "Pass-Through Puro (Dólar Fijo)":
            dolar_pred = parametro_cambiario  # Variación constante definida por el usuario
        else:
            # Modo Feedback Endógeno (Sujeto a un Shock Inicial opcional en el mes 1)
            if m == 0 and parametro_cambiario > 0:
                dolar_pred = parametro_cambiario
            else:
                dolar_pred = model_dolar.predict(x_pred)[0]
                
        proyecciones_ipc.append(ipc_pred)
        proyecciones_dolar.append(dolar_pred)
        
        # 3. Desplazar la ventana temporal de rezagos (Shifting)
        for i in range(lags, 1, -1):
            estado_actual[f'DOLAR_LAG_{i}'] = estado_actual[f'DOLAR_LAG_{i-1}']
            estado_actual[f'IPC_LAG_{i}'] = estado_actual[f'IPC_LAG_{i-1}']
            
        # El Lag 1 absorbe los resultados que acabamos de simular
        estado_actual['DOLAR_LAG_1'] = dolar_pred
        estado_actual['IPC_LAG_1'] = ipc_pred
        
    return proyecciones_ipc, proyecciones_dolar

# =========================================================
# INTERFAZ DE STREAMLIT
# =========================================================
st.subheader(":material/chart_data: IA Multivariada: Simulador Simétrico Tipo Cambio vs IPC")

st.info("""
**Intuición del Enfoque Macroeconómico Bilateral**
Los modelos tradicionales asumen que el dólar es una variable exógena (fija). Esta versión avanzada rompe esa limitación entrenando dos redes de árboles de decisión en paralelo:
* **Modelo A (Traslado):** $IPC_t = f(\\text{Lags Dólar}, \\text{Lags IPC})$
* **Modelo B (Inercia Cambiaria):** $\\text{Dólar}_t = f(\\text{Lags Dólar}, \\text{Lags IPC})$

De esta forma, si la inflación sube hoy, el modelo proyectará de forma autónoma una aceleración del dólar en los meses subsiguientes basados en los patrones de comportamiento históricos registrados en tu dataset.
""")

filepath = os.path.join(DATA_PATH, FILE_NAME)

if os.path.exists(filepath):
    data = load_and_prepare_data(filepath)
    
    if data is not None and not data.empty:
        # =================================================
        # PANEL DE CONTROL (SIDEBAR)
        # =================================================
        with st.sidebar:
            st.subheader(":material/settings: Configuración Estructural")
            lags_seleccionados = st.slider("Meses de memoria (Rezagos)", 1, 6, 4)
            meses_a_proyectar = st.slider("Horizonte de proyección (Meses)", 3, 12, 6)
            
            st.markdown("---")
            st.subheader(":material/select_check_box: Selección de Dinámica")
            modo_seleccionado = st.radio(
                "Mecánica de Simulación:",
                ["Pass-Through Puro (Dólar Fijo)", "Bucle de Retroalimentación IA"]
            )
            
            st.markdown("---")
            st.subheader(":material/animated_images: Escenarios a Comparar")
            
            if modo_seleccionado == "Pass-Through Puro (Dólar Fijo)":
                st.caption("Variación mensual fija simulada para el Dólar:")
                val_esc1 = st.number_input("Escenario Estable (%)", value=2.0, step=0.5)
                val_esc2 = st.number_input("Escenario Moderado (%)", value=6.0, step=0.5)
                val_esc3 = st.number_input("Shock Cambiario (%)", value=15.0, step=0.5)
                labels = [f"Dólar Fijo: {val_esc1}%", f"Dólar Fijo: {val_esc2}%", f"Dólar Fijo: {val_esc3}%"]
            else:
                st.caption("Magnitud del Shock Cambiario inicial (Solo ocurre en el Mes 1, luego libre):")
                val_esc1 = st.number_input("Sin Shock (Inercial) (%)", value=0.0, step=1.0)
                val_esc2 = st.number_input("Shock Inicial Moderado (%)", value=10.0, step=1.0)
                val_esc3 = st.number_input("Shock Inicial Severo (%)", value=25.0, step=1.0)
                labels = [f"Inercial pura", f"Deval. Inicial +{val_esc2}%", f"Deval. Inicial +{val_esc3}%"]

        # =================================================
        # ENTRENAMIENTO DE MODELOS EN PARALELO
        # =================================================
        df_ml, predictores = build_lagged_features(data, max_lags=lags_seleccionados)
        X = df_ml[predictores]
        y_ipc = df_ml['PCT_IPC']
        y_dolar = df_ml['PCT_DOLAR']

        # Entrenar estimador para IPC
        model_rf_ipc = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model_rf_ipc.fit(X, y_ipc)

        # Entrenar estimador para DÓLAR
        model_rf_dolar = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model_rf_dolar.fit(X, y_dolar)

        # =================================================
        # CÁLCULO DE PROYECCIONES TRIPLE ESCENARIO
        # =================================================
        p_ipc_1, p_dol_1 = ejecutar_simulacion(model_rf_ipc, model_rf_dolar, df_ml, predictores, lags_seleccionados, meses_a_proyectar, modo_seleccionado, val_esc1)
        p_ipc_2, p_dol_2 = ejecutar_simulacion(model_rf_ipc, model_rf_dolar, df_ml, predictores, lags_seleccionados, meses_a_proyectar, modo_seleccionado, val_esc2)
        p_ipc_3, p_dol_3 = ejecutar_simulacion(model_rf_ipc, model_rf_dolar, df_ml, predictores, lags_seleccionados, meses_a_proyectar, modo_seleccionado, val_esc3)

        # Vector de tiempo futuro
        ultima_fecha = df_ml.index[-1]
        fechas_futuras = [ultima_fecha + pd.DateOffset(months=i) for i in range(1, meses_a_proyectar + 1)]
        historico_reciente = df_ml.tail(12)

        # =================================================
        # VISUALIZACIÓN EN DASHBOARD (DOS TABLEROS ASOCIADOS)
        # =================================================
        tab1, tab2 = st.tabs([":material/analytics: Análisis Cruzado de Proyecciones", ":material/chart_data: Estructura de Causalidad (Pesos)"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(":material/finance_mode: Proyección de Inflación (IPC %)")
                fig_ipc = go.Figure()
                fig_ipc.add_trace(go.Scatter(x=historico_reciente.index, y=historico_reciente['PCT_IPC'], mode='lines', name='Histórico Real', line=dict(color='white', width=3)))
                fig_ipc.add_trace(go.Scatter(x=[ultima_fecha] + fechas_futuras, y=[historico_reciente['PCT_IPC'].iloc[-1]] + p_ipc_1, mode='lines+markers', name=labels[0], line=dict(color='#00CC96', dash='dot')))
                fig_ipc.add_trace(go.Scatter(x=[ultima_fecha] + fechas_futuras, y=[historico_reciente['PCT_IPC'].iloc[-1]] + p_ipc_2, mode='lines+markers', name=labels[1], line=dict(color='#FFA15A', dash='dot')))
                fig_ipc.add_trace(go.Scatter(x=[ultima_fecha] + fechas_futuras, y=[historico_reciente['PCT_IPC'].iloc[-1]] + p_ipc_3, mode='lines+markers', name=labels[2], line=dict(color='#EF553B', dash='dot')))
                fig_ipc.update_layout(template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_ipc, width='stretch')
                
            with col2:
                st.subheader(":material/add_chart: Proyección del Tipo de Cambio (Dólar %)")
                fig_dol = go.Figure()
                fig_dol.add_trace(go.Scatter(x=historico_reciente.index, y=historico_reciente['PCT_DOLAR'], mode='lines', name='Histórico Real', line=dict(color='white', width=3)))
                fig_dol.add_trace(go.Scatter(x=[ultima_fecha] + fechas_futuras, y=[historico_reciente['PCT_DOLAR'].iloc[-1]] + p_dol_1, mode='lines+markers', name=labels[0], line=dict(color='#00CC96', dash='dot')))
                fig_dol.add_trace(go.Scatter(x=[ultima_fecha] + fechas_futuras, y=[historico_reciente['PCT_DOLAR'].iloc[-1]] + p_dol_2, mode='lines+markers', name=labels[1], line=dict(color='#FFA15A', dash='dot')))
                fig_dol.add_trace(go.Scatter(x=[ultima_fecha] + fechas_futuras, y=[historico_reciente['PCT_DOLAR'].iloc[-1]] + p_dol_3, mode='lines+markers', name=labels[2], line=dict(color='#EF553B', dash='dot')))
                fig_dol.update_layout(template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_dol, width='stretch')
                
        with tab2:
            st.subheader(":material/analytics: Importancia de Variables en la Co-Integración Temporal")
            c1, c2 = st.columns(2)
            
            with c1:
                st.caption("¿Qué variables determinan más el **IPC**?")
                df_imp_ipc = pd.DataFrame({"Variable": predictores, "Impacto": model_rf_ipc.feature_importances_}).sort_values(by="Impacto", ascending=True)
                fig_imp_ipc = go.Figure(go.Bar(x=df_imp_ipc['Impacto'], y=df_imp_ipc['Variable'], orientation='h', marker=dict(color='#636EFA')))
                fig_imp_ipc.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_imp_ipc, width='stretch')
                
            with c2:
                st.caption("¿Qué variables determinan más la re-acción del **Dólar**?")
                df_imp_dol = pd.DataFrame({"Variable": predictores, "Impacto": model_rf_dolar.feature_importances_}).sort_values(by="Impacto", ascending=True)
                fig_imp_dol = go.Figure(go.Bar(x=df_imp_dol['Impacto'], y=df_imp_dol['Variable'], orientation='h', marker=dict(color='#00CC96')))
                fig_imp_dol.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_imp_dol, width='stretch')

else:
    st.error(f":material/error: No se detectó el archivo unificado en la ruta: `{filepath}`")