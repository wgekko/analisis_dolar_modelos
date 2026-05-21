import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Econofísica: Consola Dólar ", layout="wide")

st.subheader(":material/chip_extraction: Econo-física Avanzada: Análisis del Dólar")
st.markdown("""
Esta consola está automatizada para procesar el histórico de cotizaciones del **Dólar Blue** en formato Excel, 
traduciendo spreads, brechas e impulsos en vectores de fuerzas cinemáticas y termodinámicas.
""")

# --- BARRA LATERAL (CONFIGURACIÓN DE PARÁMETROS) ---
st.sidebar.subheader(":materal/settings: Parámetros del Motor")
dias_historia = st.sidebar.slider("Días de Análisis Histórico", 30, 1500, 365)
ventana_e = st.sidebar.slider("Sensibilidad del Caos (Ventana Móvil)", 10, 60, 20)

# --- CONFIGURACIÓN DE RUTA AUTOMÁTICA ---
# Apunta directamente a la carpeta 'data' y al archivo 'Dolar.xlsx'
RUTA_EXCEL = os.path.join("data", "Dolar.xlsx")

@st.cache_data(ttl=600)  # Caché de 10 minutos para optimizar performance
def cargar_y_adaptar_excel(ruta, dias):
    if not os.path.exists(ruta):
        st.error(f":material/error: **Error de Sistema:** No se encontró el archivo en la ruta esperada: `{ruta}`. Por favor, verifica que la carpeta 'data' y el archivo 'Dolar.xlsx' existan.")
        return None
    
    try:
        # Lectura nativa de Excel (.xlsx)
        df = pd.read_excel(ruta)
        
        # Limpieza y normalización de nombres de columnas
        df.columns = df.columns.str.strip().str.upper()
        
        # Filtrado selectivo de columnas financieras para limpiar basura del Excel
        columnas_interes = [c for c in ['FECHA', 'COMPRA', 'VENTA', 'PROMEDIO', 'BRECHA'] if c in df.columns]
        df = df[columnas_interes]
        
        # Conversión y ordenamiento temporal
        # df['FECHA'] = pd.to_datetime(df['FECHA'])
        # df = df.sort_values('FECHA').set_index('FECHA') 
        df['FECHA'] = pd.to_datetime(df['FECHA'])        
        # --- SOLUCIÓN: Limpiar fechas duplicadas antes de setear el índice ---
        df = df.drop_duplicates(subset=['FECHA'], keep='last')        
        df = df.sort_values('FECHA').set_index('FECHA')

        
        # Asegurar tipos flotantes para evitar fricciones en el cálculo numérico
        for col in ['COMPRA', 'VENTA', 'PROMEDIO', 'BRECHA']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Descartar filas vacías críticas
        df = df.dropna(subset=['PROMEDIO'])
        
        return df.iloc[-dias:]
    except Exception as e:
        st.error(f":material/error: **Error al procesar el archivo Excel:** {e}")
        return None

# --- EJECUCIÓN AUTOMÁTICA DEL MOTOR ---
df_dolar = cargar_y_adaptar_excel(RUTA_EXCEL, dias_historia)

if df_dolar is not None:
    # Aplanamiento absoluto (1D) de vectores para blindar Plotly de dataframes multidimensionales
    fechas = df_dolar.index
    precio_promedio = df_dolar['PROMEDIO'].values.flatten()
    precio_compra = df_dolar['COMPRA'].values.flatten()
    precio_venta = df_dolar['VENTA'].values.flatten()
    brecha = df_dolar['BRECHA'].values.flatten() if 'BRECHA' in df_dolar.columns else (precio_venta - precio_compra)

    # ---------------------------------------------------------
    # 1. CINEMÁTICA: CAMPO DE FUERZA Y VELOCIDAD
    # ---------------------------------------------------------
    st.subheader(":material/fast_forward: 1. Cinemática Cambiaria: Velocidad y Canal de Fluctuación")
    
    serie_promedio = pd.Series(precio_promedio, index=fechas)
    velocidad = serie_promedio.diff().dropna()
    aceleracion = velocidad.diff().dropna()

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_kin = go.Figure()
        # Canal de dispersión entre Compra y Venta
        fig_kin.add_trace(go.Scatter(x=fechas, y=precio_venta, name="Punta Vendedora", line=dict(color='#00CC96', width=1.5)))
        fig_kin.add_trace(go.Scatter(x=fechas, y=precio_compra, name="Punta Compradora", line=dict(color='#FF4136', width=1.5), fill='tonexty', fillcolor='rgba(0, 204, 150, 0.05)'))
        # Representación de la velocidad del flujo (Pesos ganados/perdidos por día)
        fig_kin.add_trace(go.Bar(x=velocidad.index, y=velocidad, name="Velocidad (Δ$/Día)", opacity=0.2, marker_color='#AB63FA'))
        
        fig_kin.update_layout(title="Bandas de Cotización y Vector de Velocidad Diaria", hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig_kin, width='stretch')

    with col2:
        st.subheader(":material/speed_2: Métricas de Inercia")
        ultima_v = float(velocidad.iloc[-1])
        ultima_a = float(aceleracion.iloc[-1])
        
        st.metric("Velocidad Actual", f"${ultima_v:.2f} / día", delta=f"${ultima_v - velocidad.iloc[-2]:.2f}")
        st.metric("Aceleración (Impulso)", f"{ultima_a:.4f}", delta=f"{ultima_a - aceleracion.iloc[-2]:.4f}")
        
        st.info("""
        * **Interpretación Cinematográfica:** El área sombreada mapea la dispersión de puntas. 
        * Si la **velocidad** es positiva pero la **aceleración** cruza a terreno negativo, la partícula (precio) está experimentando una desaceleración. Esto suele indicar la cercanía de un techo o resistencia del mercado.
        """)

    # ---------------------------------------------------------
    # 2. FRICCIÓN Y TENSIÓN: ANÁLISIS DE LA BRECHA
    # ---------------------------------------------------------
    # st.header("2. Dinámica de Fluidos: Tensión y Fricción (Brecha)")
    
    # col_b1, col_b2 = st.columns([1, 2])
    # with col_b1:
    #     ultima_brecha = float(brecha[-1])
    #     brecha_media = float(brecha.mean())
    #     st.metric("Brecha / Spread Actual", f"{ultima_brecha:.2f}%", delta=f"{(ultima_brecha - brecha[-2]):.2f}%")
    #     st.write(f"**Brecha Promedio del Período:** {brecha_media:.2f}%")
    #     st.help("En ambientes hiperinflacionarios o de alta volatilidad, la brecha actúa como fricción hidrodinámica. Ensanchamientos rápidos de la brecha suelen generar inestabilidad y posterior aumento del caos (entropía).")

    # with col_b2:
    #     fig_brecha = go.Figure()
    #     fig_brecha.add_trace(go.Scatter(x=fechas, y=brecha, name="Nivel de Brecha", line=dict(color='#FFDC00', width=2)))
    #     fig_brecha.add_trace(go.Scatter(x=fechas, y=[brecha_media]*len(fechas), name="Fricción Media", line=dict(color='rgba(255,255,255,0.3)', dash='dash')))
    #     fig_brecha.update_layout(title="Evolución de la Viscosidad (Brecha Cambiaria)", template="plotly_dark", yaxis_title="Porcentaje (%)")
    #     st.plotly_chart(fig_brecha, width='stretch')

# ---------------------------------------------------------
    # 2. FRICCIÓN Y TENSIÓN: ANÁLISIS DE LA BRECHA (VERSIÓN COMPACTA)
    # ---------------------------------------------------------
    st.header(":material/speed_4: 2. Dinámica de Fluidos: Tensión y Fricción (Spread Compra-Venta)")
    
    col_b1, col_b2 = st.columns([1.2, 1.8]) # Ajustamos levemente la proporción para dar más aire al texto
    
    with col_b1:
        ultima_brecha = float(brecha[-1])
        brecha_media = float(brecha.mean())
        
        # Creamos sub-columnas internas para aprovechar el espacio HORIZONTAL
        sub_col1, sub_col2 = st.columns([1, 1.3])
        
        with sub_col1:
            st.metric(
                label="Fricción Actual", 
                value=f"${ultima_brecha:.2f}", 
                delta=f"${(ultima_brecha - brecha[-2]):.2f}"
            )
            st.caption(f"Media del período: **${brecha_media:.2f}**")
        
        with sub_col2:
            # Un pequeño espacio para alinear verticalmente el texto con el número de la métrica
            st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)
            
            # Diagnóstico con HTML/CSS Micro-compacto (Ahorra un 70% de espacio vertical)
            if ultima_brecha > brecha_media * 1.2:
                st.markdown(f"""
                <div style="background-color: rgba(255, 75, 75, 0.1); border-left: 4px solid #FF4B4B; padding: 10px; border-radius: 4px; font-size: 13px; line-height: 1.4;">
                    <span style="color: #FF4B4B; font-weight: bold;">⚠️ FRICCIÓN ALTA</span><br>
                    El spread supera la media. Las cuevas se están cubriendo; alta incertidumbre en el fluido cambiario.
                </div>
                """, unsafe_allow_html=True)
                
            elif ultima_brecha < brecha_media * 0.8:
                st.markdown(f"""
                <div style="background-color: rgba(0, 204, 150, 0.1); border-left: 4px solid #00CC96; padding: 10px; border-radius: 4px; font-size: 13px; line-height: 1.4;">
                    <span style="color: #00CC96; font-weight: bold;">✅ FRICCIÓN BAJA</span><br>
                    Spread por debajo de la media. Mercado normalizado y flujo en calma.
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div style="background-color: rgba(31, 119, 180, 0.1); border-left: 4px solid #1F77B4; padding: 10px; border-radius: 4px; font-size: 13px; line-height: 1.4;">
                    <span style="color: #1F77B4; font-weight: bold;">⚖️ FRICCIÓN NORMAL</span><br>
                    El mercado opera con total normalidad dentro de sus rangos de equilibrio promedio habituales.
                </div>
                """, unsafe_allow_html=True)

    with col_b2:
        fig_brecha = go.Figure()
        fig_brecha.add_trace(go.Scatter(x=fechas, y=brecha, name="Brecha en Pesos ($)", line=dict(color='#FFDC00', width=2)))
        fig_brecha.add_trace(go.Scatter(x=fechas, y=[brecha_media]*len(fechas), name="Fricción Media", line=dict(color='rgba(255,255,255,0.3)', dash='dash')))
        
        # Reducimos los márgenes del gráfico para que use mejor el espacio vertical
        fig_brecha.update_layout(
            title="Evolución de la Viscosidad (Spread Absoluto)", 
            template="plotly_dark", 
            yaxis_title="Pesos ($)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=220 # Forzamos una altura compacta para alinearse con las métricas
        )
        st.plotly_chart(fig_brecha, width='stretch')



    # ---------------------------------------------------------
    # 3. PROYECCIÓN FUTURA (DIFUSIÓN ESTOCÁSTICA)
    # ---------------------------------------------------------
    st.subheader(":material/brightness_5: 3. Proyección de Escenarios: Difusión Probable")
    dias_proy = st.slider("Días de proyección hacia el futuro", 5, 90, 30)
    
    log_returns = np.log(serie_promedio / serie_promedio.shift(1)).dropna()
    u, var, stdev = float(log_returns.mean()), float(log_returns.var()), float(log_returns.std())
    drift = u - (0.5 * var)
    
    sims = 50
    last_price = float(precio_promedio[-1])
    
    yields = np.exp(drift + stdev * np.random.standard_normal((dias_proy, sims)))
    predictions = np.zeros_like(yields)
    predictions[0] = last_price
    for t in range(1, dias_proy):
        predictions[t] = predictions[t-1] * yields[t]
        
    fig_proy = go.Figure()
    for i in range(sims):
        fig_proy.add_trace(go.Scatter(y=predictions[:, i], mode='lines', line=dict(width=1), opacity=0.1, showlegend=False))
    
    # Delimitadores de control estadístico (Percentiles)
    fig_proy.add_trace(go.Scatter(y=np.percentile(predictions, 90, axis=1), name="Escenario Techo Estresado (P90)", line=dict(color='#FF4136', width=2, dash='dash')))
    fig_proy.add_trace(go.Scatter(y=np.percentile(predictions, 50, axis=1), name="Trayectoria Central (Mediana)", line=dict(color='white', width=2)))
    fig_proy.add_trace(go.Scatter(y=np.percentile(predictions, 10, axis=1), name="Escenario Piso Optimista (P10)", line=dict(color='#00CC96', width=2, dash='dash')))
    
    fig_proy.update_layout(title="Simulación Estocástica de Escenarios de Difusión", xaxis_title="Días a Futuro", yaxis_title="Precio Proyectado ($)", template="plotly_dark")
    st.plotly_chart(fig_proy, width='stretch')

    # ---------------------------------------------------------
    # 4. TERMODINÁMICA: RELACIÓN ACELERACIÓN VS ENTROPÍA
    # ---------------------------------------------------------
    st.subheader(":material/satellite_alt: 4. Termodinámica Cambiaria: Sincronización de Fuerza y Caos")

    def calc_h(s):
        c, _ = np.histogram(s, bins=10, density=True)
        p = c / (c.sum() + 1e-10)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    rolling_entropy = log_returns.rolling(window=ventana_e).apply(calc_h).dropna()
    idx_comun = rolling_entropy.index.intersection(aceleracion.index)
    
    # Sincronización estricta 1D
    df_corr = pd.DataFrame({
        'Aceleracion': aceleracion.loc[idx_comun].values.flatten(),
        'Entropia': rolling_entropy.loc[idx_comun].values.flatten()
    }, index=idx_comun)

    def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-10)
    df_corr['Acel_Norm'] = norm(df_corr['Aceleracion'].abs())
    df_corr['Entropia_Norm'] = norm(df_corr['Entropia'])

    with st.expander(":material/podcasts: ¿Cómo anticipar tendencias usando este Radar?"):
        st.write("""
        El análisis termodinámico te permite separar las **tendencias reales** del simple **ruido**:
        * **Inicio de Corrida / Impulso Firme:** Notarás que el *Nivel de Caos* (Entropía) cae drásticamente mientras la *Fuerza* (Aceleración) escala. Esto describe un sistema donde los operadores actúan sincronizados (consenso de mercado); el desorden baja y se arma un vector de fuerza único.
        * **Mercado Lateral / Calma Artificial:** La entropía se estabiliza en máximos y la aceleración se desinfla. El precio fluctúa por mero ruido transaccional sin una dirección dominante.
        """)

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Acel_Norm'], name="Intensidad de Fuerza (Aceleración)", line=dict(color='cyan')))
    fig_corr.add_trace(go.Scatter(x=df_corr.index, y=df_corr['Entropia_Norm'], name="Nivel de Caos (Entropía)", line=dict(color='magenta', dash='dot')))
    fig_corr.update_layout(title="Monitoreo de Energía y Desorden Temporal", template="plotly_dark")
    st.plotly_chart(fig_corr, width='stretch')

    # MAPA DEL ESPACIO DE FASE
    st.subheader(":material/map_search: Mapa del Espacio de Fase: Estado Estructural del Sistema")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_corr['Acel_Norm'], y=df_corr['Entropia_Norm'], 
        mode='markers+lines',
        line=dict(width=0.5, color='rgba(255,255,255,0.15)'),
        marker=dict(size=10, color=np.arange(len(df_corr)), colorscale='Cividis', showscale=True, colorbar=dict(title="Línea de Tiempo"))
    ))
    fig_scatter.update_layout(xaxis_title="Fuerza del Movimiento (Aceleración)", yaxis_title="Nivel de Caos (Entropía)", template="plotly_dark")
    st.plotly_chart(fig_scatter, width='stretch')
    st.caption("Los puntos claros representan las jornadas recientes. El desplazamiento coordinado hacia la esquina inferior derecha valida un movimiento directivo de baja incertidumbre.")