import streamlit as st
import base64
from pathlib import Path
import os
import streamlit.components.v1 as components

# --- Configuración página ---
st.set_page_config(
    page_title="Dashboard Análisis Inflacion", 
    layout="wide", 
    page_icon=":material/currency_exchange:", 
    initial_sidebar_state="collapsed"
)

# --- Estilos Globales (Ocultar elementos de Streamlit) ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stAppViewContainer"] { margin-left: 0px; }
    </style>
""", unsafe_allow_html=True)


# --- CARGA DE CSS PARA BOTONES HOLOGRAMA ---
try:
    # Se asume que el CSS está en static/boton.css según tu solicitud anterior
    boton_css_raw = Path("static/boton.css").read_text(encoding="utf-8")
    hologram_css = f"<style>{boton_css_raw}</style>"
except:
    # Fallback al CSS embebido si el archivo no existe
    hologram_css = """
    <style>
    div[data-testid="stButton"] > button {
        width: 100% !important; /* Fuerza el ancho total */
        position: relative;
        padding: 1.2rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #fff !important;
        background: rgba(0, 255, 255, 0.1) !important;
        border: 2px solid rgba(0, 255, 255, 0.5) !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3) !important;
        backdrop-filter: blur(5px) !important;
        text-transform: uppercase;
        transition: all 0.4s ease !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: rgba(0, 255, 255, 0.2) !important;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.5) !important;
        border-color: rgba(0, 255, 255, 0.8) !important;
    }
    </style>
    """



col_izq, col_central, col_der = st.columns([1, 10, 1])

with col_central:
    # 1. Subheader centrado usando el parámetro nativo
    st.header("Dashboard Análisis y Proyecciones de Dólar", anchor=False, text_alignment="center")

    # 2. CSS para centrar el texto dentro de los componentes st.info (o alertas)
    st.markdown("""
        <style>
        .stAlert > div {
            text-align: center;
            display: flex;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # 3. Cuadro de información centrado
    #st.info("aplicando modelos GRU-LSTM-TCN-Transformer")

    # (Aquí continuarían tus animaciones y botones...)
    v1, v2 = st.columns(2)

# --- Carga de Animaciones ---
def load_html(file_name):
    return Path(file_name).read_text(encoding="utf-8")

try:
    tunnel_html = load_html("static/matrix-terminal-3.html") #
    crt_html = load_html("static/crt-boot-sequence.html") #
except:
    tunnel_html = crt_html = ""

# --- Función para cargar HTML como Data URL para st.iframe ---
def get_html_data_url(file_path):
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        b64 = base64.b64encode(content.encode()).decode()
        return f"data:text/html;base64,{b64}"
    except:
        return ""

#-------------------------------------------------------
# inyeccion de las tarjetas 
def load_component():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "assets")
    
    try:
        with open(os.path.join(base_path, "card.html"), "r", encoding="utf-8") as f:
            html_content = f.read()
        with open(os.path.join(base_path, "card.css"), "r", encoding="utf-8") as f:
            css_content = f.read()
        with open(os.path.join(base_path, "card.js"), "r", encoding="utf-8") as f:
            js_content = f.read()

        full_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <style>{css_content}</style>
        </head>
        <body>
            {html_content}
            <script>{js_content}</script>
        </body>
        </html>
        """
        
        b64_html = base64.b64encode(full_code.encode("utf-8")).decode("utf-8")
        return f"data:text/html;base64,{b64_html}"

    except FileNotFoundError as e:
        st.error(f"No se encontró el archivo: {e.filename}")
        return None


# Obtener las URLs de datos
tunnel_url = get_html_data_url("static/matrix-terminal.html")
crt_url = get_html_data_url("static/crt-boot-sequence.html")
data_url_cards = load_component()

# --- RENDERIZADO ---
col_izq, col_central, col_der = st.columns([1, 10, 1]) #

with col_central:
    # 1. Animaciones en ventanas paralelas
    v1, v2 = st.columns(2)
    with v1:        
        #components.html(tunnel_html, height=400, scrolling=False)
        if tunnel_url:
            st.iframe(tunnel_url, height=480)
    with v2:
        #components.html(crt_html, height=400, scrolling=False)
        if crt_url:
            st.iframe(crt_url, height=480)

    st.write("") 

    if data_url_cards:
        st.iframe(src=data_url_cards, height=500)

    # Espaciado
    st.write("") 
    
    # Inyectar CSS global desde archivo
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


    
    # 2. Inyectamos el CSS para que afecte a los botones de abajo
    st.markdown(hologram_css, unsafe_allow_html=True)
    
    # 3. Contenedor de botones ajustado
    # Para que los botones coincidan con el ancho de las animaciones,
    # usamos las mismas columnas (3) sin márgenes internos extra.
    with st.container(border=True):
        st.subheader("Opciones de modelos de Análisis", anchor=False, text_alignment="center")
        
        # Usamos width='stretch' para que ocupen todo el espacio de su columna
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button(":material/threat_intelligence: Modelo#1", key="acceso", width='stretch'):
                st.switch_page("pages/1-modeloSarimaHoltWintersLstm.py")
        with b2:
            if st.button(":material/threat_intelligence: Modelo#2", key="acceso1", width='stretch'):
                st.switch_page("pages/2-modeloGRU-TCN-BILstm.py")
        with b3:
            if st.button(":material/threat_intelligence: Modelo#3", key="acceso2", width='stretch'):
                st.switch_page("pages/3-modeloRandomForest.py")   

        b4, b5, b6 = st.columns(3)
        with b4:
            if st.button(":material/threat_intelligence: Modelo#4", key="acceso3", width='stretch'):
                st.switch_page("pages/4-modeloVarGarch.py")
        with b5:
            if st.button(":material/threat_intelligence: Modelo#5", key="acceso4", width='stretch'):
                st.switch_page("pages/5-modeloPass-Through.py")
        with b6:
            if st.button(":material/threat_intelligence: Modelo#6", key="acceso5", width='stretch'):
                st.switch_page("pages/6-modeloTermodinámica.py")

    st.markdown("---")            