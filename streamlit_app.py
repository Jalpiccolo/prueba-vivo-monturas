import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image
import os
import copy
import base64
import json

# Importación compatible: Mac Silicon (local) y Linux (nube)
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
except (ImportError, AttributeError):
    import mediapipe as mp
    mp_face_mesh_module = mp.solutions.face_mesh

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA Y ESTILOS
# ---------------------------------------------------------
st.set_page_config(page_title="A&L VISION - Probador Virtual", layout="wide", page_icon="👓")

COLOR_TEAL = "#328e7e"
COLOR_PINK = "#d07b89"
COLOR_NAVY = "#172d54"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    .titulo-principal {{
        font-size: 42px !important; font-weight: 800; color: {COLOR_TEAL}; padding-top: 20px;
    }}
    .subtitulo {{
        font-size: 26px !important; font-weight: bold; color: {COLOR_PINK};
        margin-top: 20px; border-bottom: 2px solid {COLOR_PINK}; padding-bottom: 5px;
    }}

    /* Botones principales grandes */
    div[data-testid="stHorizontalBlock"] div.stButton > button:first-child {{
        font-size: 18px !important; font-weight: bold; height: 60px; width: 100%;
        background-color: {COLOR_TEAL}; color: white; border-radius: 8px;
        border: 2px solid {COLOR_TEAL}; transition: 0.3s;
    }}
    div[data-testid="stHorizontalBlock"] div.stButton > button:first-child:hover {{
        background-color: {COLOR_NAVY}; border-color: {COLOR_NAVY}; color: white;
    }}

    /* Ocultar menú de Streamlit */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Responsive: celulares */
    @media (max-width: 768px) {{
        .titulo-principal {{ font-size: 28px !important; }}
        .subtitulo {{ font-size: 20px !important; }}
        div[data-testid="stHorizontalBlock"] div.stButton > button:first-child {{
            font-size: 15px !important; height: 50px;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
defaults = {
    'rostro_seleccionado': "Ovalado",
    'metodo_elegido': None,
    'foto_capturada': None,       # foto de st.camera_input (modo cámara)
    'imagen_subida': None,        # foto subida (modo archivo)
    'gafa_seleccionada': "gafas1.png",
    'comparador': [],
    'ver_todas': False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------
# CATÁLOGO Y REGLAS DE OPTOMETRÍA
# ---------------------------------------------------------
CATALOGO = [
    {"id": 1,  "nombre": "Elegancia Cuadrada",     "archivo": "gafas1.png",  "forma": "Cuadrada"},
    {"id": 2,  "nombre": "Estilo Urbano",           "archivo": "gafas2.png",  "forma": "Cuadrada"},
    {"id": 3,  "nombre": "Cuadrado Grueso",         "archivo": "gafas3.png",  "forma": "Cuadrada"},
    {"id": 4,  "nombre": "Redonda Vintage",         "archivo": "gafas4.png",  "forma": "Redonda"},
    {"id": 5,  "nombre": "Redonda Delgada",         "archivo": "gafas5.png",  "forma": "Redonda"},
    {"id": 6,  "nombre": "Aviador Clásica",         "archivo": "gafas6.png",  "forma": "Aviador"},
    {"id": 7,  "nombre": "Aviador Moderna",         "archivo": "gafas7.png",  "forma": "Aviador"},
    {"id": 8,  "nombre": "Cat Eye Elegante",        "archivo": "gafas8.png",  "forma": "Ojo de Gato"},
    {"id": 9,  "nombre": "Cat Eye Casual",          "archivo": "gafas9.png",  "forma": "Ojo de Gato"},
    {"id": 10, "nombre": "Rectangular Intelectual", "archivo": "gafas10.png", "forma": "Rectangular"},
    {"id": 11, "nombre": "Estilo Moderno",          "archivo": "gafas11.png", "forma": "Cuadrada"},
]

REGLAS_OPTOMETRIA = {
    "Ovalado":             ["Cuadrada", "Redonda", "Aviador", "Ojo de Gato", "Rectangular"],
    "Redondo":             ["Cuadrada", "Rectangular"],
    "Cuadrado":            ["Redonda", "Ojo de Gato", "Aviador"],
    "En forma de corazón": ["Aviador", "Ojo de Gato", "Redonda"],
}

# ---------------------------------------------------------
# FUNCIONES DE VISIÓN POR COMPUTADORA
# ---------------------------------------------------------
@st.cache_data
def load_glasses(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = overlay.shape[:2]
    x, y = int(x), int(y)
    y1, y2 = y - fg_h // 2, y - fg_h // 2 + fg_h
    x1, x2 = x - fg_w // 2, x - fg_w // 2 + fg_w
    bg_y1, bg_y2 = max(0, y1), min(bg_h, y2)
    bg_x1, bg_x2 = max(0, x1), min(bg_w, x2)
    if bg_y1 >= bg_y2 or bg_x1 >= bg_x2:
        return background
    fg_y1 = bg_y1 - y1;  fg_y2 = fg_y1 + (bg_y2 - bg_y1)
    fg_x1 = bg_x1 - x1;  fg_x2 = fg_x1 + (bg_x2 - bg_x1)
    overlay_crop = overlay[fg_y1:fg_y2, fg_x1:fg_x2]
    alpha = overlay_crop[:, :, 3:4] / 255.0
    background[bg_y1:bg_y2, bg_x1:bg_x2] = (
        alpha * overlay_crop[:, :, :3] +
        (1 - alpha) * background[bg_y1:bg_y2, bg_x1:bg_x2]
    ).astype(np.uint8)
    return background

def procesar_imagen(image_bgr, glasses_bgra):
    """Detecta el rostro, escala y posiciona las gafas correctamente."""
    with mp_face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return image_bgr, False

    landmarks = results.multi_face_landmarks[0].landmark
    ih, iw = image_bgr.shape[:2]

    p1, p2 = landmarks[468], landmarks[473]
    pt_izq, pt_der = (p1, p2) if p1.x < p2.x else (p2, p1)
    xi, yi = int(pt_izq.x * iw), int(pt_izq.y * ih)
    xd, yd = int(pt_der.x * iw), int(pt_der.y * ih)

    dist = np.hypot(xd - xi, yd - yi)
    ancho = int(dist * 2.3)
    if ancho < 1:
        return image_bgr, False

    hg, wg = glasses_bgra.shape[:2]
    alto = int(ancho * hg / wg)
    gafas_r = cv2.resize(glasses_bgra, (ancho, alto))

    ang = np.degrees(np.arctan2(yd - yi, xd - xi))
    M = cv2.getRotationMatrix2D((ancho // 2, alto // 2), -ang, 1.0)
    gafas_rot = cv2.warpAffine(gafas_r, M, (ancho, alto),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    cx, cy = (xi + xd) / 2, (yi + yd) / 2
    return overlay_transparent(image_bgr.copy(), gafas_rot, cx, cy), True

# ---------------------------------------------------------
# HELPER: SELECTOR DE MONTURAS
# ---------------------------------------------------------
def mostrar_selector_gafas(catalogo, key_prefix):
    elementos_por_fila = 5
    for i in range(0, len(catalogo), elementos_por_fila):
        cols = st.columns(elementos_por_fila)
        for j in range(elementos_por_fila):
            if i + j < len(catalogo):
                item = catalogo[i + j]
                with cols[j]:
                    borde = (f"3px solid {COLOR_PINK}"
                             if item["archivo"] == st.session_state.gafa_seleccionada
                             else "1px solid #ccc")
                    st.markdown(
                        f'<div style="border:{borde};padding:5px;border-radius:10px;">',
                        unsafe_allow_html=True)
                    st.image(item["archivo"], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    if st.button(item["nombre"], key=f"{key_prefix}_{item['id']}",
                                 use_container_width=True):
                        st.session_state.gafa_seleccionada = item["archivo"]
                        st.rerun()

# ---------------------------------------------------------
# HELPER: PROCESAR Y MOSTRAR RESULTADO
# ---------------------------------------------------------
def mostrar_resultado(imagen_fuente, catalogo_filtrado, key_prefix):
    """Procesa una imagen y muestra el resultado con las gafas seleccionadas."""
    try:
        img_pil = Image.open(imagen_fuente).convert("RGB")
        img_arr = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

        gafas_bgra = load_glasses(st.session_state.gafa_seleccionada)
        if gafas_bgra is None:
            st.error(f"No se pudo cargar: {st.session_state.gafa_seleccionada}")
            return

        res_bgr, detectado = procesar_imagen(img_bgr, gafas_bgra)

        if detectado:
            res_rgb = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
            nombre_gafa = next(
                (g["nombre"] for g in CATALOGO
                 if g["archivo"] == st.session_state.gafa_seleccionada),
                "Montura"
            )

            # Resultado centrado y amplio
            _, c_cen, _ = st.columns([1, 3, 1])
            with c_cen:
                st.image(res_rgb, use_container_width=True,
                         caption=f"✨ Montura: {nombre_gafa}")
                if st.button("📸 Guardar en comparador", key=f"guardar_{key_prefix}",
                             use_container_width=True):
                    st.session_state.comparador.append({
                        "img": copy.deepcopy(res_rgb),
                        "nombre": nombre_gafa,
                    })
                    st.toast("✅ ¡Foto guardada en el comparador!")

            # Selector de monturas debajo del resultado
            st.markdown("#### 👓 Toca otra montura para cambiarla al instante:")
            mostrar_selector_gafas(catalogo_filtrado, key_prefix)
        else:
            st.error(
                "⚠️ No detectamos tu rostro con claridad.\n\n"
                "**Consejos:**\n"
                "- Asegúrate de tener buena iluminación (frente a una ventana).\n"
                "- Mira directamente a la cámara, de frente.\n"
                "- Acércate un poco más."
            )
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

# ---------------------------------------------------------
# GENERADOR DEL COMPONENTE DE VIDEO EN VIVO (JavaScript)
# ---------------------------------------------------------
def generar_html_vivo(catalogo):
    """Lee la plantilla HTML y le inyecta las imágenes de gafas en base64."""
    glasses_data = []
    for item in catalogo:
        img = cv2.imread(item["archivo"], cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Redimensionar para rendimiento web
            h, w = img.shape[:2]
            if w > 400:
                scale = 400 / w
                img = cv2.resize(img, (400, int(h * scale)))
            _, buf = cv2.imencode('.png', img)
            b64 = base64.b64encode(buf).decode('utf-8')
            glasses_data.append({"id": item["id"], "nombre": item["nombre"], "b64": b64})

    # Leer plantilla HTML
    template_path = os.path.join(os.path.dirname(__file__), "static", "live_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Inyectar datos de gafas antes del primer <script>
    inject = f"<script>window.GLASSES_DATA = {json.dumps(glasses_data)};</script>"
    html = html.replace("</head>", f"{inject}\n</head>")
    return html

# ==========================================================
# INTERFAZ PRINCIPAL
# ==========================================================

# --- CABECERA ---
col_logo, col_titulo = st.columns([1, 4])
with col_logo:
    logo_file = "ALVision-Oportof Logo.png"
    if os.path.exists(logo_file):
        st.image(logo_file, use_container_width=True)
    else:
        st.info("📌 Logo")
with col_titulo:
    st.markdown('<div class="titulo-principal">Asesor Virtual A&L Vision</div>',
                unsafe_allow_html=True)

# ----------------------------------------------------------
# PASO 1 – FORMA DEL ROSTRO
# ----------------------------------------------------------
st.markdown('<p class="subtitulo">Paso 1: ¿Qué forma tiene tu rostro?</p>',
            unsafe_allow_html=True)
st.write("Selecciona tu tipo de rostro para filtrar las monturas que te favorecen "
         "según las normas de optometría estética:")

forma_sel = st.radio(
    "Forma del rostro",
    ["Ovalado", "Redondo", "Cuadrado", "En forma de corazón"],
    horizontal=True,
    label_visibility="collapsed",
    index=["Ovalado", "Redondo", "Cuadrado", "En forma de corazón"]
          .index(st.session_state.rostro_seleccionado),
)
if forma_sel != st.session_state.rostro_seleccionado:
    st.session_state.rostro_seleccionado = forma_sel
    st.rerun()

recomendaciones = REGLAS_OPTOMETRIA[st.session_state.rostro_seleccionado]
if st.session_state.ver_todas:
    catalogo_filtrado = [g for g in CATALOGO if os.path.exists(g["archivo"])]
else:
    catalogo_filtrado = [g for g in CATALOGO
                         if g["forma"] in recomendaciones and os.path.exists(g["archivo"])]
if not catalogo_filtrado:
    catalogo_filtrado = [g for g in CATALOGO if os.path.exists(g["archivo"])]

col_info, col_btn = st.columns([3, 1])
with col_info:
    if st.session_state.ver_todas:
        st.info(f"👓 Mostrando **todas las monturas** ({len(catalogo_filtrado)}).")
    else:
        st.info(f"✨ Rostro **{st.session_state.rostro_seleccionado.lower()}** → "
                f"recomendadas: **{', '.join(recomendaciones)}** "
                f"({len(catalogo_filtrado)} monturas)")
with col_btn:
    label_btn = "🔍 Solo recomendadas" if st.session_state.ver_todas else "👓 Ver todas"
    if st.button(label_btn, use_container_width=True):
        st.session_state.ver_todas = not st.session_state.ver_todas
        st.rerun()

# Garantizar gafa válida
archivos_disp = [g["archivo"] for g in catalogo_filtrado]
if st.session_state.gafa_seleccionada not in archivos_disp and archivos_disp:
    st.session_state.gafa_seleccionada = archivos_disp[0]

# ----------------------------------------------------------
# PASO 2 – MÉTODO DE CAPTURA
# ----------------------------------------------------------
st.markdown('<p class="subtitulo">Paso 2: Elige cómo quieres probar las monturas</p>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🎥 Video en Vivo", use_container_width=True):
        st.session_state.metodo_elegido = "vivo"
with c2:
    if st.button("📸 Foto con Cámara", use_container_width=True):
        st.session_state.metodo_elegido = "camara"
        st.session_state.foto_capturada = None
with c3:
    if st.button("🖼️ Subir una foto", use_container_width=True):
        st.session_state.metodo_elegido = "foto"
        st.session_state.imagen_subida = None

# ==========================================================
# MODO VIDEO EN VIVO (JavaScript client-side)
# ==========================================================
if st.session_state.metodo_elegido == "vivo":
    st.markdown('<p class="subtitulo">Paso 3: Probador en Vivo 🎥</p>',
                unsafe_allow_html=True)
    st.info(
        "📋 **Instrucciones:**\n"
        "1. Permite el acceso a la cámara cuando el navegador lo solicite.\n"
        "2. Mira de frente, con buena iluminación.\n"
        "3. Toca las monturas de abajo para cambiarlas en tiempo real.\n"
        "4. Presiona 📸 para descargar tu foto favorita."
    )
    html_vivo = generar_html_vivo(catalogo_filtrado)
    components.html(html_vivo, height=650, scrolling=True)

# ==========================================================
# MODO CÁMARA (funciona en celular, tablet y PC)
# ==========================================================
elif st.session_state.metodo_elegido == "camara":
    st.markdown('<p class="subtitulo">Paso 3: Prueba las monturas 📸</p>',
                unsafe_allow_html=True)
    st.info(
        "📋 **Instrucciones:**\n"
        "1. Permite el acceso a la cámara cuando el navegador lo solicite.\n"
        "2. Mira de frente, con buena iluminación.\n"
        "3. Toma la foto presionando el botón de la cámara.\n"
        "4. **¡Cambia de montura abajo y el resultado se actualiza al instante!**"
    )

    foto = st.camera_input("📷 Toma tu foto (tócala para actualizar)")
    if foto is not None:
        st.session_state.foto_capturada = foto

    if st.session_state.foto_capturada is not None:
        mostrar_resultado(st.session_state.foto_capturada, catalogo_filtrado, "cam")

# ==========================================================
# MODO SUBIR FOTO
# ==========================================================
elif st.session_state.metodo_elegido == "foto":
    st.markdown('<p class="subtitulo">Paso 3: Sube tu foto 🖼️</p>',
                unsafe_allow_html=True)
    foto = st.file_uploader("Arrastra o selecciona tu foto",
                            type=["jpg", "jpeg", "png"])
    if foto is not None:
        st.session_state.imagen_subida = foto

    if st.session_state.imagen_subida is not None:
        mostrar_resultado(st.session_state.imagen_subida, catalogo_filtrado, "foto")

# ==========================================================
# SALA DE COMPARACIÓN
# ==========================================================
if st.session_state.comparador:
    st.write("---")
    st.markdown(
        f'<p class="subtitulo" style="color:{COLOR_NAVY};">🔍 Tu Sala de Comparación</p>',
        unsafe_allow_html=True)
    st.write("Observa lado a lado las monturas que te has probado.")
    num = len(st.session_state.comparador)
    cols = st.columns(min(num, 4))
    for idx, item in enumerate(st.session_state.comparador):
        with cols[idx % 4]:
            st.image(item["img"], caption=item["nombre"], use_container_width=True)
    if st.button("🗑️ Borrar todas las comparaciones"):
        st.session_state.comparador = []
        st.rerun()
