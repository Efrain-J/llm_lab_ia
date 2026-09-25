"""
Laboratorio de PLN y LLMs
=========================

Plataforma didactica en Streamlit para hacer visible lo que normalmente es
invisible en un pipeline de procesamiento de lenguaje natural:

  1. Esquemas de tokenizacion     - como distintos criterios parten el mismo texto.
  2. Token IDs                    - por que un ID depende del vocabulario que lo indexa.
  3. Bag of Words                 - como se pasa de tokens a vectores.
  4. Similitud / distancia coseno - como se mide el parecido entre frases.
  5. Esquema generativo           - como los parametros de muestreo cambian la
                                    salida de un modelo real servido por Groq.

Ejecutar con:  streamlit run main_app.py
"""

from __future__ import annotations

import html
import io
import re
import time
import zlib
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import tiktoken

    TIKTOKEN_INSTALLED = True
except ImportError:
    TIKTOKEN_INSTALLED = False

try:
    from groq import Groq

    GROQ_INSTALLED = True
except ImportError:
    GROQ_INSTALLED = False

try:
    from PIL import Image

    PILLOW_INSTALLED = True
except ImportError:
    PILLOW_INSTALLED = False

# RapidOCR se publica bajo dos nombres de paquete con APIs de salida distintas:
# rapidocr_onnxruntime (1.x) y rapidocr (2.x, unificado). Se detecta cual hay
# disponible al importar y la diferencia se absorbe en normalize_ocr_result().
RAPIDOCR_FLAVOR: str | None = None
RAPIDOCR_IMPORT_ERROR: str = ""
try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore

    RAPIDOCR_FLAVOR = "rapidocr_onnxruntime"
except Exception as _exc_v1:  # noqa: BLE001 - se reintenta con el otro paquete
    try:
        from rapidocr import RapidOCR  # type: ignore

        RAPIDOCR_FLAVOR = "rapidocr"
    except Exception as _exc_v2:  # noqa: BLE001
        RAPIDOCR_IMPORT_ERROR = f"{type(_exc_v1).__name__}: {_exc_v1} / {type(_exc_v2).__name__}: {_exc_v2}"


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Paleta pastel. Se fija tambien el color de texto de cada chip: si solo se
# fijara el fondo, el tema oscuro de Streamlit dejaria texto claro sobre fondo
# claro y los tokens serian ilegibles.
PALETTE = [
    "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff",
    "#a0c4ff", "#bdb2ff", "#ffc6ff", "#ffadad",
    "#d0f4de", "#e4c1f9", "#fcf6bd", "#b9fbc0",
]
TOKEN_TEXT_COLOR = "#1a1a1a"

STOPWORDS = {
    # espanol
    "a", "al", "ante", "con", "contra", "de", "del", "desde", "e", "el", "ella",
    "ellas", "ellos", "en", "entre", "era", "es", "esa", "ese", "eso", "esta",
    "este", "esto", "fue", "ha", "han", "hasta", "la", "las", "le", "les", "lo",
    "los", "mas", "me", "mi", "muy", "ni", "no", "o", "para", "pero", "por",
    "que", "se", "segun", "si", "sin", "sobre", "son", "su", "sus", "tambien",
    "te", "tu", "un", "una", "uno", "unos", "unas", "y", "ya",
    # ingles
    "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "will", "with",
}

# Corpus de ejemplo disenado para el laboratorio:
#   lineas 1-2: casi identicas          -> similitud lexica muy alta
#   linea 3:    parafrasis de las dos   -> mismo significado, casi sin terminos comunes
#   lineas 4-5: mismo tema, otro lexico -> el limite del Bag of Words
#   linea 6:    tema sin relacion       -> referencia baja
DEFAULT_CORPUS = """El gato duerme sobre el sofá de la sala.
El gato duerme encima del sofá de la sala.
Un felino descansa en el mueble del salón.
La bolsa de valores cerró la jornada con pérdidas.
El mercado bursátil terminó el día en números rojos.
Los modelos de lenguaje predicen el siguiente token de una secuencia."""

DEFAULT_SAMPLE = "El niño corrió rápidamente, ¿verdad? Tokenizar no es trivial."

GPT_PREFIX = "openai/gpt-oss"


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """Un token, venga del esquema que venga.

    Tener una unica estructura de salida es lo que permite que el render de
    colores, el Bag of Words y la similitud consuman cualquier tokenizador sin
    ramificar por esquema.
    """

    text: str   # texto crudo; puede contener espacios (tipico en BPE)
    id: int     # ID en el vocabulario correspondiente
    start: int  # offset inicial en el texto original, en caracteres
    end: int


@dataclass(frozen=True)
class OcrLine:
    """Una linea reconocida por el OCR.

    Mismo criterio que con Token: una unica estructura de salida, aqui para
    que el render y las metricas no tengan que ramificar segun la version de
    RapidOCR que haya instalada.
    """

    text: str
    score: float  # confianza del reconocimiento, entre 0 y 1


# ---------------------------------------------------------------------------
# Esquemas de tokenizacion
# ---------------------------------------------------------------------------

def _seg_whitespace(text: str) -> list[tuple[str, int, int]]:
    """Corta por espacios en blanco. El criterio mas ingenuo posible."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _seg_words(text: str) -> list[tuple[str, int, int]]:
    """Separa palabras de signos de puntuacion como unidades distintas."""
    return [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)
    ]


def _seg_chars(text: str) -> list[tuple[str, int, int]]:
    """Un token por caracter: vocabulario minimo, secuencias maximas."""
    return [(ch, i, i + 1) for i, ch in enumerate(text)]


@st.cache_resource(show_spinner=False)
def get_encoding(name: str):
    """Carga un codificador BPE de tiktoken.

    La primera llamada descarga el fichero de vocabulario desde la red y lo
    deja en cache en disco. Se envuelve en cache_resource porque el objeto no
    es serializable y es caro de construir.
    """
    return tiktoken.get_encoding(name)


def _seg_tiktoken(text: str, encoding_name: str) -> list[tuple[str, int, int, int]]:
    """Tokeniza con BPE real y reconstruye los offsets en caracteres.

    tiktoken opera sobre bytes, no sobre caracteres: un token puede ser la
    mitad de una letra acentuada. Por eso cada token se decodifica por separado
    con errors="replace", y los offsets se mapean de bytes a caracteres a
    traves de una tabla acumulada.
    """
    enc = get_encoding(encoding_name)
    ids = enc.encode(text)

    # char_byte_offsets[i] = posicion en bytes donde empieza el caracter i
    char_byte_offsets = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8"))
        char_byte_offsets.append(total)

    out: list[tuple[str, int, int, int]] = []
    byte_pos = 0
    for tid in ids:
        raw = enc.decode_single_token_bytes(tid)
        start = bisect_left(char_byte_offsets, byte_pos)
        byte_pos += len(raw)
        end = bisect_left(char_byte_offsets, byte_pos)
        out.append((raw.decode("utf-8", errors="replace"), start, end, tid))
    return out


# Registro de esquemas: nombre visible -> (tipo, implementacion)
SCHEMES: dict[str, tuple[str, object]] = {
    "Espacios en blanco": ("local", _seg_whitespace),
    "Palabras y puntuación": ("local", _seg_words),
    "Caracteres": ("local", _seg_chars),
    "Subword BPE - cl100k_base (GPT-3.5 / GPT-4)": ("tiktoken", "cl100k_base"),
    "Subword BPE - o200k_base (GPT-4o)": ("tiktoken", "o200k_base"),
}

LOCAL_FALLBACK = "Palabras y puntuación"


def scheme_is_bpe(scheme: str) -> bool:
    return SCHEMES[scheme][0] == "tiktoken"


def encoding_available(scheme: str) -> bool:
    """Indica si un esquema BPE puede cargarse (necesita tiktoken y cache o red)."""
    if not scheme_is_bpe(scheme):
        return True
    if not TIKTOKEN_INSTALLED:
        return False
    try:
        get_encoding(SCHEMES[scheme][1])
        return True
    except Exception:
        return False


def resolve_scheme(scheme: str) -> tuple[str, str | None]:
    """Degrada a un esquema local si el BPE elegido no esta disponible.

    La app tiene que seguir siendo utilizable sin red: sin esto, un fallo de
    descarga de tiktoken tumbaria las cinco pestanas.
    """
    if encoding_available(scheme):
        return scheme, None
    motivo = (
        "tiktoken no esta instalado"
        if not TIKTOKEN_INSTALLED
        else "no se pudo descargar el vocabulario BPE (se necesita red la primera vez)"
    )
    aviso = (
        f"El esquema '{scheme}' no esta disponible: {motivo}. "
        f"Se usa '{LOCAL_FALLBACK}' en su lugar."
    )
    return LOCAL_FALLBACK, aviso


def segment(text: str, scheme: str) -> list[tuple[str, int, int, int | None]]:
    """Devuelve (texto, inicio, fin, id_nativo).

    id_nativo es None cuando el esquema no trae vocabulario propio y hay que
    construirlo sobre el corpus.
    """
    kind, payload = SCHEMES[scheme]
    if kind == "local":
        return [(t, s, e, None) for t, s, e in payload(text)]
    return _seg_tiktoken(text, payload)


def normalize(piece: str, lowercase: bool) -> str:
    return piece.lower() if lowercase else piece


@st.cache_data(show_spinner=False)
def build_vocab(corpus: tuple[str, ...], scheme: str, lowercase: bool) -> dict[str, int]:
    """Construye el vocabulario sobre el corpus COMPLETO, no frase a frase.

    Esta es la idea central de la seccion de IDs: un token ID no es una
    propiedad de la palabra, sino del vocabulario que la indexo. Si el
    vocabulario se construyera por frase, la misma palabra tendria IDs
    distintos en cada linea. El 0 queda reservado para <unk>.
    """
    terms: set[str] = set()
    for line in corpus:
        for piece, _start, _end, _native in segment(line, scheme):
            terms.add(normalize(piece, lowercase))
    return {term: idx for idx, term in enumerate(sorted(terms), start=1)}


def tokenize(text: str, scheme: str, vocab: dict[str, int], lowercase: bool) -> list[Token]:
    tokens: list[Token] = []
    for piece, start, end, native_id in segment(text, scheme):
        tid = native_id if native_id is not None else vocab.get(normalize(piece, lowercase), 0)
        tokens.append(Token(piece, tid, start, end))
    return tokens


# ---------------------------------------------------------------------------
# Render de tokens con color
# ---------------------------------------------------------------------------

def color_for(text: str) -> str:
    """Color estable por contenido del token.

    Se usa un hash del texto en lugar de la posicion para que un token repetido
    reciba siempre el mismo color: eso hace visibles las repeticiones, que es
    justo lo que conecta esta vista con el Bag of Words.
    """
    return PALETTE[zlib.crc32(text.encode("utf-8")) % len(PALETTE)]


def _visible(text: str) -> str:
    """Hace visibles los caracteres de espaciado.

    Sin esto, los tokens BPE que incluyen el espacio inicial (por ejemplo
    " gato") se ven identicos a los que no lo incluyen, y se pierde la mitad
    de la leccion.
    """
    safe = html.escape(text)
    safe = safe.replace("\n", "<span style='opacity:.45'>&#8629;</span><br>")
    safe = safe.replace("\t", "<span style='opacity:.45'>&#8677;</span>")
    safe = safe.replace(" ", "<span style='opacity:.45'>&middot;</span>")
    return safe or "&nbsp;"


def render_tokens_html(tokens: list[Token], show_ids: bool = True) -> str:
    chips = []
    for tok in tokens:
        sub = (
            "<sub style='opacity:.6;font-size:.68em;padding-left:2px'>"
            f"{tok.id}</sub>"
            if show_ids
            else ""
        )
        chips.append(
            '<span style="background:{bg};color:{fg};padding:2px 5px;margin:2px;'
            "border-radius:5px;display:inline-block;font-family:ui-monospace,"
            'SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em">'
            "{body}{sub}</span>".format(
                bg=color_for(tok.text),
                fg=TOKEN_TEXT_COLOR,
                body=_visible(tok.text),
                sub=sub,
            )
        )
    return "<div style='line-height:2.5'>" + "".join(chips) + "</div>"


# ---------------------------------------------------------------------------
# Bag of Words, TF-IDF y coseno
# ---------------------------------------------------------------------------

def make_analyzer(scheme: str, lowercase: bool, drop_stopwords: bool):
    """Adapta el tokenizador elegido al formato que esperan los vectorizadores.

    Pasar este callable como analyzer a CountVectorizer y TfidfVectorizer es lo
    que mantiene coherente toda la app: si se dejara el tokenizador por defecto
    de scikit-learn, la matriz de similitud se calcularia sobre una
    segmentacion distinta a la que se ve en la pestana de tokenizacion.
    """

    def analyzer(doc: str) -> list[str]:
        terms = [normalize(p, lowercase) for p, _s, _e, _n in segment(doc, scheme)]
        # Los tokens BPE arrastran el espacio inicial; para el Bag of Words
        # interesa el termino, no su separador.
        terms = [t.strip() for t in terms]
        terms = [t for t in terms if t]
        if drop_stopwords:
            terms = [t for t in terms if t.lower() not in STOPWORDS]
        return terms

    return analyzer


def vectorize(docs: list[str], analyzer, representation: str, binary: bool, min_df: int):
    """Devuelve (matriz dispersa, lista de terminos)."""
    if representation == "TF-IDF":
        vec = TfidfVectorizer(analyzer=analyzer, min_df=min_df)
    else:
        vec = CountVectorizer(analyzer=analyzer, binary=binary, min_df=min_df)
    matrix = vec.fit_transform(docs)
    return matrix, list(vec.get_feature_names_out())


def heat_color(value: float) -> tuple[str, str]:
    """Interpola blanco -> azul. Devuelve (fondo, color de texto).

    Se calcula a mano en lugar de usar Styler.background_gradient para no
    arrastrar matplotlib como dependencia solo por un degradado.
    """
    value = float(np.clip(value, 0.0, 1.0))
    r = int(247 - 180 * value)
    g = int(251 - 130 * value)
    b = int(255 - 55 * value)
    fg = "#ffffff" if value > 0.6 else "#1a1a1a"
    return f"#{r:02x}{g:02x}{b:02x}", fg


def render_heatmap_html(matrix: np.ndarray, labels: list[str], invert_color: bool) -> str:
    """Matriz N x N como tabla HTML coloreada.

    invert_color hace que el color oscuro signifique siempre "mas parecido",
    tanto si se muestra similitud como si se muestra distancia.
    """
    header = ["<th style='padding:6px 8px'></th>"]
    for lab in labels:
        header.append(
            "<th style='padding:6px 8px;font-size:.8em;font-weight:600'>"
            f"{html.escape(lab)}</th>"
        )
    rows = ["<tr>" + "".join(header) + "</tr>"]

    for i, lab in enumerate(labels):
        row = [
            "<th style='padding:6px 8px;font-size:.8em;text-align:right;"
            f"font-weight:600'>{html.escape(lab)}</th>"
        ]
        for j in range(len(labels)):
            value = float(matrix[i, j])
            bg, fg = heat_color(1.0 - value if invert_color else value)
            row.append(
                f"<td style='background:{bg};color:{fg};padding:7px 10px;"
                "text-align:center;font-family:monospace;font-size:.86em;"
                f"border-radius:3px'>{value:.3f}</td>"
            )
        rows.append("<tr>" + "".join(row) + "</tr>")

    return (
        "<div style='overflow-x:auto'><table style='border-collapse:separate;"
        "border-spacing:3px'>" + "".join(rows) + "</table></div>"
    )


def softmax(logits, temperature: float) -> np.ndarray:
    """softmax(z / T), con resta del maximo por estabilidad numerica.

    Es literalmente lo que hace el parametro temperature en la API: dividir los
    logits antes de normalizar. T < 1 agudiza la distribucion, T > 1 la aplana.
    """
    z = np.asarray(logits, dtype=float) / max(temperature, 1e-6)
    z = z - z.max()
    exp = np.exp(z)
    return exp / exp.sum()


# ---------------------------------------------------------------------------
# Integracion con Groq
# ---------------------------------------------------------------------------

def error_text(exc: Exception) -> str:
    """Mensaje uniforme para cualquier error del SDK.

    Se captura Exception en lugar de las clases concretas del SDK porque la
    jerarquia de excepciones cambia entre versiones de groq; el tipo se muestra
    igualmente para no perder informacion de diagnostico.
    """
    return f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    return Groq(api_key=api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_catalog(api_key: str) -> list[dict]:
    """Catalogo de modelos. Doble funcion: valida la key y alimenta los selectores.

    Nada se hardcodea: si Groq anade o retira modelos, la lista se actualiza
    sola en la siguiente hora.
    """
    client = get_client(api_key)
    rows = []
    for model in client.models.list().data:
        rows.append(
            {
                "id": getattr(model, "id", ""),
                "owned_by": getattr(model, "owned_by", ""),
                "context_window": getattr(model, "context_window", None),
                "max_completion_tokens": getattr(model, "max_completion_tokens", None),
                "active": getattr(model, "active", True),
            }
        )
    return sorted(rows, key=lambda r: r["id"])


def default_model_index(model_ids: list[str]) -> tuple[int, str | None]:
    """Preselecciona un modelo GPT de Groq (openai/gpt-oss-*).

    Si no hay ninguno en el catalogo se cae al primero disponible en vez de
    reventar, avisando del motivo.
    """
    for idx, mid in enumerate(model_ids):
        if mid.startswith(GPT_PREFIX):
            return idx, None
    return 0, (
        f"No se encontro ningun modelo con prefijo '{GPT_PREFIX}' en el catalogo. "
        "Se preselecciona el primer modelo disponible."
    )


def build_params(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    frequency_penalty: float,
    presence_penalty: float,
    seed: int | None,
    reasoning_effort: str | None,
) -> dict:
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    params: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if seed is not None:
        params["seed"] = seed
    # reasoning_effort solo lo aceptan los modelos gpt-oss.
    if reasoning_effort and GPT_PREFIX in model:
        params["reasoning_effort"] = reasoning_effort
    return params


def stream_completion(client, params: dict, sink: dict):
    """Generador para st.write_stream que ademas recoge el usage final.

    En streaming, Groq manda las metricas de tokens en el ultimo chunk (dentro
    de x_groq.usage), no en la respuesta principal: por eso se van guardando en
    un diccionario mutable en lugar de devolverlas.
    """
    stream = client.chat.completions.create(stream=True, **params)
    for chunk in stream:
        extra = getattr(chunk, "x_groq", None)
        if extra is not None and getattr(extra, "usage", None) is not None:
            sink["usage"] = extra.usage
        if getattr(chunk, "usage", None) is not None:
            sink["usage"] = chunk.usage
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        piece = getattr(choices[0].delta, "content", None)
        if piece:
            yield piece


def usage_row(usage) -> dict:
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


# ---------------------------------------------------------------------------
# Puerta de entrada: API key
# ---------------------------------------------------------------------------

def api_key_gate() -> str:
    """Bloquea la app hasta validar la key contra el endpoint de modelos.

    La key vive solo en session_state durante la sesion: no se escribe a disco
    ni se registra en ningun log.
    """
    st.sidebar.header("Acceso")

    if not GROQ_INSTALLED:
        st.sidebar.error("El paquete 'groq' no esta instalado.")
        st.error(
            "Falta la dependencia 'groq'. Instala los requisitos con "
            "pip install -r requirements.txt y vuelve a ejecutar la app."
        )
        st.stop()

    if st.session_state.get("groq_key_ok"):
        st.sidebar.success("API key validada")
        if st.sidebar.button("Cambiar API key", use_container_width=True):
            st.session_state.pop("groq_key", None)
            st.session_state.pop("groq_key_ok", None)
            fetch_catalog.clear()
            st.rerun()
        return st.session_state["groq_key"]

    key = st.sidebar.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Se guarda solo en memoria, durante esta sesion del navegador.",
    )
    if st.sidebar.button("Validar y entrar", type="primary", use_container_width=True):
        if not key.strip():
            st.sidebar.error("Introduce una API key.")
        else:
            try:
                fetch_catalog(key)
            except Exception as exc:
                st.sidebar.error(f"No se pudo validar la key. {error_text(exc)}")
            else:
                st.session_state["groq_key"] = key
                st.session_state["groq_key_ok"] = True
                st.rerun()

    st.title("Laboratorio de PLN y LLMs")
    st.info(
        "Introduce tu API key de Groq en la barra lateral para comenzar. "
        "La key se valida contra el catalogo de modelos antes de abrir la plataforma."
    )
    st.caption(
        "Puedes generar una key gratuita en console.groq.com. "
        "La aplicacion no la almacena en disco."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Pestana 1: Tokenizacion
# ---------------------------------------------------------------------------

def tab_tokenization(scheme: str, lowercase: bool, show_ids: bool, corpus: list[str]) -> None:
    st.subheader("Esquemas de tokenizacion y token IDs")
    st.caption(
        "Un token ID no es una propiedad de la palabra, sino del vocabulario que "
        "la indexo. Los esquemas locales construyen su vocabulario sobre el corpus "
        "de la barra lateral; los esquemas BPE usan los IDs reales del modelo."
    )

    text = st.text_area(
        "Texto a tokenizar",
        value=DEFAULT_SAMPLE,
        height=110,
        key="token_text",
    )

    if not text.strip():
        st.info("Escribe algo para ver la tokenizacion.")
        return

    vocab = build_vocab(tuple(corpus), scheme, lowercase)
    tokens = tokenize(text, scheme, vocab, lowercase)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tokens", len(tokens))
    col2.metric("Tokens unicos", len({t.text for t in tokens}))
    col3.metric("Caracteres", len(text))
    col4.metric(
        "Caracteres por token",
        f"{len(text) / len(tokens):.2f}" if tokens else "0.00",
    )

    st.markdown(render_tokens_html(tokens, show_ids), unsafe_allow_html=True)

    st.caption(
        "Los espacios se dibujan como un punto atenuado y los saltos de linea "
        "como una flecha. El color depende del contenido del token, asi que un "
        "token repetido aparece siempre del mismo color."
    )

    with st.expander("Tabla de tokens"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "posicion": i,
                        "token": repr(t.text),
                        "id": t.id,
                        "inicio": t.start,
                        "fin": t.end,
                    }
                    for i, t in enumerate(tokens)
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Comparar todos los esquemas sobre este texto"):
        filas = []
        for nombre in SCHEMES:
            if not encoding_available(nombre):
                filas.append({"esquema": nombre, "tokens": None, "estado": "no disponible"})
                continue
            piezas = segment(text, nombre)
            filas.append(
                {
                    "esquema": nombre,
                    "tokens": len(piezas),
                    "caracteres por token": round(len(text) / len(piezas), 2) if piezas else 0.0,
                    "estado": "ok",
                }
            )
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        st.caption(
            "El BPE produce menos tokens que el criterio por caracteres y mas que "
            "el criterio por palabras: ese equilibrio entre tamano de vocabulario "
            "y longitud de secuencia es justo lo que se busca al entrenarlo."
        )


# ---------------------------------------------------------------------------
# Pestana 2: Bag of Words
# ---------------------------------------------------------------------------

def tab_bag_of_words(
    corpus: list[str], analyzer, binary: bool, min_df: int
) -> None:
    st.subheader("Bag of Words")
    st.caption(
        "Cada linea del corpus es un documento. El Bag of Words descarta el orden "
        "y se queda solo con que terminos aparecen y cuantas veces."
    )

    if len(corpus) < 1:
        st.info("Anade al menos una frase en el corpus de la barra lateral.")
        return

    try:
        matrix, terms = vectorize(corpus, analyzer, "Bag of Words", binary, min_df)
    except ValueError as exc:
        st.warning(
            f"No queda ningun termino tras aplicar los filtros ({exc}). "
            "Baja el min_df o desactiva las stopwords."
        )
        return

    dense = matrix.toarray()
    labels = [f"D{i + 1}" for i in range(len(corpus))]
    frame = pd.DataFrame(dense, index=labels, columns=terms)

    col1, col2, col3 = st.columns(3)
    col1.metric("Documentos", len(corpus))
    col2.metric("Terminos del vocabulario", len(terms))
    ceros = float((dense == 0).sum()) / dense.size * 100 if dense.size else 0.0
    col3.metric("Dispersion (ceros)", f"{ceros:.1f}%")

    st.dataframe(frame, use_container_width=True)

    st.caption(
        "Esa dispersion es el problema estructural del Bag of Words: la mayor parte "
        "de la matriz son ceros y el tamano crece con el vocabulario. Es la razon "
        "por la que despues aparecen los embeddings densos."
    )

    st.markdown("**Terminos mas frecuentes en el corpus**")
    totals = pd.Series(dense.sum(axis=0), index=terms).sort_values(ascending=False)
    # st.slider exige minimo < maximo: con un vocabulario diminuto (corpus de una
    # frase, min_df alto o stopwords agresivas) el rango se invertiria y la
    # llamada fallaria, asi que en ese caso se grafica todo sin slider.
    if len(terms) > 5:
        top_n = st.slider(
            "Cuantos terminos mostrar", 5, min(40, len(terms)), min(15, len(terms))
        )
    else:
        top_n = len(terms)
    st.bar_chart(totals.head(top_n))

    with st.expander("Ver el documento que genera cada fila"):
        st.dataframe(
            pd.DataFrame({"documento": labels, "texto": corpus}),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Pestana 3: Similitud de coseno
# ---------------------------------------------------------------------------

def tab_cosine(corpus: list[str], analyzer, binary: bool, min_df: int) -> None:
    st.subheader("Similitud y distancia de coseno entre frases")
    st.caption(
        "La unidad de comparacion es la frase: cada linea del corpus se vectoriza "
        "por separado y se compara contra todas las demas."
    )

    if len(corpus) < 2:
        st.info("Se necesitan al menos dos frases en el corpus para comparar.")
        return

    col_a, col_b = st.columns(2)
    representation = col_a.radio(
        "Representacion", ["Bag of Words", "TF-IDF"], horizontal=True
    )
    metric = col_b.radio(
        "Metrica", ["Similitud (cos)", "Distancia (1 - cos)"], horizontal=True
    )

    try:
        matrix, terms = vectorize(corpus, analyzer, representation, binary, min_df)
    except ValueError as exc:
        st.warning(
            f"No queda ningun termino tras aplicar los filtros ({exc}). "
            "Baja el min_df o desactiva las stopwords."
        )
        return

    sim = cosine_similarity(matrix)
    is_distance = metric.startswith("Distancia")
    shown = 1.0 - sim if is_distance else sim
    labels = [f"D{i + 1}" for i in range(len(corpus))]

    st.markdown(render_heatmap_html(shown, labels, invert_color=is_distance), unsafe_allow_html=True)
    st.caption(
        "El color oscuro significa siempre 'mas parecido'. En similitud la diagonal "
        "vale 1.000; en distancia, 0.000. Con vectores de conteo o TF-IDF no hay "
        "componentes negativas, asi que ambas metricas quedan acotadas en [0, 1]."
    )

    st.dataframe(
        pd.DataFrame({"documento": labels, "texto": corpus}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Pares ordenados por similitud**")
    pares = []
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            pares.append(
                {
                    "par": f"{labels[i]} - {labels[j]}",
                    "similitud": round(float(sim[i, j]), 4),
                    "distancia": round(1.0 - float(sim[i, j]), 4),
                    "frase A": corpus[i],
                    "frase B": corpus[j],
                }
            )
    pares.sort(key=lambda r: r["similitud"], reverse=True)
    st.dataframe(pd.DataFrame(pares), use_container_width=True, hide_index=True)

    st.markdown("**De donde sale el numero**")
    col_x, col_y = st.columns(2)
    idx_a = col_x.selectbox("Frase A", range(len(corpus)), format_func=lambda i: f"{labels[i]}: {corpus[i]}")
    idx_b = col_y.selectbox(
        "Frase B",
        range(len(corpus)),
        index=min(1, len(corpus) - 1),
        format_func=lambda i: f"{labels[i]}: {corpus[i]}",
    )

    dense = matrix.toarray().astype(float)
    # El coseno es el producto punto de los vectores normalizados: al desglosar
    # ese producto termino a termino se ve exactamente que palabras aportan.
    norms = np.linalg.norm(dense, axis=1)
    unit = dense / np.where(norms[:, None] == 0, 1.0, norms[:, None])
    contribuciones = unit[idx_a] * unit[idx_b]
    activos = np.nonzero(contribuciones)[0]

    if activos.size == 0:
        st.warning(
            f"{labels[idx_a]} y {labels[idx_b]} no comparten ningun termino: el coseno "
            "es exactamente 0. No es un fallo del calculo, es el limite de una "
            "representacion puramente lexica. Dos frases sinonimas sin palabras en "
            "comun tambien dan 0, y de ahi nace la necesidad de los embeddings."
        )
    else:
        detalle = pd.DataFrame(
            {
                "termino": [terms[k] for k in activos],
                "peso en A": np.round(unit[idx_a][activos], 4),
                "peso en B": np.round(unit[idx_b][activos], 4),
                "contribucion": np.round(contribuciones[activos], 4),
            }
        ).sort_values("contribucion", ascending=False)
        st.dataframe(detalle, use_container_width=True, hide_index=True)
        st.metric(
            f"Suma de contribuciones = coseno({labels[idx_a]}, {labels[idx_b]})",
            f"{float(sim[idx_a, idx_b]):.4f}",
        )


# ---------------------------------------------------------------------------
# Pestana 4: Tokens reales segun Groq
# ---------------------------------------------------------------------------

def tab_real_tokens(api_key: str, catalog: list[dict], corpus: list[str]) -> None:
    st.subheader("Conteo de tokens local frente al del modelo")
    st.caption(
        "El modelo informa de cuantos tokens de entrada ha consumido realmente. "
        "Comparar ese numero con los tokenizadores locales mide cuanto se acerca "
        "cada aproximacion."
    )

    model_ids = [row["id"] for row in catalog]
    if not model_ids:
        st.warning("El catalogo de modelos vino vacio.")
        return

    idx, aviso = default_model_index(model_ids)
    if aviso:
        st.info(aviso)
    model = st.selectbox("Modelo", model_ids, index=idx, key="count_model")

    text = st.text_area(
        "Texto a medir",
        value=corpus[0] if corpus else DEFAULT_SAMPLE,
        height=110,
        key="count_text",
    )

    if st.button("Consultar a Groq", type="primary", key="count_run"):
        if not text.strip():
            st.error("Escribe algun texto.")
            return
        client = get_client(api_key)
        try:
            with st.spinner("Consultando..."):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": text}],
                    max_completion_tokens=1,
                    temperature=0.0,
                )
        except Exception as exc:
            st.error(f"La llamada fallo. {error_text(exc)}")
            return

        remotos = getattr(response.usage, "prompt_tokens", None)

        filas = []
        for nombre in SCHEMES:
            if not encoding_available(nombre):
                filas.append({"tokenizador": nombre, "tokens": None, "diferencia": None})
                continue
            cuenta = len(segment(text, nombre))
            filas.append(
                {
                    "tokenizador": nombre,
                    "tokens": cuenta,
                    "diferencia": (remotos - cuenta) if remotos is not None else None,
                }
            )
        filas.append(
            {
                "tokenizador": f"Groq / {model} (prompt_tokens)",
                "tokens": remotos,
                "diferencia": 0 if remotos is not None else None,
            }
        )
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        st.info(
            "El conteo de Groq sera siempre algo mayor que el conteo crudo: "
            "prompt_tokens incluye el coste de la plantilla de chat, es decir los "
            "tokens de rol y los delimitadores que el modelo anade alrededor del "
            "mensaje. Esa diferencia casi constante no es un error de medicion."
        )


# ---------------------------------------------------------------------------
# Pestana 5: Esquema generativo
# ---------------------------------------------------------------------------

def tab_generation(api_key: str, catalog: list[dict]) -> None:
    st.subheader("Esquema generativo")
    st.caption(
        "Mismo prompt, distintos parametros de muestreo, para aislar el efecto "
        "de cada uno sobre la salida."
    )

    with st.expander("Catalogo de modelos disponibles en Groq", expanded=False):
        st.dataframe(pd.DataFrame(catalog), use_container_width=True, hide_index=True)
        st.caption(
            "context_window es cuantos tokens caben entre prompt y respuesta; "
            "max_completion_tokens es el techo de la respuesta sola."
        )

    model_ids = [row["id"] for row in catalog]
    if not model_ids:
        st.warning("El catalogo de modelos vino vacio.")
        return

    idx, aviso = default_model_index(model_ids)
    if aviso:
        st.info(aviso)
    model = st.selectbox("Modelo", model_ids, index=idx, key="gen_model")

    info = next((row for row in catalog if row["id"] == model), {})
    # Techo de respuesta leido del catalogo. Se fuerza un minimo por si un modelo
    # reporta un valor mas bajo que el arranque del slider.
    techo = max(int(info.get("max_completion_tokens") or 4096), 32)

    system_prompt = st.text_input(
        "System prompt (opcional)",
        value="Eres un asistente conciso que responde en espanol.",
        key="gen_system",
    )
    user_prompt = st.text_area(
        "Prompt",
        value="Explica en tres frases que es la tokenizacion en un modelo de lenguaje.",
        height=110,
        key="gen_prompt",
    )

    st.markdown("**Parametros de muestreo**")
    col1, col2, col3 = st.columns(3)
    temperature = col1.slider("temperature", 0.0, 2.0, 0.7, 0.05)
    top_p = col2.slider("top_p", 0.0, 1.0, 1.0, 0.05)
    max_tokens = col3.slider("max_completion_tokens", 16, techo, min(512, techo), 16)

    col4, col5, col6 = st.columns(3)
    frequency_penalty = col4.slider("frequency_penalty", -2.0, 2.0, 0.0, 0.1)
    presence_penalty = col5.slider("presence_penalty", -2.0, 2.0, 0.0, 0.1)
    reasoning_effort = None
    if GPT_PREFIX in model:
        reasoning_effort = col6.selectbox("reasoning_effort", ["low", "medium", "high"], index=1)
    else:
        col6.caption("reasoning_effort no aplica a este modelo.")

    usar_seed = st.checkbox("Fijar seed (hace reproducible el muestreo)", value=False)
    seed = st.number_input("seed", value=42, step=1) if usar_seed else None

    st.warning(
        "El learning rate NO aparece aqui a proposito: es un hiperparametro de "
        "entrenamiento, controla el tamano del paso al ajustar los pesos por "
        "descenso de gradiente. En inferencia los pesos ya estan congelados, asi "
        "que no existe tal control. Lo que si cambia la salida es el muestreo: "
        "temperature, top_p y las penalizaciones."
    )

    modo = st.radio(
        "Modo de ejecucion",
        ["Individual (streaming)", "Comparativo por temperatura"],
        horizontal=True,
    )

    client = get_client(api_key)

    if modo == "Individual (streaming)":
        if st.button("Generar", type="primary", key="gen_single"):
            if not user_prompt.strip():
                st.error("Escribe un prompt.")
                return
            params = build_params(
                model, system_prompt, user_prompt, temperature, top_p, max_tokens,
                frequency_penalty, presence_penalty,
                int(seed) if seed is not None else None, reasoning_effort,
            )
            sink: dict = {}
            inicio = time.perf_counter()
            try:
                st.write_stream(stream_completion(client, params, sink))
            except Exception as exc:
                st.error(f"La generacion fallo. {error_text(exc)}")
                return
            transcurrido = time.perf_counter() - inicio

            uso = usage_row(sink.get("usage"))
            cols = st.columns(4)
            cols[0].metric("Latencia", f"{transcurrido:.2f} s")
            cols[1].metric("Tokens de prompt", uso.get("prompt_tokens", "n/d"))
            cols[2].metric("Tokens generados", uso.get("completion_tokens", "n/d"))
            cols[3].metric("Tokens totales", uso.get("total_tokens", "n/d"))

    else:
        temps_texto = st.text_input(
            "Temperaturas a comparar (separadas por comas)", value="0.0, 0.7, 1.2"
        )
        if st.button("Generar comparativa", type="primary", key="gen_compare"):
            if not user_prompt.strip():
                st.error("Escribe un prompt.")
                return
            try:
                temps = [float(x.strip()) for x in temps_texto.split(",") if x.strip()]
            except ValueError:
                st.error("Las temperaturas deben ser numeros separados por comas.")
                return
            if not temps:
                st.error("Indica al menos una temperatura.")
                return
            temps = [float(np.clip(t, 0.0, 2.0)) for t in temps[:4]]

            columnas = st.columns(len(temps))
            for columna, temp in zip(columnas, temps):
                with columna:
                    st.markdown(f"**temperature = {temp}**")
                    params = build_params(
                        model, system_prompt, user_prompt, temp, top_p, max_tokens,
                        frequency_penalty, presence_penalty,
                        int(seed) if seed is not None else None, reasoning_effort,
                    )
                    try:
                        inicio = time.perf_counter()
                        with st.spinner("Generando..."):
                            respuesta = client.chat.completions.create(**params)
                        transcurrido = time.perf_counter() - inicio
                    except Exception as exc:
                        st.error(error_text(exc))
                        continue
                    st.write(respuesta.choices[0].message.content)
                    uso = usage_row(respuesta.usage)
                    st.caption(
                        f"{transcurrido:.2f} s | "
                        f"{uso.get('completion_tokens', 'n/d')} tokens generados"
                    )

            st.info(
                "Lanza la comparativa dos veces seguidas: la columna de temperature "
                "igual a 0.0 deberia repetirse casi palabra por palabra, mientras "
                "que las de temperatura alta divergiran. Esa es la prueba de que el "
                "parametro llega de verdad al muestreo."
            )

    with st.expander("Como actua la temperatura sobre la distribucion de salida"):
        st.caption(
            "Sin llamadas a la API: se parte de unos logits de juguete sobre seis "
            "tokens candidatos y se aplica softmax(z / T)."
        )
        temp_demo = st.slider("Temperatura de la demostracion", 0.05, 2.0, 1.0, 0.05, key="demo_t")
        logits = np.array([3.2, 2.7, 2.1, 1.4, 0.8, 0.1])
        candidatos = ["gato", "perro", "felino", "animal", "coche", "teorema"]
        probabilidades = softmax(logits, temp_demo)
        st.bar_chart(pd.DataFrame({"probabilidad": probabilidades}, index=candidatos))
        col_i, col_j = st.columns(2)
        col_i.metric("Probabilidad del token favorito", f"{probabilidades.max():.3f}")
        col_j.metric(
            "Entropia (bits)",
            f"{float(-(probabilidades * np.log2(probabilidades + 1e-12)).sum()):.3f}",
        )
        st.caption(
            "Al bajar la temperatura la masa se concentra en el favorito y la "
            "entropia cae: el modelo se vuelve predecible. Al subirla, la "
            "distribucion se aplana y entran candidatos improbables."
        )


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

OCR_TEMPLATES: dict[str, str] = {
    "Ampliar y explicar en detalle": (
        "Amplia y explica en detalle el texto extraido de una imagen que aparece "
        "mas abajo. Desarrolla los conceptos que menciona, aporta contexto y "
        "ejemplos concretos, y senala cualquier punto que quede ambiguo o "
        "incompleto en el original."
    ),
    "Resumir": (
        "Resume el texto extraido de una imagen que aparece mas abajo. Quedate "
        "con las ideas principales y presentalas en una lista breve."
    ),
    "Corregir ortografia y formato": (
        "Corrige la ortografia, la puntuacion y el formato del texto extraido de "
        "una imagen que aparece mas abajo. Ten en cuenta que procede de un OCR y "
        "puede contener errores de reconocimiento: reconstruye las palabras "
        "dudosas segun el contexto. Devuelve solo el texto corregido."
    ),
    "Traducir al ingles": (
        "Traduce al ingles el texto extraido de una imagen que aparece mas abajo, "
        "conservando su formato y su tono."
    ),
    "Instruccion propia": "",
}


@st.cache_resource(show_spinner=False)
def load_ocr_engine():
    """Instancia el motor de OCR.

    Va en cache_resource porque carga modelos ONNX en memoria: es un recurso
    vivo y no serializable, al contrario que el resultado del reconocimiento.
    """
    return RapidOCR()


def normalize_ocr_result(raw) -> list[OcrLine]:
    """Absorbe las dos formas de salida de RapidOCR.

    La version 2.x devuelve un objeto con .txts y .scores; la 1.x devuelve la
    tupla (lista de [caja, texto, score], tiempos). Normalizar aqui evita que
    la diferencia se propague al resto de la pestana.
    """
    if raw is None:
        return []

    if hasattr(raw, "txts"):  # rapidocr 2.x
        txts = list(raw.txts or [])
        scores = list(raw.scores or [])
        if not scores:
            scores = [1.0] * len(txts)
        return [OcrLine(str(t), float(s)) for t, s in zip(txts, scores)]

    # rapidocr_onnxruntime 1.x
    result = raw[0] if isinstance(raw, tuple) else raw
    if not result:
        return []

    lineas: list[OcrLine] = []
    for item in result:
        if len(item) >= 3:      # [caja, texto, score]
            lineas.append(OcrLine(str(item[1]), float(item[2])))
        elif len(item) == 2:    # [texto, score]
            lineas.append(OcrLine(str(item[0]), float(item[1])))
    return lineas


@st.cache_data(show_spinner=False)
def run_ocr_cached(image_bytes: bytes) -> tuple[list[dict], float]:
    """Ejecuta el OCR cacheando por CONTENIDO del archivo, no por nombre.

    Aqui el cacheo no es una optimizacion sino parte del diseno: el usuario va
    a iterar sobre la misma imagen probando plantillas de prompt distintas, y
    sin esto cada clic repetiria la inferencia ONNX entera. Devuelve dicts
    porque cache_data serializa su resultado.
    """
    engine = load_ocr_engine()
    imagen = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arreglo = np.array(imagen)

    inicio = time.perf_counter()
    raw = engine(arreglo)
    transcurrido = time.perf_counter() - inicio

    lineas = normalize_ocr_result(raw)
    return [{"text": ln.text, "score": ln.score} for ln in lineas], transcurrido


def render_ocr_lines_html(lineas: list[dict]) -> str:
    """Pinta cada linea con el color de su confianza.

    Reutiliza heat_color(), el mismo degradado del heatmap de coseno: azul
    oscuro es confianza alta, casi blanco es confianza baja. Asi se localiza
    de un vistazo donde el OCR ha dudado.
    """
    filas = []
    for i, linea in enumerate(lineas):
        score = float(linea["score"])
        bg, fg = heat_color(score)
        filas.append(
            f"<div style='background:{bg};color:{fg};padding:5px 9px;margin:3px 0;"
            "border-radius:4px;display:flex;justify-content:space-between;gap:12px'>"
            f"<span style='font-family:ui-monospace,Menlo,Consolas,monospace;"
            f"font-size:.88em'>{html.escape(linea['text'])}</span>"
            f"<span style='opacity:.8;font-size:.78em;white-space:nowrap'>"
            f"#{i + 1} &middot; {score:.2f}</span></div>"
        )
    return "<div>" + "".join(filas) + "</div>"


def build_ocr_prompt(instruccion: str, texto: str) -> str:
    """Compone instruccion + material con delimitadores explicitos.

    Los delimitadores importan: sin ellos el modelo no puede distinguir donde
    acaba tu encargo y donde empieza el texto de la imagen, y un documento que
    contenga algo parecido a una orden puede acabar interpretandose como tal.
    """
    return (
        f"{instruccion.strip()}\n\n"
        "--- TEXTO EXTRAIDO POR OCR ---\n"
        f"{texto.strip()}\n"
        "--- FIN DEL TEXTO EXTRAIDO ---"
    )


def tab_ocr(api_key: str, catalog: list[dict]) -> None:
    st.subheader("OCR: de la imagen al prompt")
    st.caption(
        "El reconocimiento corre en local con RapidOCR (PP-OCR sobre ONNX): no "
        "consume cuota de Groq ni necesita red. Solo la ampliacion de la "
        "respuesta llama al modelo."
    )

    if not PILLOW_INSTALLED:
        st.error(
            "Falta Pillow para decodificar imagenes. Instala las dependencias con "
            "pip install -r requirements.txt"
        )
        return

    if RAPIDOCR_FLAVOR is None:
        # Distinguir "no esta instalado" de "esta instalado pero le falta una
        # libreria del sistema" importa: son fallos con arreglos opuestos, y el
        # segundo es el tipico de OpenCV en Linux headless, donde pip no ayuda.
        falta_libreria_sistema = any(
            marca in RAPIDOCR_IMPORT_ERROR
            for marca in ("libGL", "libgthread", "libglib", "libSM", "libXext")
        )
        if falta_libreria_sistema:
            st.error(
                "El motor de OCR esta instalado, pero OpenCV no encuentra una "
                "libreria del sistema. Es el fallo tipico de OpenCV en Linux sin "
                "entorno grafico, y no se arregla con pip."
            )
            st.markdown(
                "- **Streamlit Community Cloud u otro despliegue desde GitHub:** "
                "el repositorio incluye un `packages.txt` con `libgl1` y "
                "`libglib2.0-0`. Confirma que esta subido y vuelve a desplegar.\n"
                "- **Docker, WSL o Linux local:** "
                "`sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0`\n"
                "- **Sin permisos de root:** "
                "`pip uninstall -y opencv-python && pip install opencv-python-headless`"
            )
        else:
            st.error(
                "El motor de OCR no esta instalado. Instalalo con "
                "pip install rapidocr-onnxruntime onnxruntime"
            )
        if RAPIDOCR_IMPORT_ERROR:
            with st.expander("Detalle del error de importacion"):
                st.code(RAPIDOCR_IMPORT_ERROR)
        st.caption("El resto de pestanas sigue funcionando con normalidad.")
        return

    st.caption(f"Motor detectado: {RAPIDOCR_FLAVOR}")

    archivo = st.file_uploader(
        "Imagen con texto",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="ocr_file",
    )
    if archivo is None:
        st.info("Sube una imagen para extraer su texto.")
        return

    datos = archivo.getvalue()

    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.image(datos, caption=archivo.name, use_container_width=True)

    with col_res:
        try:
            with st.spinner("Reconociendo texto..."):
                lineas, transcurrido = run_ocr_cached(datos)
        except Exception as exc:
            st.error(f"El OCR fallo. {error_text(exc)}")
            return

        if not lineas:
            st.warning(
                "No se reconocio ningun texto. Prueba con una imagen de mayor "
                "resolucion o con mas contraste entre el texto y el fondo."
            )
            return

        texto_extraido = "\n".join(ln["text"] for ln in lineas)
        confianzas = [float(ln["score"]) for ln in lineas]

        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("Lineas detectadas", len(lineas))
        m2.metric("Caracteres", len(texto_extraido))
        m3.metric("Confianza media", f"{sum(confianzas) / len(confianzas):.3f}")
        m4.metric("Tiempo de OCR", f"{transcurrido:.2f} s")

        st.markdown(render_ocr_lines_html(lineas), unsafe_allow_html=True)
        st.caption(
            "Cuanto mas claro es el fondo de una linea, menos seguro estuvo el "
            "motor de su lectura. Son las candidatas a revisar antes de enviar."
        )

    # El area de texto conserva su valor entre ejecuciones por tener key propia.
    # Al cambiar de imagen hay que refrescarla a mano, y la asignacion se hace
    # ANTES de instanciar el widget: modificar la clave de un widget ya creado
    # en la misma pasada lanzaria una excepcion de Streamlit.
    huella = zlib.crc32(datos)
    if st.session_state.get("ocr_digest") != huella:
        st.session_state["ocr_digest"] = huella
        st.session_state["ocr_text"] = texto_extraido

    st.markdown("**Texto extraido (editable)**")
    st.caption(
        "El OCR se equivoca. Corrige aqui lo que haga falta antes de enviarlo: "
        "es este texto, y no la transcripcion cruda, el que va al modelo."
    )
    texto_editado = st.text_area(
        "Texto extraido", height=200, key="ocr_text", label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Ampliar la respuesta con un modelo de Groq**")

    plantilla = st.selectbox("Que hacer con el texto", list(OCR_TEMPLATES.keys()), index=0)
    if plantilla == "Instruccion propia":
        instruccion = st.text_area(
            "Tu instruccion",
            value="Explica el texto anterior como si fuera para un estudiante de primer curso.",
            height=90,
            key="ocr_custom_instruction",
        )
    else:
        instruccion = OCR_TEMPLATES[plantilla]
        st.caption(instruccion)

    prompt_final = build_ocr_prompt(instruccion, texto_editado)
    with st.expander("Prompt exacto que se enviara"):
        st.code(prompt_final, language="text")
        st.caption(
            "Ver el prompt ensamblado es parte del ejercicio: la instruccion y el "
            "material van separados por delimitadores para que el modelo no "
            "confunda uno con otro."
        )

    model_ids = [row["id"] for row in catalog]
    if not model_ids:
        st.warning("El catalogo de modelos vino vacio; no se puede generar.")
        return

    idx, aviso = default_model_index(model_ids)
    if aviso:
        st.info(aviso)

    col_m, col_t, col_k = st.columns([2, 1, 1])
    model = col_m.selectbox("Modelo", model_ids, index=idx, key="ocr_model")
    temperature = col_t.slider("temperature", 0.0, 2.0, 0.6, 0.05, key="ocr_temp")

    info = next((row for row in catalog if row["id"] == model), {})
    techo = max(int(info.get("max_completion_tokens") or 4096), 32)
    max_tokens = col_k.slider(
        "max_completion_tokens", 64, techo, min(1024, techo), 64, key="ocr_max_tokens"
    )

    if st.button("Ampliar la respuesta", type="primary", key="ocr_generate"):
        if not texto_editado.strip():
            st.error("No hay texto que enviar.")
            return
        if plantilla == "Instruccion propia" and not instruccion.strip():
            st.error("Escribe una instruccion.")
            return

        params = build_params(
            model,
            "Eres un asistente que trabaja sobre texto extraido por OCR y responde en espanol.",
            prompt_final,
            temperature,
            1.0,
            max_tokens,
            0.0,
            0.0,
            None,
            "medium",
        )
        sink: dict = {}
        inicio = time.perf_counter()
        try:
            st.write_stream(stream_completion(get_client(api_key), params, sink))
        except Exception as exc:
            st.error(f"La generacion fallo. {error_text(exc)}")
            return
        elapsed = time.perf_counter() - inicio

        uso = usage_row(sink.get("usage"))
        cols = st.columns(4)
        cols[0].metric("Latencia", f"{elapsed:.2f} s")
        cols[1].metric("Tokens de prompt", uso.get("prompt_tokens", "n/d"))
        cols[2].metric("Tokens generados", uso.get("completion_tokens", "n/d"))
        cols[3].metric("Tokens totales", uso.get("total_tokens", "n/d"))


# ---------------------------------------------------------------------------
# Aplicacion
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Laboratorio de PLN y LLMs",
        page_icon=None,
        layout="wide",
    )

    api_key = api_key_gate()

    st.sidebar.divider()
    st.sidebar.header("Tokenizacion")
    scheme_elegido = st.sidebar.selectbox("Esquema", list(SCHEMES.keys()), index=1)
    scheme, aviso = resolve_scheme(scheme_elegido)
    if aviso:
        st.sidebar.warning(aviso)

    lowercase = st.sidebar.checkbox("Pasar a minusculas", value=True)
    show_ids = st.sidebar.checkbox("Mostrar token IDs", value=True)

    st.sidebar.divider()
    st.sidebar.header("Vectorizacion")
    drop_stopwords = st.sidebar.checkbox("Eliminar stopwords", value=False)
    binary = st.sidebar.checkbox("Bag of Words binario (presencia)", value=False)
    min_df = st.sidebar.slider("min_df (documentos minimos por termino)", 1, 5, 1)

    st.sidebar.divider()
    st.sidebar.header("Corpus")
    st.sidebar.caption("Una frase por linea. Lo comparten Bag of Words y similitud.")
    corpus_texto = st.sidebar.text_area(
        "Frases", value=DEFAULT_CORPUS, height=220, key="corpus", label_visibility="collapsed"
    )
    corpus = [linea.strip() for linea in corpus_texto.splitlines() if linea.strip()]

    analyzer = make_analyzer(scheme, lowercase, drop_stopwords)

    try:
        catalog = fetch_catalog(api_key)
    except Exception as exc:
        st.error(f"No se pudo leer el catalogo de modelos. {error_text(exc)}")
        catalog = []

    st.title("Laboratorio de PLN y LLMs")
    st.caption(
        f"Esquema activo: {scheme} | {len(corpus)} frases en el corpus | "
        f"{len(catalog)} modelos disponibles en Groq"
    )

    tabs = st.tabs(
        [
            "Tokenizacion",
            "Bag of Words",
            "Similitud de coseno",
            "Tokens reales (Groq)",
            "Generacion (Groq)",
            "OCR",
        ]
    )

    with tabs[0]:
        tab_tokenization(scheme, lowercase, show_ids, corpus or [DEFAULT_SAMPLE])
    with tabs[1]:
        tab_bag_of_words(corpus, analyzer, binary, min_df)
    with tabs[2]:
        tab_cosine(corpus, analyzer, binary, min_df)
    with tabs[3]:
        tab_real_tokens(api_key, catalog, corpus)
    with tabs[4]:
        tab_generation(api_key, catalog)
    with tabs[5]:
        tab_ocr(api_key, catalog)


if __name__ == "__main__":
    main()
