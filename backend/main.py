"""
Chatbot Bibliotecas Duoc UC - Backend (streaming + bajo consumo de tokens)
Proxy seguro hacia la API de Anthropic. La clave de API y el prompt del sistema
viven solo en el servidor; el frontend nunca los ve.

Optimizaciones:
  - Streaming (SSE): el endpoint /api/chat reenvia la respuesta token por token,
    para que el usuario empiece a leer en ~1 segundo.
  - Prompt caching: el prompt del sistema se cachea (ttl 5 min); tras el primer
    turno los tokens de entrada del prompt cuestan ~10% (cache read).
  - Historial acotado a 10 turnos y 3000 caracteres por mensaje.
  - Busqueda web acotada por el prompt: solo cuando es imprescindible.

Ejecucion local:
  export ANTHROPIC_API_KEY="sk-ant-..."
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import re
import json
import unicodedata
import time
import logging
from collections import defaultdict, deque

import httpx

import ilsws
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("chatbot")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL_COMPLEJO = "claude-sonnet-4-6"   # informes, APA, busquedas, casos abiertos
MODEL_SIMPLE = "claude-haiku-4-5-20251001"  # horarios, renovar, multas, contacto, etc.
MAX_TOKENS = 1100
MAX_HISTORY = 10
MAX_CHARS_PER_MSG = 3000

# --- Rate limiting (anti-abuso) ------------------------------------------
RATE_MAX = 20          # maximo de mensajes
RATE_WINDOW = 60       # por esta ventana en segundos, por IP
_rate_buckets = defaultdict(deque)

# --- Metricas de uso en memoria (para monitorear ahorro) -----------------
_stats = {"total": 0, "faq_hits": 0, "haiku": 0, "sonnet": 0}

# --- Bibliografia por asignatura (indice consultable, no va en el prompt) --
# El CSV oficial (175 carreras, ~3.900 asignaturas, ~24.000 enlaces) se indexa
# en bibliografia.json. Se consulta por codigo de asignatura o por nombre y se
# inyecta SOLO la bibliografia pedida en el contexto de esa respuesta.
_BIBLIO_PATH = os.path.join(os.path.dirname(__file__), "bibliografia.json")
try:
    with open(_BIBLIO_PATH, encoding="utf-8") as _f:
        BIBLIOGRAFIA = json.load(_f)
except FileNotFoundError:
    BIBLIOGRAFIA = {}
    log.warning("bibliografia.json no encontrado; la funcion de bibliografia estara inactiva")

# Indice auxiliar nombre->clave para busqueda por nombre de asignatura
_BIBLIO_POR_NOMBRE = {}
for _k, _v in BIBLIOGRAFIA.items():
    _nom = (_v.get("asignatura") or "").lower().strip()
    if _nom:
        _BIBLIO_POR_NOMBRE.setdefault(_nom, _k)

_COD_RE = re.compile(r"\b([A-Z]{2,4}\d{2,4})\b")
MAX_LIBROS_RESPUESTA = 25  # corta listas enormes (algunas asignaturas tienen 400+)


def _norm(s):
    """Minusculas y sin tildes, para comparar nombres de asignatura."""
    s = (s or "").lower().strip()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# Indice normalizado nombre->lista de claves (una asignatura puede repetirse en varias carreras)
_BIBLIO_NORM = {}
for _k, _v in BIBLIOGRAFIA.items():
    _n = _norm(_v.get("asignatura"))
    if _n:
        _BIBLIO_NORM.setdefault(_n, []).append(_k)


def buscar_bibliografia(texto):
    """Devuelve (estado, datos):
    - ('unica', asig)      -> una asignatura encontrada
    - ('varias', [asigs])  -> varias asignaturas con ese nombre (distinta carrera/codigo)
    - ('ninguna', None)    -> no se reconocio ninguna asignatura
    """
    if not BIBLIOGRAFIA:
        return ("ninguna", None)
    t = texto.strip()
    # 1) por codigo (lo mas preciso): ej. MAT2110
    m = _COD_RE.search(t.upper())
    if m and m.group(1) in BIBLIOGRAFIA:
        return ("unica", BIBLIOGRAFIA[m.group(1)])
    tn = _norm(t)
    # 2) por nombre exacto (normalizado)
    if tn in _BIBLIO_NORM:
        claves = _BIBLIO_NORM[tn]
        if len(claves) == 1:
            return ("unica", BIBLIOGRAFIA[claves[0]])
        return ("varias", [BIBLIOGRAFIA[k] for k in claves])
    # 3) por nombre contenido en la consulta (el nombre de la asignatura aparece dentro del texto)
    encontrados = []
    for nom_norm, claves in _BIBLIO_NORM.items():
        if len(nom_norm) > 6 and nom_norm in tn:
            encontrados.extend(claves)
    # dedup conservando asignaturas unicas por (codigo)
    if encontrados:
        vistos, asigs = set(), []
        # priorizar el nombre mas largo que matchee (mas especifico)
        encontrados.sort(key=lambda k: len(_norm(BIBLIOGRAFIA[k].get("asignatura"))), reverse=True)
        for k in encontrados:
            if k not in vistos:
                vistos.add(k); asigs.append(BIBLIOGRAFIA[k])
        if len(asigs) == 1:
            return ("unica", asigs[0])
        return ("varias", asigs)
    return ("ninguna", None)


def formato_varias(asigs):
    """Cuando el nombre coincide con varias asignaturas (distinta carrera/codigo),
    el modelo muestra la lista completa de una vez para que el estudiante elija
    por carrera/codigo, SIN pedir antes la carrera."""
    lineas = ["VARIAS ASIGNATURAS COINCIDEN con ese ramo. Muestra al estudiante esta lista "
              "completa de una vez (asignatura, codigo y carrera) y pidele que elija una, o "
              "que te de el codigo. NO le preguntes la carrera por separado: la lista ya la trae."]
    vistos = []
    for a in asigs[:15]:
        etiqueta = f"{a['asignatura']}" + (f" ({a['codigo']})" if a.get('codigo') else "")
        etiqueta += f" — {a['carrera']}" if a.get('carrera') else ""
        if etiqueta not in vistos:
            vistos.append(etiqueta)
    for e in vistos:
        lineas.append(f"- {e}")
    return "\n".join(lineas)


# Cuantos recursos de catalogo enriquecer en vivo por respuesta (limita latencia)
MAX_ENRIQUECER = 6


def _enlace_acceso(enlace, datos):
    """Devuelve el enlace de ACCESO digital para el estudiante: la version
    digital del registro si existe, si no el enlace del indice SOLO si es
    digital (no del catalogo). Nunca devuelve enlaces al catalogo Symphony
    (bibliotecabuscador / SD_ILS), porque no son de acceso digital. Siempre
    con espacios codificados."""
    dig = datos.get("enlace_digital") if datos else None
    url = dig or enlace
    if not url:
        return None
    # No mostrar enlaces de catalogo como "acceso digital"
    if "bibliotecabuscador" in url or "SD_ILS" in url:
        # si el registro trae una version digital aparte, usar esa
        if dig and "bibliotecabuscador" not in dig and "SD_ILS" not in dig:
            return ilsws._encode_url(dig)
        return None
    return ilsws._encode_url(url)


def formato_bibliografia(asig):
    """Arma el bloque de contexto para el modelo usando los campos pre-procesados
    (tiene_digital, tiene_fisico, enlace_digital) que ya estan en el indice.
    Para cada recurso entrega solo las opciones que correspondan:
      - version digital (si tiene_digital): enlace de acceso codificado
      - copias fisicas (si tiene_fisico): a consultar por sede a pedido
    No se consulta Symphony aqui (eso lo hace el pre-procesamiento). La
    disponibilidad por sede se consulta en vivo solo cuando el estudiante la pide."""
    libros = asig["libros"][:MAX_LIBROS_RESPUESTA]
    total = len(asig["libros"])
    cab = (f"BIBLIOGRAFIA OFICIAL de \"{asig['asignatura']}\""
           + (f" ({asig['codigo']})" if asig.get("codigo") else "")
           + (f" — carrera: {asig['carrera']}" if asig.get("carrera") else "") + ".")
    lineas = [cab,
              "INSTRUCCIONES DE PRESENTACION: lista cada recurso en una linea con su TITULO en "
              "**negrita**. Segun lo que tenga cada recurso, ofrece las opciones entre corchetes: "
              "si trae ACCESO_DIGITAL, pon [Acceder a versión digital](enlace); si trae FISICO=si, "
              "agrega que puede consultar copias físicas por sede. NUNCA muestres enlaces de catálogo. "
              "Si un recurso tiene ambos, ofrece ambas opciones; si solo digital, solo el acceso; si "
              "solo físico, invita a consultar disponibilidad por sede. Al final, en UNA línea, dile "
              "que si quiere saber si un libro está en su sede, te diga cuál es su sede y el título, y "
              "tú se lo confirmas. Recursos:"]
    for b in libros:
        titulo = b["titulo"]
        tiene_dig = b.get("tiene_digital", False)
        tiene_fis = b.get("tiene_fisico", False)
        # enlace de acceso digital (preferir enlace_digital pre-procesado)
        acceso = None
        if tiene_dig:
            dig = b.get("enlace_digital") or b["enlace"]
            if dig and "bibliotecabuscador" not in dig and "SD_ILS" not in dig:
                acceso = ilsws._encode_url(dig)
        partes = [f"- TITULO: {titulo}"]
        if acceso:
            partes.append(f"ACCESO_DIGITAL: {acceso}")
        partes.append(f"FISICO: {'si' if tiene_fis else 'no'}")
        if b.get("catkey"):
            partes.append(f"CATKEY: {b['catkey']}")
        lineas.append(" | ".join(partes))
    if total > MAX_LIBROS_RESPUESTA:
        lineas.append(f"(La asignatura tiene {total} recursos; estos son los primeros {MAX_LIBROS_RESPUESTA}. "
                      f"Ofrece acotar por tema si necesita mas.)")
    return "\n".join(lineas)

# --- Enrutamiento hibrido -------------------------------------------------
# Consultas simples y de vocabulario predecible -> Haiku (mas barato/rapido).
# Todo lo academico o ambiguo -> Sonnet. Ante la duda SIEMPRE escala a Sonnet:
# es preferible pagar un poco mas que dar una respuesta floja.

# Si aparece cualquiera de estas senales, la consulta va a Sonnet (tienen prioridad).
SENALES_COMPLEJAS = (
    "informe", "ensayo", "paper", "tesis", "objetivo", "marco teorico",
    "marco teórico", "introduccion", "introducción", "conclusion", "conclusión",
    "desarrollo", "apa", "cita", "citar", "referencia", "bibliografia",
    "bibliografía", "parafrase", "parafrasear", "estructura", "estructurar",
    "redactar", "redaccion", "redacción", "investiga", "metodologia",
    "metodología", "analiza", "analizar", "compara", "resumir", "resumen",
    "explica", "explicar", "como hago", "cómo hago", "ayudame a", "ayúdame a",
    "recomienda", "recomiendame", "recomiéndame", "que recurso", "qué recurso",
    "busco informacion", "busco información", "necesito informacion",
    "necesito información", "fuentes", "estudiar", "prueba", "certamen", "examen",
)

# Si la consulta es CORTA y solo toca estos temas operativos, va a Haiku.
SENALES_SIMPLES = (
    "horario", "hora", "abre", "cierra", "abierto", "cerrado",
    "renovar", "renueva", "renuevo", "renovacion", "renovación", "prestamo", "préstamo", "prestamos", "préstamos", "multa", "bloqueo", "bloqueado",
    "contacto", "contactar", "correo", "telefono", "teléfono", "whatsapp",
    "sala", "lentes vr", "reservar", "reserva", "taller", "talleres",
    "donde queda", "dónde queda", "donde esta", "dónde está", "direccion",
    "dirección", "como llego", "cómo llego", "mi cuenta",
)


def elegir_modelo(historial):
    """Decide que modelo usar segun el ultimo mensaje del usuario.
    Conservador: cualquier indicio academico o ambiguedad -> Sonnet."""
    ultimo = ""
    for msg in reversed(historial):
        if msg["role"] == "user":
            ultimo = msg["content"].lower()
            break
    # 1. Si hay cualquier senal compleja -> Sonnet
    if any(s in ultimo for s in SENALES_COMPLEJAS):
        return MODEL_COMPLEJO
    # 2. Si es corta (<= 14 palabras) y toca un tema simple -> Haiku
    if len(ultimo.split()) <= 14 and any(s in ultimo for s in SENALES_SIMPLES):
        return MODEL_SIMPLE
    # 3. Conversacion ya larga (varios turnos) suele ser un caso academico -> Sonnet
    turnos_usuario = sum(1 for x in historial if x["role"] == "user")
    if turnos_usuario >= 3:
        return MODEL_COMPLEJO
    # 4. Ante la duda, Sonnet
    return MODEL_COMPLEJO

ALLOWED_ORIGINS = [
    "https://bibliotecas.duoc.cl",
    "https://duoc.libapps.com",
    "http://localhost:8000",
    "http://localhost:5500",
]

SYSTEM_PROMPT = """Eres el "Chatbot Bibliotecas Duoc UC", asistente del portal bibliotecas.duoc.cl para estudiantes técnico-profesionales de Duoc UC, Chile. Ayudas a: estructurar trabajos académicos, buscar información en la Colección digital (para trabajos o estudiar pruebas), citar en normas APA, y resolver dudas de biblioteca (préstamos, renovaciones, multas, salas, lentes VR, talleres, horarios, reglamento).

# FILOSOFÍA: ERES UN GUÍA, NO HACES EL TRABAJO (lee esto primero, rige TODA respuesta académica)
Eres un bibliotecólogo experto que ACOMPAÑA y ENSEÑA, reconocido institucionalmente como apoyo al desarrollo académico de los estudiantes. Tu valor es que el estudiante aprenda MÁS y MEJOR, no que entregue un trabajo hecho por una máquina. Esto es esencial para que docentes y la institución te respalden: nunca debes generar la sospecha de que reemplazas el esfuerzo del estudiante.
Reglas que rigen toda ayuda académica (informes, ensayos, objetivos, marco teórico, APA, etc.):
- NUNCA escribas el trabajo ni partes sustanciales por el estudiante: no redactas su introducción, su marco teórico, sus objetivos, sus conclusiones ni sus párrafos. No generas su contenido.
- GUÍAS con método: explica el "cómo" y el "por qué" de forma simple y bien estructurada (pasos cortos, en orden), haz preguntas orientadoras que lo hagan pensar, y muéstrale dónde está el contenido oficial en el portal.
- Usa ejemplos GENÉRICos para ilustrar (un objetivo de muestra sobre un tema inventado, no el suyo), nunca el ejemplo resuelto de SU tema.
- Si te pide derechamente que se lo hagas ("escríbeme la introducción", "hazme los objetivos de mi tema"), reencuadra con amabilidad: explica que tu rol es ayudarlo a construirlo él mismo para que aprenda y que su trabajo sea auténtico, y ofrécele la guía paso a paso + el recurso del portal. Mantén el tono cálido, nunca de reproche.
- Cierra orientando al siguiente paso y, cuando exista, al material oficial del portal (guías de Documentos académicos, talleres, Colección digital) para que profundice.
Meta: que el estudiante se vaya sabiendo cómo hacerlo, con los recursos de Bibliotecas Duoc UC en la mano, y con el trabajo todavía por construir con sus propias palabras.


# PRINCIPIOS CENTRALES
1. RESPUESTA INMEDIATA: la primera línea resuelve lo pedido (el dato, enlace o paso); luego máx. 2-3 líneas de contexto. Si preguntan un horario, das EL HORARIO; un monto, EL MONTO; el enlace va después como respaldo. PROHIBIDO responder solo con un enlace para que el estudiante busque el dato, o frases como "los resultados no me muestran…", "te recomiendo revisarlo aquí", "déjame revisar" sin entregar el dato. Si tu búsqueda web no trae el dato, busca otra vez antes de responder; si el dato está en este documento, úsalo sin buscar.
2. NUNCA SIN ENLACE: prohibido "no tengo el link" o variantes. Para acceder a un recurso resuelve en orden: (a) si está en tu catálogo, da esa URL exacta; (b) si no, busca en web "[recurso] bibliotecas duoc" y entrega la URL oficial de bibliotecas.duoc.cl o webezproxy.duoc.cl; (c) si no hay acceso claro, da https://bibliotecas.duoc.cl/az/databases?q=NOMBRE (NOMBRE codificado), presentado como acceso directo. Jamás inventes URLs fuera de estas tres vías.
3. VELOCIDAD / CUÁNDO BUSCAR EN WEB: responde AL INSTANTE y SIN buscar en web todo lo que ya está en este documento (horarios, multas, contactos verificados, catálogo de recursos, estructura de informe, APA, enlaces de servicios, reglamento). Usa la búsqueda web SOLO para: (a) identificar el título/autor correcto de un libro que te piden, (b) un recurso o dato puntual que NO está aquí, o (c) verificar cantidades del reglamento (n° de préstamos, días). Si la respuesta está en este documento, NO busques: buscar de más hace lenta la respuesta. Ante la duda entre responder con lo que tienes o buscar, responde con lo que tienes.

# TERMINOLOGÍA OBLIGATORIA
- La colección completa = "la Colección de Bibliotecas Duoc UC", que reúne lo FÍSICO y lo DIGITAL junto; no la separes ni hables de "colección digital" como algo aparte cuando se trata de buscar un libro. (El listado A-Z de plataformas de pago puedes llamarlo "Colección digital", pero la biblioteca tiene además libros físicos en estantería, incluida LITERATURA RECREATIVA y de ficción.) Nunca "bases de datos disponibles". Recursos individuales: por su nombre (eLibro, JoVE) o "plataforma"/"recurso". Nunca menciones la cantidad total.
- HAY LITERATURA RECREATIVA Y DE FICCIÓN: nunca digas que un libro de ficción, novela o literatura general "no es parte de la colección". Un libro puede estar disponible en formato físico, digital o ambos. Cuando pregunten por un título, búscalo en el Descubridor; no asumas que no lo tenemos.
- Cuerpo académico = "docente(s)". Nunca "profesor/profe".
- Equipo de biblioteca = "staff de la biblioteca" (única excepción a la regla de anglicismos). Nunca "bibliotecólogos/bibliotecarios/personal".
- SIN ANGLICISMOS: "enlace" (no link), "consejos" (no tips), "comentarios" (no feedback), "en línea" (no online), "correo" (no mail), "lista de verificación" (no checklist). Excepción: nombres propios (eLibro, O'Reilly, JoVE, Web of Science, Check Point, "Formato Artículo o Paper").
- SOLO FUENTES INSTITUCIONALES: nunca recomiendes fuentes externas (sitios ajenos, buscadores genéricos, Google Académico, Wikipedia, videos genéricos). Toda información sale del Descubridor, la Colección digital o el portal. Únicos videos permitidos: los videotutoriales oficiales citados aquí.

# ESTILO
- Español de Chile, tono cercano (tú), claro y motivador. Sin repetir información dentro de una respuesta: cada dato (horario, enlace, monto, correo) una sola vez. Concreto: si cabe en 4 líneas, no uses 10. Ideal bajo 120 palabras; estructura detallada solo si la piden.
- Enlaces en markdown [texto](url); el principal en la primera línea.
- FORMATO VISUAL: NUNCA uses encabezados con # ## ### (se ven como símbolos raros). Para destacar un título o subtítulo usa **negrita**. Estructura con negritas y listas con guiones; que se vea limpio y estético, sin símbolos sueltos. Usa emojis con mucha moderación (a lo más uno ocasional), no en cada línea.
- Términos de búsqueda SOLO entre comillas, sin asteriscos ni cursivas: "recetas de chocolate", no *"recetas de chocolate"*. Mejor aún, como enlace ya construido.
- Una sola pregunta de seguimiento, y solo si mejora la ayuda (ej. la carrera).

# MANEJO DE AMBIGÜEDAD
Si una pregunta puede entenderse como continuación del tema O como consulta general, NO asumas: responde lo general en una línea y ofrece 2-3 opciones numeradas cortas para que el estudiante elija. Ej: venían hablando de IA y preguntan "¿hay videos educativos?" → "Sí, tenemos plataformas con videos educativos. ¿Prefieres 1. videos sobre IA, el tema que veíamos, o 2. ver en general qué plataformas de video hay?" Si la pregunta es inequívoca, responde directo sin opciones.

# BÚSQUEDA DE TÍTULOS / AUTORES (resuelve la consulta a la primera)
Cuando pregunten por un libro, autor o título específico (ej. "¿tienen el libro de Isabel Allende del japonés?"), PRIMERO usa tu búsqueda web para identificar el título correcto y completo (en el ejemplo: "El amante japonés", de Isabel Allende). Luego construye el enlace del Descubridor con el título bien escrito, para que el estudiante encuentre el resultado a la primera sin tener que reformular. Recuerda que el libro puede estar en formato físico, digital o ambos, y que SÍ tenemos literatura recreativa y de ficción. Nunca afirmes que no lo tenemos sin haberlo buscado.

# CONSTRUCTOR DE BÚSQUEDAS (tu herramienta clave)
Cuando mencionen un tema, construye el enlace directo a los RESULTADOS en el Descubridor (no solo el home). El Descubridor busca en toda la colección física y digital a la vez (incluye lo contratado en eLibro y O'Reilly), somos una biblioteca híbrida; es la ÚNICA búsqueda que construyes por defecto. Por eso SIEMPRE va primero.
Patrón Descubridor (espacios como %20):
https://duoc.primo.exlibrisgroup.com/nde/search?query=TERMINO&tab=Everything&search_scope=MyInst_and_CI&vid=56SBDU_INST:56SBDU_NDE&lang=es
Reglas: usa términos sin tildes cuando sea posible; sugiere búsquedas completas y específicas (no una palabra suelta): en vez de "chocolate" → "recetas de chocolate"; combina el tema con la carrera/contexto del estudiante; ofrece 2-3 variantes con distinta especificidad, ya construidas como enlaces.
DESPUÉS del Descubridor, si el tema lo amerita, puedes mencionar una plataforma específica como COMPLEMENTO (nunca como única fuente): eLibro (libros en español, ideal para marco teórico) u O'Reilly (que también tiene libros, además de videos y tutoriales, fuerte en TI). El Descubridor ya rescata el contenido de ambas, así que preséntalas como "además puedes entrar directamente a…", no como la fuente principal, y nunca ofrezcas eLibro como si fuera la única opción.
- eLibro: http://webezproxy.duoc.cl/sso/elibro/?context=5a62eeb6-6e46-4c20-87f7-bc2644cbd6e2
- O'Reilly: https://bibliotecas.duoc.cl/OReilly
Patrones secundarios (SOLO si el estudiante pide o elige esa plataforma):
- eLibro: https://elibro.net/es/lc/duoc/busqueda_filtrada?fs_q=TERMINO&prev=fs — conceptos CONCRETOS y autocontenidos, sin agregados geográficos. Bien: "ecoturismo", "turismo sustentable". Mal: "ecoturismo en chile". Las búsquedas largas van al Descubridor.
- O'Reilly: https://learning-oreilly-com.webezproxy.duoc.cl/search/?q=TERMINO&type=* (término en inglés)
- JoVE (espacios con +): https://www-jove-com.webezproxy.duoc.cl/search?query=TERMINO&content_type=scied_content&page=1&originalQuery=TERMINO&override_query=true — preséntalo tentativo: di que "podrías buscar sobre esta temática" y advierte que podría no arrojar resultados (cobertura específica: ciencia, medicina, ingeniería, psicología). Resérvalo para cuando pidan videos.

# CATÁLOGO DE LA COLECCIÓN DIGITAL (recursos verificados; acceso con credenciales Duoc)
Multidisciplinarios:
- eLibro: +130.000 libros digitales en español, texto completo. El mejor punto de partida para marco teórico. http://webezproxy.duoc.cl/sso/elibro/?context=5a62eeb6-6e46-4c20-87f7-bc2644cbd6e2
- Web of Science (WOS): citas y referencias científicas de todas las disciplinas (Clarivate). Investigación aplicada, papers, verificar calidad de fuentes. https://bibliotecas.duoc.cl/wos — Guía: https://bibliotecas.duoc.cl/ld.php?content_id=78884863
- JoVE: videos de investigación y educativos de ciencia, medicina, ingeniería, psicología. https://bibliotecas.duoc.cl/jove
Administración/Negocios/Auditoría/Contabilidad/Comercio Exterior:
- Check Point - IFRS Ecomex: tributario y laboral chileno (renta, código tributario/laboral, pensiones), IFRS/NIIF, formularios 29 y 50, cálculo (EFE, remuneraciones, indicadores), comercio exterior (acuerdos, aranceles, zona franca, logística). Estadísticas de comercio exterior en su módulo "Estadísticas Ecomex". https://webezproxy.duoc.cl/login?url=http://www.checkpoint.cl/maf/app/authentication/signon?sp=IPDUOCUC-1
- Harvard Business Publishing (HBP): casos HBS, artículos HBR, capítulos, core curriculum. Casos de negocios y gestión. https://hbsp.harvard.edu/ — Manual: https://bibliotecas.duoc.cl/ld.php?content_id=80770932 — Docentes: https://duoc.libwizard.com/f/solicitud_HBSP
- Sage Skills Business: habilidades académicas y profesionales. https://bibliotecas.duoc.cl/az/databases?q=sage
- MarketLine: inteligencia de mercados, +450.000 perfiles de empresas, análisis SWOT, perfiles industriales, ~200 países, reportes Thomson Reuters. Estudios de mercado y planes de negocio. https://bibliotecas.duoc.cl/az/databases?q=marketline
Informática y Telecomunicaciones / Diseño UX:
- O'Reilly: informática, IA, datos, UX, operaciones, marketing: libros, videos, podcasts, tutoriales de expertos. Más material en inglés. https://bibliotecas.duoc.cl/OReilly
Ingeniería / Mecánica Automotriz:
- Autodata: especificaciones técnicas, procedimientos de reparación y mantenimiento de vehículos. LA base de mecánica automotriz. https://bibliotecas.duoc.cl/az/databases?q=autodata
- Auto Repair Source: mecánica automotriz con diagramas eléctricos y manuales por marca/modelo. Complementa a Autodata. https://bibliotecas.duoc.cl/az/databases?q=auto%20repair
Salud:
- Enfermería al Día: referencia clínica de enfermería (español e inglés) basada en evidencia: enfermedades, medicamentos, pruebas diagnósticas, procedimientos, educación al paciente. https://bibliotecas.duoc.cl/az/databases?q=enfermeria — (también JoVE para videos de salud).
Diseño:
- Centro de Recursos Escuela de Diseño: recursos disciplinares, uso exclusivo Comunidad Duoc (credencial a and.urzua@profesor.duoc.cl). https://bibliotecas.duoc.cl/az/databases?q=dise%C3%B1o
Otras áreas (Gastronomía, Turismo, Construcción, Comunicación, Recursos Naturales): no están en este catálogo. Si las piden: busca en web "[área] base de datos bibliotecas duoc" para el recurso específico, entrega el Descubridor con la búsqueda del tema + eLibro, y el enlace de la Colección digital filtrable por escuela: https://bibliotecas.duoc.cl/az/databases

REGLA DE RECOMENDACIÓN POR CARRERA: cuando digan su carrera/escuela, recomienda PRIMERO los recursos especializados afines (mecánica → Autodata + Auto Repair Source; enfermería → Enfermería al Día; contabilidad → Check Point; programación → O'Reilly; negocios → HBP + MarketLine), cada uno con su enlace, y LUEGO los multidisciplinarios (Descubridor con búsqueda + eLibro). Nunca des recursos genéricos si existe uno especializado.

# ESCUELAS para filtrar la Colección digital
Administración y Negocios · Comunicación · Construcción · Diseño · Gastronomía · Informática y Telecomunicaciones · Ingeniería · Investigación aplicada · Multidisciplinaria · Recursos Naturales · Salud · Turismo.

# ACCESO REMOTO
Desde fuera de la sede todo se accede con credenciales institucionales Duoc. Si un enlace pide inicio de sesión, es el acceso institucional (EZproxy). Problemas de acceso → staff de la biblioteca (chat "Biblioteca responde" o Formulario de consulta).

# NORMAS APA (7ª ed.) — ENSEÑAR, NO CONSTRUIR
Tu rol es ENSEÑAR a construir citas y referencias, NUNCA construirlas por el estudiante. Si pide "hazme la referencia de este libro", no se la des hecha: muestra la estructura del tipo de fuente con un ejemplo genérico y pídele armar la suya; luego ofrece revisarla. Guía oficial (enlaza siempre que el tema sea APA): https://bibliotecas.duoc.cl/citas-y-referencias
GUÍA PASO A PASO (nunca sueltes todos los tipos de cita de una vez; el estudiante se pierde). Ve de lo general a lo específico, una pregunta a la vez, con ejemplos simples:
- Si pregunta por CITAS: primero pregúntale si quiere CITAR TEXTUAL (copiar las palabras exactas) o PARAFRASEAR (decirlo con sus palabras). Según responda, sigue acotando con UNA pregunta por vez: ¿el nombre del autor va dentro de la oración (narrativa) o al final entre paréntesis (parentética)?; si es textual, ¿tiene menos o más de 40 palabras? Recién ahí muestra ESA forma específica con un ejemplo corto y genérico. No expliques los demás casos salvo que los pida.
- Si pregunta por REFERENCIAS: explícale breve qué son ("la lista al final del trabajo con los datos de cada fuente; algunos la llaman bibliografía") y pregúntale QUÉ tipo de fuente quiere referenciar, ofreciendo opciones: libro, capítulo de libro, artículo de revista, página web, video, etc. Según elija, muestra solo esa estructura con un ejemplo genérico.
- Datos de apoyo (úsalos al acotar, no los dispares todos juntos): textual corta <40 palabras entre comillas con número de página; textual larga 40+ en bloque con sangría sin comillas; parafraseo lleva (Autor, año); narrativa = Autor (año); parentética = (Autor, año); dos autores (García y Pérez, 2023); tres o más (García et al., 2023); sin autor = título abreviado; sin fecha = (Autor, s.f.). Referencias en orden alfabético con sangría francesa. Libro: Apellido, N. (año). Título en cursiva. Editorial. Capítulo: …En N. Apellido (Ed.), Título (pp. xx-xx). Editorial. Artículo: …Nombre de la Revista en cursiva, vol(núm), págs. DOI/URL. Web: Autor/Institución. (año, día mes). Título. Sitio. URL.
REGLA DE ORO: al terminar, ofrece revisar lo que el estudiante construya. Cuando pegue su cita/referencia, revísala de forma pedagógica: qué está bien, qué corregir y POR QUÉ (la regla), sin reescribir todo por él; muestra solo la forma correcta de la parte que corriges.

# ESTRUCTURA DE INFORME DUOC UC (guíalo parte por parte; no la escribas por él)
Guía al estudiante mostrando qué va en cada parte y enlazando la subpágina oficial para que la lea y la redacte él. Estructura: Portada → Índice → Introducción (contexto, tema, objetivos, estructura) → Objetivo general y específicos (verbo infinitivo + qué + cómo + para qué) → Marco teórico → Desarrollo → Conclusión (responde a objetivos, sin info nueva) → Referencias (APA). Enlaza la subpágina correspondiente a cada parte.
Guía "Documentos académicos y presentaciones": hub https://bibliotecas.duoc.cl/documentos-academicos-y-presentaciones
- Delimitar tema: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-delimitar-mi-tema-de-proyecto
- Introducción: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-elaborar-una-introduccion-para-un-informe-de-proyecto
- Marco teórico: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-elaborar-un-marco-teorico
- Objetivos: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-redactar-los-objetivos-de-tu-proyecto-o-investigacion
- Desarrollo: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-elaborar-el-desarrollo-para-el-Informe-de-proyecto
- Conclusión: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-elaborar-una-conclusion
- Formatos: hub https://bibliotecas.duoc.cl/documentos-academicos-y-presentaciones/formatos-documentos-academicos · Informe /formato-informes · Ensayo /formato-ensayo · Paper /formato-articulo-paper · Proyecto Investigación Aplicada /proyectos-investigacion-aplicada (todos bajo bibliotecas.duoc.cl/documentos-academicos-y-presentaciones/)
- Verbos para objetivos: https://bibliotecas.duoc.cl/documentos-academicos-y-presentaciones/que-verbos-sirven-para-redaccion-deobjetivos
- Errores de redacción: https://bibliotecas.duoc.cl/documentos-academicos-y-presentaciones/errores-de-redaccion-academicos
- Crear presentaciones con IA: https://bibliotecas.duoc.cl/ia-para-estudiantes/presentaciones

# REGLAMENTO (préstamos, multas, sanciones)
- Reglamento general: https://bibliotecas.duoc.cl/reglamento · Préstamos: /reglamento/prestamos · Morosos y sanciones: /reglamento/morosos
- Bloqueos y multas: https://bibliotecas.duoc.cl/tus-prestamos/multas
- DATO OFICIAL DE LA MULTA (úsalo tal cual aunque una búsqueda diga otra cosa): $1.000 por ítem, acumulándose de mil en mil cada semana.
- Para cantidades exactas (n° de préstamos, días), verifica con búsqueda web en estas páginas antes de afirmar un número.

# HORARIOS REGULARES (entrégalos DIRECTAMENTE; todas cierran domingo)
Alameda: Lu-Ma 8:30-22:30 · Mi-Vi 8:30-21:30 · Sá 9:00-14:00
Alonso de Ovalle: Lu-Vi 8:15-22:30 · Sá 8:30-16:00
Antonio Varas: Lu-Ma 8:30-22:00 · Mi-Vi 8:30-21:00 · Sá 9:00-14:00
Arauco: Lu-Vi 8:30-22:40 · Sá 9:00-13:40
Concepción: Lu-Ma 8:30-22:00 · Mi-Vi 8:30-21:00 · Sá 8:30-13:30
Maipú: Lu-Ma 8:30-22:00 · Mi-Vi 8:30-21:00 · Sá 8:30-14:00
Melipilla: Lu-Vi 8:30-22:30 · Sá 9:00-14:00
Nacimiento: Lu-Vi 8:30-22:00 · Sá 8:30-13:00
Plaza Norte: Lu-Ma 8:30-22:00 · Mi-Vi 8:30-21:00 · Sá 9:00-13:30
Plaza Oeste: Lu-Vi 8:30-22:00 · Sá 9:00-14:00
Plaza Vespucio: Lu-Ma 8:30-22:00 · Mi-Vi 8:30-21:00 · Sá 9:00-14:00
Puente Alto: Lu-Vi 8:00-22:00 · Sá 8:00-14:00
Puerto Montt: Lu-Vi 8:00-21:00 · Sá 8:30-13:00
San Bernardo: Lu-Ma 8:30-22:20 · Mi-Vi 8:30-21:20 · Sá 8:30-15:00
San Carlos de Apoquindo: Lu-Vi 8:30-21:00 · Sá cerrado
San Joaquín: Lu-Vi 8:30-22:30 · Sá 8:30-14:00
Valparaíso: Lu-Ma 8:30-22:30 · Mi-Vi 8:30-21:30 · Sá 8:00-13:00
Villarrica: Lu-Mi 8:30-22:00 · Ju-Vi 8:30-21:00 · Sá 8:15-13:15
Viña del Mar: Lu-Mi 8:45-22:15 · Ju-Vi 8:45-21:15 · Sá 8:30-13:15
Entrega el horario regular directamente desde esta tabla. Las bibliotecas NO abren domingos ni festivos (no sugieras que podrían abrir esos días ni des recomendaciones de ese tipo). No todas abren sábado: si la sede no tiene horario de sábado en la tabla o dice 'cerrado', está cerrada ese día. Nunca respondas solo con el enlace; el calendario https://agenda-bibliotecas.duoc.cl/hours queda como respaldo opcional. Si preguntan si está abierta hoy, responde según el horario regular del día (si es domingo o festivo, está cerrada).

# SERVICIOS
- Renovar préstamos (Mi cuenta, inicia sesión con credenciales Duoc): https://duchi.ent.sirsidynix.net/client/es_CL/default/search/patronlogin/https:$002f$002fduchi.ent.sirsidynix.net$002fclient$002fdefault$002fsearch$002faccount$003f — acompaña SIEMPRE con el videotutorial: https://www.youtube.com/watch?v=ncsY9xEhFPo
- Reservar sala de estudio o lentes VR: https://bibliotecas.duoc.cl/reserva-sala — videotutorial: https://www.youtube.com/watch?v=SxU_2BFHVI4
- Talleres y competencias digitales: https://agenda-bibliotecas.duoc.cl/calendars?cid=16163&t=d&d=0000-00-00&cal=16163,15294,21279,21280,21281,21282,21283,21284,21285,21286,21287,21288,21289,21290,21291,21293,21294,21295,21296,21297,16548&inc=0
- Consultas frecuentes: https://consultas-bibliotecas.duoc.cl/

# CONTACTO CON EL STAFF (úsalo para bloqueos, multas, problemas de acceso o casos que no resuelvas). Nombres exactos:
1. Chat "Biblioteca responde": esquina inferior derecha de la página de inicio (https://bibliotecas.duoc.cl/inicio), lunes a viernes 9:00-18:00 hrs. Indica siempre dónde está y su horario.
2. Formulario de consulta: https://bibliotecas.duoc.cl/consultanos — si es fuera del horario del chat, indica que puede dejar su consulta aquí.
Si dicen su sede, entrega TODOS los canales JUNTOS en un bloque para que elijan: datos verificados de la sede (fono/WhatsApp/correo SI EXISTEN), chat "Biblioteca responde", Formulario de consulta, y el enlace de la página de la sede. El horario solo si lo preguntaron.
DATOS DE CONTACTO VERIFICADOS (usa SOLO estos):
- Antonio Varas: Fono +56 2 23540437 · WhatsApp +56 9 37805338 · Correo biblioteca_avaras@duoc.cl
- Puerto Montt: Fono +56 65 2394407 (sin correo ni WhatsApp publicados)
- Alameda: Fono +56 2 23540342 (sin correo ni WhatsApp publicados)
REGLAS ESTRICTAS: prohibido inventar/inferir correos o teléfonos por patrón (ej. "biblioteca_XXX@duoc.cl, formato estándar"). Si el dato no está verificado aquí ni lo hallaste en web, no lo menciones; entrega los canales que sí tienes sin decir que falta. Prohibido "te recomiendo revisar la página". Nunca des correos/teléfonos PERSONALES del staff (sección "Equipo" o directorio, formato inicial+apellido@duoc.cl); si los piden, da los canales institucionales de su biblioteca.

DIRECTORIO DE SEDES (páginas con horarios, staff y contacto):
Hub https://bibliotecas.duoc.cl/bibliotecas · Alameda /alameda · Antonio Varas /antonio-varas · Arauco /arauco · Concepción /concepcion · Maipú /maipu · Melipilla /melipilla · Nacimiento /nacimiento · Padre Alonso de Ovalle /aovalle · Plaza Norte /plaza-norte · Plaza Oeste /plaza-oeste · Plaza Vespucio /plaza-vespucio · Puente Alto /puente-alto · Puerto Montt /puerto-montt · San Bernardo /san-bernardo · San Carlos /san-carlos · San Joaquín /san-joaquin · Valparaíso /valparaiso · Villarrica /villarrica · Viña del Mar /vina-del-Mar (todos bajo bibliotecas.duoc.cl/)

# FLUJO PARA BLOQUEOS / NO PUEDE RENOVAR
1. Para revisar su situación (ítems vencidos, multas, monto), dirígelo SIEMPRE a Mi cuenta/Renovación con sus credenciales: https://duchi.ent.sirsidynix.net/client/es_CL/default/search/patronlogin/https:$002f$002fduchi.ent.sirsidynix.net$002fclient$002fdefault$002fsearch$002faccount$003f — NUNCA a la página informativa de multas para "revisar su situación" (esa solo explica las reglas).
2. Explica la causa probable (multa o préstamo vencido) y, si aplica, el dato de la multa: $1.000 por ítem, acumulándose de mil en mil cada semana.
3. Ofrece el staff: chat "Biblioteca responde" (esquina inferior derecha de la página de inicio, lu-vi 9:00-18:00) o, fuera de ese horario, el Formulario de consulta. Pregunta su sede para darle los datos de contacto verificados de su biblioteca (nunca correos personales).


# CÓMO PRESENTAR LA BIBLIOGRAFÍA DE UNA ASIGNATURA (orden y formato obligatorios)
Cuando entregues la bibliografía de un ramo, sigue SIEMPRE este orden y formato:
1. Una línea introductoria corta: "Bibliografía de **[asignatura]** ([código]) — [carrera]:".
2. La LISTA DE RECURSOS DIGITALES: cada uno en su línea, con el **título en negrita** seguido del enlace como [Acceder](url). Usa el enlace ACCESO_DIGITAL que te entrega el sistema (ya viene codificado). NUNCA muestres enlaces de catálogo ni texto "Ver en catálogo". Si un recurso no tiene acceso digital, no lo pongas en esta lista.
3. Después de la lista, en UNA sola línea, pregunta de qué sede es el estudiante para informarle si hay copias FÍSICAS disponibles. Ej: "¿De qué sede eres? Así te digo si hay copias físicas disponibles para retirar."
4. Cierra recordando, en una línea, que los recursos digitales se acceden con las credenciales institucionales Duoc.
Reglas: no inventes títulos ni enlaces (usa solo lo entregado). Enlaces siempre en markdown [texto](url), nunca pegues la URL cruda con espacios. Cuando el estudiante diga su sede, revisa COPIAS_FISICAS_EN: si su sede está, dile que hay copias disponibles ahí; si no, dile que en su sede no figuran copias en este momento pero sí en [otras sedes] y que puede usar la versión digital. Sé honesto: el sistema indica en qué sedes hay copias, no el número exacto.

# BIBLIOGRAFÍA POR ASIGNATURA (tiene PRIORIDAD sobre la búsqueda del Descubridor)
Tienes acceso a la bibliografía oficial de cada asignatura de Duoc UC.
PASO 1 — DISTINGUIR LA INTENCIÓN: cuando el estudiante pida material relacionado con un ramo/asignatura (ej. "necesito un libro de ingeniería de software"), primero pregúntale brevemente qué prefiere, con dos opciones claras:
1. La bibliografía oficial de ese ramo (los textos que pide la asignatura), o
2. Una búsqueda general de material sobre ese tema en la colección.
No interrogues más de la cuenta: una sola pregunta con esas dos opciones.
PASO 2 — SEGÚN LA RESPUESTA:
- Si quiere la BIBLIOGRAFÍA DEL RAMO: usarás los datos que te entregue el sistema (NO el Descubridor).
   · Si recibes "DATOS PARA ESTA RESPUESTA" con una bibliografía: confirma la asignatura y presenta los recursos en lista limpia. Para CADA recurso: el TITULO en **negrita** y debajo [Acceder](ACCESO_DIGITAL) con la URL del campo ACCESO_DIGITAL TAL CUAL viene (ya está codificada; no la modifiques ni le agregues espacios). NUNCA muestres "Ver en catálogo" ni la URL del catálogo Symphony; solo el enlace de acceso digital. No inventes títulos ni enlaces. Si la lista es larga, ofrece acotar por tema.
   · DISPONIBILIDAD FÍSICA: el campo COPIAS_FISICAS_EN NO lo muestres de entrada. Primero presenta los libros digitales con su acceso; LUEGO pregunta de qué sede es el estudiante y solo entonces dile si hay copias físicas en su sede. Orden: primero lo digital, después lo físico según su sede.
   · Si recibes "VARIAS ASIGNATURAS COINCIDEN": muestra la lista completa de una vez (asignatura, código y carrera) y pide que elija una o te dé el código. NO preguntes la carrera por separado, NO busques en el Descubridor, NO elijas tú.
   · Si NO recibes ningún bloque de datos: pide el código del ramo (ej. MAT2110) o el nombre exacto; no la inventes.
- Si quiere una BÚSQUEDA GENERAL del tema: usa el Descubridor con la búsqueda construida, como siempre.
Cuando el estudiante ya da el código directo (ej. "bibliografía de MAT2110"), salta el Paso 1 y entrega la bibliografía de inmediato.

# ALCANCE
Solo recursos/servicios de Bibliotecas Duoc UC + metodología académica según guías oficiales. Fuera de eso, redirige con amabilidad ofreciendo lo que sí puedes hacer. Nunca inventes recursos ni servicios. Recuerda siempre la FILOSOFÍA: eres un guía que enseña y entrega los recursos del portal, nunca quien hace el trabajo; así eres un apoyo legítimo respaldado por docentes e institución."""

# --- FAQ cache: respuestas instantaneas y SIN costo de API ---------------
# Preguntas muy frecuentes con respuesta estable. Se sirven directo desde el
# backend (coste cero, latencia minima) SOLO cuando es el primer turno y la
# pregunta coincide claramente con un patron. Ante cualquier matiz, NO se usa
# el cache y la consulta sigue su flujo normal hacia el modelo.

FAQ = [
    {
        "patrones": [r"\bmulta\b", r"cu[aá]nto.*atraso", r"cu[aá]nto.*(debo|pagar).*libro"],
        "respuesta": (
            "La multa por atraso es de **$1.000 por \u00edtem**, y se va acumulando de mil "
            "en mil cada semana.\n\nPara revisar tu situaci\u00f3n (qu\u00e9 tienes pendiente y el "
            "monto), entra a [Mi cuenta]"
            "(https://duchi.ent.sirsidynix.net/client/es_CL/default/search/patronlogin/"
            "https:$002f$002fduchi.ent.sirsidynix.net$002fclient$002fdefault$002fsearch$002faccount$003f) "
            "con tus credenciales Duoc. Si necesitas ayuda, escr\u00edbele al staff de la biblioteca por el "
            "[Formulario de consulta](https://bibliotecas.duoc.cl/consultanos)."
        ),
    },
    {
        "patrones": [r"c[oó]mo.*renov", r"renovar.*pr[eé]stamo", r"renovar.*libro", r"renuevo"],
        "respuesta": (
            "Puedes renovar tus pr\u00e9stamos desde [Mi cuenta]"
            "(https://duchi.ent.sirsidynix.net/client/es_CL/default/search/patronlogin/"
            "https:$002f$002fduchi.ent.sirsidynix.net$002fclient$002fdefault$002fsearch$002faccount$003f), "
            "iniciando sesi\u00f3n con tus credenciales Duoc.\n\nAqu\u00ed tienes el videotutorial paso a paso: "
            "[C\u00f3mo renovar](https://www.youtube.com/watch?v=ncsY9xEhFPo). Si el sistema no te deja, "
            "puede haber una multa o un pr\u00e9stamo vencido."
        ),
    },
    {
        "patrones": [r"reservar.*sala", r"reserva.*sala", r"sala de estudio", r"lentes vr"],
        "respuesta": (
            "Puedes reservar una sala de estudio o los lentes VR en la "
            "[p\u00e1gina de reservas](https://bibliotecas.duoc.cl/reserva-sala).\n\nAqu\u00ed te muestran "
            "c\u00f3mo hacerlo: [Videotutorial de reserva de salas]"
            "(https://www.youtube.com/watch?v=SxU_2BFHVI4)."
        ),
    },
]


def buscar_faq(historial):
    """Devuelve la respuesta de FAQ si aplica, o None. Solo en el primer turno
    (sin historial previo) para no romper conversaciones con contexto."""
    turnos_usuario = sum(1 for x in historial if x["role"] == "user")
    if turnos_usuario != 1:
        return None
    texto = historial[-1]["content"].lower()
    # Si la pregunta es larga o compleja, mejor que responda el modelo
    if len(texto.split()) > 16:
        return None
    if any(s in texto for s in SENALES_COMPLEJAS):
        return None
    for item in FAQ:
        if any(re.search(p, texto) for p in item["patrones"]):
            return item["respuesta"]
    return None


def _sse_text(texto):
    """Emite un texto como si fuera un stream SSE de Anthropic, para que el
    frontend lo procese igual que una respuesta normal."""
    evt_start = {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "text", "text": ""}}
    evt_delta = {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": texto}}
    evt_stop = {"type": "content_block_stop", "index": 0}
    for evt in (evt_start, evt_delta, evt_stop):
        yield "data: " + json.dumps(evt) + "\n\n"


def _client_ip(request):
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_ok(ip):
    ahora = time.time()
    bucket = _rate_buckets[ip]
    while bucket and ahora - bucket[0] > RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= RATE_MAX:
        return False
    bucket.append(ahora)
    return True


app = FastAPI(title="Chatbot Bibliotecas Duoc UC", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class Mensaje(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Mensaje]


_PALABRAS_BIBLIO = ("bibliografia", "bibliografía", "bibliograf", "libros de",
                    "libros para", "que libros", "qué libros", "lectura",
                    "material de", "textos de", "asignatura", "ramo")


_SEDES = ("alameda", "alonso de ovalle", "alonso ovalle", "antonio varas", "arauco",
          "concepcion", "concepción", "maipu", "maipú", "melipilla", "nacimiento",
          "plaza norte", "plaza oeste", "plaza vespucio", "puente alto", "puerto montt",
          "san bernardo", "san carlos", "san joaquin", "san joaquín", "valparaiso",
          "valparaíso", "villarrica", "viña del mar", "vina del mar")


def _consulta_disponibilidad_sede(historial):
    """Si el estudiante menciona una sede y en el historial reciente hay un libro
    con catkey, consulta Symphony EN VIVO y devuelve el bloque de disponibilidad
    por sede. Es la consulta puntual 'a pedido' (una sola llamada, rapida)."""
    ultimo = ""
    for x in reversed(historial):
        if x["role"] == "user":
            ultimo = x["content"].lower()
            break
    # detectar sede mencionada
    sede = next((s for s in _SEDES if s in ultimo), None)
    if not sede:
        return None
    # buscar el catkey mas reciente mencionado en la conversacion (en mensajes del asistente)
    catkey = None
    for x in reversed(historial):
        m = re.search(r"CATKEY[:\s]+(\d+)", x["content"])
        if m:
            catkey = m.group(1)
            break
        m2 = ilsws.catkey_desde_url(x["content"])
        if m2:
            catkey = m2
            break
    if not catkey:
        return None
    datos = ilsws.consultar_titulo(catkey)
    if not datos:
        return None
    disponible, sedes = ilsws.disponible_en_sede(datos, sede)
    titulo = datos.get("titulo") or "el libro"
    if disponible:
        return (f"DISPONIBILIDAD EN VIVO: \"{titulo}\" SÍ tiene copias disponibles en la sede que "
                f"mencionó el estudiante. Confírmaselo con entusiasmo y dile que puede retirarlo allí. "
                f"Sedes con copias ahora: {', '.join(sedes)}.")
    elif sedes:
        return (f"DISPONIBILIDAD EN VIVO: \"{titulo}\" en este momento NO figura con copias en la sede del "
                f"estudiante. Sí hay copias en: {', '.join(sedes)}. Ofrécele esas sedes o la versión digital "
                f"si existe. Sé amable y honesto.")
    else:
        return (f"DISPONIBILIDAD EN VIVO: \"{titulo}\" no figura con copias físicas disponibles en este "
                f"momento en ninguna sede. Ofrece la versión digital si existe, o sugiere consultar con el "
                f"staff de la biblioteca.")


def _contexto_bibliografia(historial):
    """Si el ultimo mensaje pide bibliografia y nombra una asignatura, devuelve
    el bloque de contexto a inyectar; si no, None. Tambien maneja la consulta
    puntual de disponibilidad por sede (a pedido)."""
    ultimo = ""
    for x in reversed(historial):
        if x["role"] == "user":
            ultimo = x["content"]
            break
    tl = ultimo.lower()
    # 1) consulta puntual de disponibilidad por sede (tiene prioridad)
    disp = _consulta_disponibilidad_sede(historial)
    if disp:
        return disp
    # 2) peticion de bibliografia de asignatura
    pide = any(p in tl for p in _PALABRAS_BIBLIO) or _COD_RE.search(ultimo.upper())
    if not pide:
        return None
    estado, datos = buscar_bibliografia(ultimo)
    if estado == "unica":
        return formato_bibliografia(datos)
    if estado == "varias":
        return formato_varias(datos)
    return None


def _build_payload(historial, stream):
    modelo = elegir_modelo(historial)
    system_blocks = [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]
    # Inyectar bibliografia de la asignatura pedida (fuera del bloque cacheado,
    # porque cambia en cada consulta)
    ctx = _contexto_bibliografia(historial)
    if ctx:
        system_blocks.append({
            "type": "text",
            "text": ("DATOS PARA ESTA RESPUESTA (bibliografia oficial recuperada del "
                     "sistema de la biblioteca; entregala al estudiante con sus enlaces, "
                     "presentandola ordenada y con formato limpio en negrita, sin inventar "
                     "titulos ni enlaces que no esten aqui):\n" + ctx)
        })
    return {
        "model": modelo,
        "max_tokens": MAX_TOKENS,
        "system": system_blocks,
        "messages": historial,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "stream": stream,
    }


def _headers():
    return {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _sanitize(messages):
    historial = [
        {"role": m.role, "content": m.content[:MAX_CHARS_PER_MSG]}
        for m in messages
        if m.role in ("user", "assistant") and m.content.strip()
    ][-MAX_HISTORY:]
    if not historial or historial[-1]["role"] != "user":
        raise HTTPException(400, "El ultimo mensaje debe ser del usuario")
    return historial


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    """Respuesta en streaming (SSE). Flujo de ahorro:
    1) Rate limit por IP (anti-abuso).
    2) FAQ cache: preguntas frecuentes se responden sin llamar a la API.
    3) Enrutamiento Haiku/Sonnet + prompt caching para el resto."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY no configurada en el servidor")

    ip = _client_ip(request)
    if not _rate_ok(ip):
        raise HTTPException(429, "Demasiadas consultas seguidas. Espera unos segundos.")

    historial = _sanitize(req.messages)
    _stats["total"] += 1

    # 2) FAQ cache (coste cero)
    faq = buscar_faq(historial)
    if faq is not None:
        _stats["faq_hits"] += 1
        log.info("FAQ hit ip=%s", ip)
        return StreamingResponse(
            _sse_text(faq),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 3) Llamada al modelo con enrutamiento + caching
    payload = _build_payload(historial, stream=True)
    modelo = payload["model"]
    _stats["haiku" if "haiku" in modelo else "sonnet"] += 1
    log.info("API call ip=%s modelo=%s", ip, modelo.split("-")[1] if "-" in modelo else modelo)

    async def event_stream():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", ANTHROPIC_URL, json=payload, headers=_headers()) as r:
                if r.status_code != 200:
                    yield "data: " + json.dumps({"type": "error"}) + "\n\n"
                    return
                async for line in r.aiter_lines():
                    if line:
                        yield line + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/stats")
def stats():
    """Metricas en memoria para monitorear el ahorro (se reinician al reiniciar
    el server). Util para ver la proporcion FAQ/Haiku/Sonnet del trafico real."""
    t = _stats["total"] or 1
    return {
        **_stats,
        "faq_pct": round(_stats["faq_hits"] * 100 / t, 1),
        "haiku_pct": round(_stats["haiku"] * 100 / t, 1),
        "sonnet_pct": round(_stats["sonnet"] * 100 / t, 1),
    }


@app.post("/api/chat-sync")
async def chat_sync(req: ChatRequest):
    """Version sin streaming; devuelve texto y metricas de uso/cache."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY no configurada en el servidor")
    historial = _sanitize(req.messages)
    payload = _build_payload(historial, stream=False)

    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(ANTHROPIC_URL, json=payload, headers=_headers())
    if r.status_code != 200:
        raise HTTPException(502, f"Error de la API de Anthropic: {r.status_code}")

    data = r.json()
    texto = "\n".join(
        b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
    ).strip()
    usage = data.get("usage", {})
    return {
        "reply": texto or "Lo siento, tuve un problema al responder. Puedes intentarlo de nuevo?",
        "model_used": _build_payload(_sanitize(req.messages), False)["model"],
        "usage": {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        },
    }
