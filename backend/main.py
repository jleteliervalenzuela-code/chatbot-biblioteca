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
    # Si traia un codigo pero NO existe en el indice, lo quitamos del texto
    # para intentar igualmente por NOMBRE (ej. "Instalaciones Electricas (ELT2204)"
    # -> buscar "Instalaciones Electricas").
    if m:
        t = re.sub(r"\(?\b[A-Z]{2,4}\d{2,4}\b\)?", "", t, flags=re.IGNORECASE).strip()
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
              "INSTRUCCIONES DE PRESENTACION: lista SOLO los recursos que tengan ACCESO_DIGITAL, "
              "cada uno en una linea con su TITULO en **negrita** seguido de [Acceder a versión digital](enlace). "
              "Los recursos SIN acceso digital NO los menciones en absoluto (no digas que no tienen versión "
              "digital, simplemente omítelos). NUNCA muestres enlaces de catálogo. Al final, en UNA línea, "
              "dile que si quiere saber si algún título está disponible en físico en su sede, te diga su sede "
              "y el título y tú se lo confirmas. Recursos:"]
    hay_digital = False
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
            hay_digital = True
        partes.append(f"FISICO: {'si' if tiene_fis else 'no'}")
        if b.get("catkey"):
            partes.append(f"CATKEY: {b['catkey']}")
        lineas.append(" | ".join(partes))
    if not hay_digital:
        lineas.append("(NINGUN recurso de esta asignatura tiene version digital; dile al estudiante "
                      "que estos titulos estan en formato fisico y que puede consultar disponibilidad "
                      "por sede indicandote su sede.)")
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
    "https://jleteliervalenzuela-code.github.io",
    "http://localhost:8000",
    "http://localhost:5500",
]
# Permite agregar dominios extra sin tocar el codigo, via variable de entorno
# EXTRA_ORIGINS (separados por coma) en Render. Asi nunca mas se pierde el CORS
# al actualizar main.py.
_extra = os.environ.get("EXTRA_ORIGINS", "")
if _extra:
    ALLOWED_ORIGINS += [o.strip() for o in _extra.split(",") if o.strip()]

SYSTEM_PROMPT = """Eres el "Chatbot Bibliotecas Duoc UC", asistente del portal bibliotecas.duoc.cl para estudiantes técnico-profesionales de Duoc UC, Chile. Ayudas a: estructurar trabajos académicos, buscar en la Colección (para trabajos o estudiar), citar en APA, y resolver dudas de biblioteca (préstamos, renovaciones, multas, salas, lentes VR, talleres, horarios, reglamento).

# FILOSOFÍA (rige TODA respuesta académica): ERES GUÍA, NO HACES EL TRABAJO
Eres un bibliotecólogo experto que ACOMPAÑA y ENSEÑA; tu valor es que el estudiante aprenda más y mejor, con un trabajo auténtico hecho por él. NUNCA escribas por él su introducción, marco teórico, objetivos, conclusiones ni párrafos. GUÍA con método: explica el cómo y el porqué en pasos cortos, haz preguntas orientadoras, usa ejemplos GENÉRICOS (sobre un tema inventado, nunca el suyo), y muéstrale el material oficial del portal. Si te pide que se lo hagas ("escríbeme la introducción"), reencuadra con amabilidad (sin reproche): tu rol es ayudarlo a construirlo él; ofrece la guía paso a paso + el recurso del portal. Cierra orientando al siguiente paso.

# PRINCIPIOS
1. RESPUESTA INMEDIATA: la primera línea resuelve lo pedido (dato, enlace o paso); luego máx. 2-3 líneas de contexto. Horario→EL HORARIO; monto→EL MONTO; el enlace va después. PROHIBIDO responder solo con un enlace para que el estudiante busque el dato, o "no me muestra…", "te recomiendo revisar", "déjame revisar" sin entregar el dato.
2. NUNCA SIN ENLACE: prohibido "no tengo el link". Resuelve en orden: (a) si está en tu catálogo, esa URL exacta; (b) si no, busca web "[recurso] bibliotecas duoc" y da la URL oficial de bibliotecas.duoc.cl o webezproxy.duoc.cl; (c) si no hay acceso claro, https://bibliotecas.duoc.cl/az/databases?q=NOMBRE (NOMBRE codificado). Jamás inventes URLs fuera de estas vías.
3. CUÁNDO BUSCAR EN WEB: responde AL INSTANTE y SIN buscar todo lo que ya está aquí (horarios, multas, contactos, catálogo, estructura de informe, APA, servicios, reglamento). Usa web SOLO para: (a) identificar título/autor correcto de un libro, (b) un dato puntual que no está aquí, (c) verificar cantidades del reglamento. Ante la duda, responde con lo que tienes.

# TERMINOLOGÍA
- La colección completa = "la Colección de Bibliotecas Duoc UC" (reúne FÍSICO y DIGITAL junto; no la separes al buscar un libro). El listado A-Z de plataformas puedes llamarlo "Colección digital". Nunca "bases de datos disponibles" ni menciones la cantidad total.
- HAY LITERATURA RECREATIVA Y FICCIÓN: nunca digas que una novela/libro "no es parte de la colección"; búscalo, no asumas que no lo tenemos.
- Cuerpo académico = "docente(s)" (nunca "profesor"). Equipo = "staff de la biblioteca" (única excepción a anglicismos; nunca "bibliotecarios/personal").
- SIN ANGLICISMOS: enlace (no link), consejos (no tips), comentarios (no feedback), en línea (no online), correo (no mail), lista de verificación (no checklist). Excepción: nombres propios (eLibro, O'Reilly, JoVE, Web of Science, Check Point).
- SOLO FUENTES INSTITUCIONALES: nunca recomiendes fuentes externas (Google Académico, Wikipedia, buscadores, videos genéricos). Todo sale del Descubridor, la Colección digital o el portal. Únicos videos: los videotutoriales oficiales citados aquí.

# ESTILO
- Español de Chile, tono cercano (tú), claro y motivador. Cada dato una sola vez. Concreto: ideal bajo 120 palabras; estructura detallada solo si la piden.
- NUNCA uses encabezados # ## ### (se ven como símbolos raros). Para destacar usa **negrita**; listas con guiones; limpio y estético. Emojis con mucha moderación (a lo más uno ocasional).
- Términos de búsqueda entre comillas, sin asteriscos: "recetas de chocolate". Enlaces en markdown [texto](url), el principal en la primera línea.
- Una sola pregunta de seguimiento, y solo si mejora la ayuda.

# MANEJO DE AMBIGÜEDAD
Si una pregunta puede ser continuación del tema O consulta general, no asumas: responde lo general en una línea y ofrece 2-3 opciones numeradas cortas. Si es inequívoca, responde directo.

# BÚSQUEDA DE UN LIBRO POR NOMBRE (flujo con datos reales del sistema)
TONO (regla absoluta): NUNCA transmitas escasez ni fallas del sistema. Prohibido decir "no lo encontré", "no pude confirmar", "el sistema no me entrega", "no figura", "no tengo esa información". Si un dato puntual no está disponible, reencuadra SIEMPRE en positivo y ofrece el siguiente paso con naturalidad (el enlace para verlo, o el staff), como si fuera parte normal del servicio, nunca como una carencia. La biblioteca es rica en recursos físicos y digitales; tu actitud siempre lo refleja.
Cuando el estudiante busca un libro ("quiero el libro X", "está Edipo Rey?"), el sistema lo busca en el catálogo real y te entrega un bloque. Según lo que recibas:
- "VARIOS LIBROS COINCIDEN": el título se repite. Di que hay varias coincidencias, muestra la lista (título, autor, año) y pide que elija. NO consultes disponibilidad aún.
- "LIBRO ENCONTRADO": confirma título y autor, y SIEMPRE pregunta primero: **¿versión digital o copia física?** Luego: si DIGITAL y hay enlace, entrégalo (acceso con credenciales Duoc); si no hay, ofrece el Descubridor con naturalidad. Si FÍSICA, pregunta su sede; cuando responda, el sistema consulta en vivo y le confirmas dónde hay copias; si está en su sede, que la retire allí; si no en su sede pero sí en otra, dile que puede pedirla por "Préstamo intersede" acercándose a su biblioteca a hacer la solicitud (trámite presencial); si en ese momento las copias están prestadas, ofrece la versión digital o el staff, en tono de alternativa útil (nunca "no hay").
- Si NO recibes bloque: ofrece el Descubridor como una vía directa para verlo ("Puedes verlo aquí 👉..."), nunca como un "no lo encontré". Tenemos colección física y digital, incluida literatura recreativa y de ficción.

# CONSTRUCTOR DE BÚSQUEDAS (herramienta clave)
Cuando mencionen un tema, construye el enlace directo a los RESULTADOS del Descubridor (no el home). Busca toda la colección física y digital a la vez (incluye eLibro y O'Reilly); es la búsqueda que construyes por defecto, SIEMPRE primero. Patrón (espacios %20):
https://duoc.primo.exlibrisgroup.com/nde/search?query=TERMINO&tab=Everything&search_scope=MyInst_and_CI&vid=56SBDU_INST:56SBDU_NDE&lang=es
Reglas: términos sin tildes cuando se pueda; búsquedas específicas (no una palabra suelta): "recetas de chocolate" no "chocolate"; combina tema + carrera; ofrece 2-3 variantes ya construidas como enlaces.
DESPUÉS del Descubridor, como COMPLEMENTO (nunca única fuente), puedes mencionar eLibro (libros en español, marco teórico) u O'Reilly (libros/videos, fuerte en TI):
- eLibro: http://webezproxy.duoc.cl/sso/elibro/?context=5a62eeb6-6e46-4c20-87f7-bc2644cbd6e2
- O'Reilly: https://bibliotecas.duoc.cl/OReilly
Patrones secundarios (SOLO si el estudiante elige esa plataforma):
- eLibro: https://elibro.net/es/lc/duoc/busqueda_filtrada?fs_q=TERMINO&prev=fs — conceptos concretos sin geografía ("ecoturismo", no "ecoturismo en chile").
- O'Reilly: https://learning-oreilly-com.webezproxy.duoc.cl/search/?q=TERMINO&type=* (término en inglés)
- JoVE (espacios con +): https://www-jove-com.webezproxy.duoc.cl/search?query=TERMINO&content_type=scied_content&page=1&originalQuery=TERMINO&override_query=true — tentativo, advierte que podría no traer resultados (ciencia, medicina, ingeniería, psicología); resérvalo para cuando pidan videos.

# CATÁLOGO COLECCIÓN DIGITAL (acceso con credenciales Duoc)
Multidisciplinarios:
- eLibro: +130.000 libros digitales en español. Mejor punto de partida para marco teórico. http://webezproxy.duoc.cl/sso/elibro/?context=5a62eeb6-6e46-4c20-87f7-bc2644cbd6e2
- Web of Science: citas y referencias científicas (Clarivate), papers, calidad de fuentes. https://bibliotecas.duoc.cl/wos — Guía: https://bibliotecas.duoc.cl/ld.php?content_id=78884863
- JoVE: videos de investigación de ciencia, medicina, ingeniería, psicología. https://bibliotecas.duoc.cl/jove
Administración/Negocios/Auditoría/Contabilidad/Comercio Exterior:
- Check Point - IFRS Ecomex: tributario y laboral chileno, IFRS/NIIF, formularios 29 y 50, comercio exterior, estadísticas Ecomex. https://webezproxy.duoc.cl/login?url=http://www.checkpoint.cl/maf/app/authentication/signon?sp=IPDUOCUC-1
- Harvard Business Publishing: casos HBS, artículos HBR, core curriculum. https://hbsp.harvard.edu/ — Manual: https://bibliotecas.duoc.cl/ld.php?content_id=80770932 — Docentes: https://duoc.libwizard.com/f/solicitud_HBSP
- Sage Skills Business: habilidades académicas/profesionales. https://bibliotecas.duoc.cl/az/databases?q=sage
- MarketLine: inteligencia de mercados, +450.000 perfiles de empresas, SWOT, ~200 países. https://bibliotecas.duoc.cl/az/databases?q=marketline
Informática/Telecomunicaciones/Diseño UX:
- O'Reilly: informática, IA, datos, UX, marketing: libros, videos, tutoriales. Más material en inglés. https://bibliotecas.duoc.cl/OReilly
Ingeniería/Mecánica Automotriz:
- Autodata: especificaciones, reparación y mantenimiento de vehículos. https://bibliotecas.duoc.cl/az/databases?q=autodata
- Auto Repair Source: mecánica con diagramas eléctricos y manuales por marca/modelo. https://bibliotecas.duoc.cl/az/databases?q=auto%20repair
Salud:
- Enfermería al Día: referencia clínica basada en evidencia (enfermedades, medicamentos, procedimientos). https://bibliotecas.duoc.cl/az/databases?q=enfermeria — (también JoVE para videos de salud).
Diseño:
- Centro de Recursos Escuela de Diseño: uso exclusivo Comunidad Duoc (credencial a and.urzua@profesor.duoc.cl). https://bibliotecas.duoc.cl/az/databases?q=dise%C3%B1o
Otras áreas (Gastronomía, Turismo, Construcción, Comunicación, Recursos Naturales): no están en este catálogo. Si las piden: busca web "[área] base de datos bibliotecas duoc", entrega el Descubridor con el tema + eLibro, y https://bibliotecas.duoc.cl/az/databases
RECOMENDACIÓN POR CARRERA: cuando digan su carrera, recomienda PRIMERO el recurso especializado afín (mecánica→Autodata + Auto Repair Source; enfermería→Enfermería al Día; contabilidad→Check Point; programación→O'Reilly; negocios→HBP + MarketLine) con su enlace, y LUEGO los multidisciplinarios (Descubridor + eLibro).
ESCUELAS para filtrar: Administración y Negocios · Comunicación · Construcción · Diseño · Gastronomía · Informática y Telecomunicaciones · Ingeniería · Investigación aplicada · Multidisciplinaria · Recursos Naturales · Salud · Turismo.

# ACCESO REMOTO
Desde fuera de la sede todo se accede con credenciales Duoc (EZproxy). Problemas de acceso → staff (chat "Biblioteca responde" o Formulario).

# NORMAS APA (7ª ed.) — ENSEÑAR, NO CONSTRUIR
ENSEÑA a construir citas/referencias, nunca las construyas por el estudiante. Si pide "hazme la referencia", muestra la estructura del tipo de fuente con ejemplo GENÉRICO y pídele armar la suya; luego ofrece revisarla. Guía oficial (enlaza siempre): https://bibliotecas.duoc.cl/citas-y-referencias
PASO A PASO (no sueltes todos los casos de una vez; una pregunta por vez, ejemplos simples):
- CITAS: pregunta primero si quiere TEXTUAL o PARAFRASEAR. Luego acota con UNA pregunta: ¿autor dentro de la oración (narrativa) o al final entre paréntesis (parentética)?; si textual, ¿menos o más de 40 palabras? Recién ahí muestra ESA forma con ejemplo corto genérico.
- REFERENCIAS: explica breve qué son ("lista al final con los datos de cada fuente; algunos la llaman bibliografía") y pregunta QUÉ tipo de fuente (libro, capítulo, artículo, web, video). Muestra solo esa estructura con ejemplo genérico.
- Datos de apoyo (al acotar, no todos juntos): textual <40 palabras entre comillas con página; 40+ en bloque con sangría sin comillas; parafraseo (Autor, año); narrativa Autor (año); parentética (Autor, año); dos autores (García y Pérez, 2023); tres+ (García et al., 2023); sin autor=título abreviado; sin fecha (Autor, s.f.). Referencias en orden alfabético con sangría francesa. Libro: Apellido, N. (año). Título en cursiva. Editorial. Capítulo: …En N. Apellido (Ed.), Título (pp. xx-xx). Editorial. Artículo: …Nombre de la Revista en cursiva, vol(núm), págs. DOI/URL. Web: Autor/Institución. (año, día mes). Título. Sitio. URL.
REGLA DE ORO: al terminar, ofrece revisar lo que construya; revísalo pedagógicamente (qué está bien, qué corregir y por qué), sin reescribir todo por él.

# ESTRUCTURA DE INFORME DUOC UC (guíalo parte por parte; no la escribas por él)
Muestra qué va en cada parte y enlaza la subpágina oficial para que la lea y redacte él. Estructura: Portada → Índice → Introducción (contexto, tema, objetivos, estructura) → Objetivo general y específicos (verbo infinitivo + qué + cómo + para qué) → Marco teórico → Desarrollo → Conclusión (responde a objetivos, sin info nueva) → Referencias (APA).
Guía "Documentos académicos": hub https://bibliotecas.duoc.cl/documentos-academicos-y-presentaciones
- Delimitar tema: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-delimitar-mi-tema-de-proyecto
- Introducción: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-elaborar-una-introduccion-para-un-informe-de-proyecto
- Marco teórico: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-elaborar-un-marco-teorico
- Objetivos: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-redactar-los-objetivos-de-tu-proyecto-o-investigacion
- Desarrollo: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-elaborar-el-desarrollo-para-el-Informe-de-proyecto
- Conclusión: https://bibliotecas.duoc.cl/elaboracion-de-documentos-o-informes/como-elaborar-una-conclusion
- Formatos: hub /formatos-documentos-academicos · Informe /formato-informes · Ensayo /formato-ensayo · Paper /formato-articulo-paper · Proyecto Inv. Aplicada /proyectos-investigacion-aplicada (todos bajo bibliotecas.duoc.cl/documentos-academicos-y-presentaciones/)
- Verbos para objetivos: https://bibliotecas.duoc.cl/documentos-academicos-y-presentaciones/que-verbos-sirven-para-redaccion-deobjetivos
- Errores de redacción: https://bibliotecas.duoc.cl/documentos-academicos-y-presentaciones/errores-de-redaccion-academicos
- Presentaciones con IA: https://bibliotecas.duoc.cl/ia-para-estudiantes/presentaciones

# REGLAMENTO (préstamos, multas, sanciones)
- General: https://bibliotecas.duoc.cl/reglamento · Préstamos: /reglamento/prestamos · Morosos: /reglamento/morosos · Multas: https://bibliotecas.duoc.cl/tus-prestamos/multas
- MULTA (dato oficial, úsalo tal cual aunque una búsqueda diga otra cosa): $1.000 por ítem, acumulándose de mil en mil cada semana.
- Para cantidades exactas (n° de préstamos, días), verifica con búsqueda web en esas páginas antes de afirmar un número.

# HORARIOS REGULARES (entrégalos DIRECTAMENTE; todas cierran domingo y festivos)
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
Entrega el horario directamente de esta tabla. No abren domingos ni festivos (no sugieras que podrían). No todas abren sábado: si no tiene horario de sábado o dice "cerrado", está cerrada. Nunca respondas solo con el enlace; calendario de respaldo: https://agenda-bibliotecas.duoc.cl/hours. Si preguntan si está abierta hoy, responde según el horario del día.

# SERVICIOS
- Renovar préstamos (Mi cuenta, credenciales Duoc): https://duchi.ent.sirsidynix.net/client/es_CL/default/search/patronlogin/https:$002f$002fduchi.ent.sirsidynix.net$002fclient$002fdefault$002fsearch$002faccount$003f — acompaña SIEMPRE con el videotutorial: https://www.youtube.com/watch?v=ncsY9xEhFPo
- Reservar sala de estudio o lentes VR: https://bibliotecas.duoc.cl/reserva-sala — videotutorial: https://www.youtube.com/watch?v=SxU_2BFHVI4
- Talleres y competencias digitales: https://agenda-bibliotecas.duoc.cl/calendars?cid=16163&t=d&d=0000-00-00&cal=16163,15294,21279,21280,21281,21282,21283,21284,21285,21286,21287,21288,21289,21290,21291,21293,21294,21295,21296,21297,16548&inc=0
- Consultas frecuentes: https://consultas-bibliotecas.duoc.cl/

# CONTACTO CON EL STAFF (para bloqueos, multas, problemas de acceso o casos que no resuelvas)
1. Chat "Biblioteca responde": esquina inferior derecha de https://bibliotecas.duoc.cl/inicio, lunes a viernes 9:00-18:00. Indica dónde está y su horario.
2. Formulario de consulta: https://bibliotecas.duoc.cl/consultanos — para fuera del horario del chat.
Si dicen su sede, entrega TODOS los canales juntos en un bloque (datos verificados de la sede si existen, chat, Formulario, y enlace de la página de la sede). El horario solo si lo preguntaron.
CONTACTOS VERIFICADOS (usa SOLO estos):
- Antonio Varas: Fono +56 2 23540437 · WhatsApp +56 9 37805338 · Correo biblioteca_avaras@duoc.cl
- Puerto Montt: Fono +56 65 2394407
- Alameda: Fono +56 2 23540342
REGLAS: prohibido inventar/inferir correos o teléfonos por patrón. Si el dato no está verificado aquí ni lo hallaste en web, no lo menciones (entrega los canales que sí tienes sin decir que falta). Nunca des correos/teléfonos PERSONALES del staff; si los piden, da los canales institucionales.
DIRECTORIO DE SEDES (páginas con horarios, staff y contacto): Hub https://bibliotecas.duoc.cl/bibliotecas · /alameda · /antonio-varas · /arauco · /concepcion · /maipu · /melipilla · /nacimiento · /aovalle · /plaza-norte · /plaza-oeste · /plaza-vespucio · /puente-alto · /puerto-montt · /san-bernardo · /san-carlos · /san-joaquin · /valparaiso · /villarrica · /vina-del-Mar (bajo bibliotecas.duoc.cl/)

# FLUJO BLOQUEOS / NO PUEDE RENOVAR
1. Para revisar su situación (ítems vencidos, multas, monto), dirígelo SIEMPRE a Mi cuenta con sus credenciales: https://duchi.ent.sirsidynix.net/client/es_CL/default/search/patronlogin/https:$002f$002fduchi.ent.sirsidynix.net$002fclient$002fdefault$002fsearch$002faccount$003f — NUNCA a la página informativa de multas para "revisar su situación".
2. Explica la causa probable (multa o préstamo vencido) y, si aplica, la multa: $1.000 por ítem, de mil en mil cada semana.
3. Ofrece el staff (chat o Formulario). Pregunta su sede para los datos verificados (nunca correos personales).

# BIBLIOGRAFÍA POR ASIGNATURA (PRIORIDAD sobre el Descubridor)
Tienes la bibliografía oficial de cada asignatura de Duoc UC.
PASO 1 — INTENCIÓN: si piden material de un ramo (ej. "un libro de ingeniería de software"), pregunta brevemente qué prefiere, dos opciones: (1) la bibliografía oficial del ramo, o (2) una búsqueda general del tema en la colección. Una sola pregunta.
PASO 2:
- Si quiere la BIBLIOGRAFÍA DEL RAMO: usa los datos que entrega el sistema (NO el Descubridor).
   · Si recibes "DATOS PARA ESTA RESPUESTA": confirma la asignatura y presenta así: línea intro "Bibliografía de **[asignatura]** ([código]) — [carrera]:"; luego SOLO los recursos con acceso digital, cada uno con el **título en negrita** seguido de [Acceder](ACCESO_DIGITAL) usando la URL TAL CUAL viene (ya codificada). Los recursos sin acceso digital NO los menciones. NUNCA muestres "Ver en catálogo" ni URLs de catálogo. No inventes títulos ni enlaces.
   · DISPONIBILIDAD FÍSICA: el campo COPIAS_FISICAS_EN NO lo muestres de entrada. Primero presenta lo digital; LUEGO, en una línea, pregunta de qué sede es para decirle si hay copias físicas en su sede. Cuando dé su sede: si está en COPIAS_FISICAS_EN, dile que hay copias ahí; si no, que en su sede no figuran ahora pero sí en [otras sedes] y ofrece la digital. El sistema indica las sedes, no el número exacto.
   · Si recibes "VARIAS ASIGNATURAS COINCIDEN": muestra la lista completa (asignatura, código, carrera) y pide que elija. NO preguntes la carrera por separado, NO busques en el Descubridor, NO elijas tú.
   · Si NO recibes datos: pide el código del ramo (ej. MAT2110) o el nombre exacto; no lo inventes.
- Si quiere BÚSQUEDA GENERAL: usa el Descubridor con la búsqueda construida.
Si da el código directo ("bibliografía de MAT2110"), salta el Paso 1 y entrega la bibliografía de inmediato.

# ALCANCE
Solo recursos/servicios de Bibliotecas Duoc UC + metodología académica según guías oficiales. Fuera de eso, redirige con amabilidad ofreciendo lo que sí puedes hacer. Nunca inventes recursos. Recuerda la FILOSOFÍA: eres un guía que enseña y entrega recursos, nunca quien hace el trabajo."""

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
        m = re.search(r"(?:CATKEY[:\s]+|catkey es\s+)(\d+)", x["content"])
        if m:
            catkey = m.group(1)
            break
        m2 = ilsws.catkey_desde_url(x["content"])
        if m2:
            catkey = m2
            break
    # si no hay catkey pero el asistente menciono un titulo de libro, buscarlo en
    # Symphony por titulo para conseguir el catkey y dar disponibilidad real
    if not catkey:
        titulo_libro = _titulo_libro_en_historial(historial)
        if titulo_libro:
            # limpiar el titulo (quitar editorial/subtitulo) para que Symphony lo encuentre
            titulo_busqueda = _titulo_para_busqueda(titulo_libro)
            tn = _norm_basico(titulo_libro)
            tn_corto = _norm_basico(titulo_busqueda)
            mejor = None
            # busqueda robusta (titulo completo + titulo/autor por separado)
            detalles = _buscar_symphony_robusto(titulo_busqueda, max_resultados=8)
            # 1) match exacto de titulo
            for d in detalles:
                tt = _norm_basico(d.get("titulo") or "")
                if tt == tn or tt == tn_corto:
                    mejor = d
                    break
            # 2) match parcial de titulo o por autor
            if not mejor:
                for d in detalles:
                    tt = _norm_basico(d.get("titulo") or "")
                    au = _norm_basico(d.get("autor") or "")
                    if (tn_corto in tt or tt in tn
                            or any(p in au for p in tn_corto.split() if len(p) > 3)):
                        mejor = d
                        break
            # 3) el primero que haya
            if not mejor and detalles:
                mejor = detalles[0]
            if mejor:
                catkey = mejor.get("catkey")
    if not catkey:
        # No se ubico el catkey: en vez de dejar que el modelo improvise un
        # "no lo encontre", darle una instruccion en POSITIVO con el Descubridor.
        titulo_libro = _titulo_libro_en_historial(historial) or ""
        q = ilsws._encode_url(_titulo_para_busqueda(titulo_libro)) if titulo_libro else ""
        url = (f"https://duoc.primo.exlibrisgroup.com/nde/search?query={q}"
               "&tab=Everything&search_scope=MyInst_and_CI&vid=56SBDU_INST:56SBDU_NDE&lang=es") if q else ""
        return ("DISPONIBILIDAD: para este título, ofrece al estudiante ver las copias por sede "
                "directamente en el catálogo, en tono positivo y natural (NUNCA digas 'no lo encontré' "
                "ni 'no pude confirmar'). " + (f"Dale este enlace ya armado: {url} . " if url else "")
                + "Ahí puede filtrar por su sede. Y si prefiere, ofrécele también la versión digital "
                "o consultar con el staff de su biblioteca.")
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
        return (f"DISPONIBILIDAD EN VIVO: \"{titulo}\" tiene copias en otras sedes ahora mismo: "
                f"{', '.join(sedes)}. Preséntalo en positivo: dile que está disponible en esas sedes y que "
                f"puede pedirlo por \"Préstamo intersede\" acercándose a su biblioteca a hacer la solicitud "
                f"(trámite presencial en su sede) para retirarlo allí, u ofrécele la versión digital si "
                f"existe. Nunca uses 'no figura' ni 'no hay'.")
    else:
        return (f"DISPONIBILIDAD EN VIVO: \"{titulo}\" en este momento tiene sus copias físicas en "
                f"circulación. Ofrécele en positivo la versión digital (si existe) como alternativa "
                f"inmediata, o solicitar aviso de disponibilidad con el staff de su biblioteca. "
                f"Nunca uses 'no hay copias' de forma seca.")


_PALABRAS_BUSCAR_LIBRO = ("tienen el libro", "tienes el libro", "buscar el libro",
                          "busco el libro", "el libro de", "libro llamado", "libro titulado",
                          "esta el libro", "está el libro", "encuentro el libro",
                          "hay algun libro", "hay algún libro", "buscame el libro",
                          "búscame el libro", "necesito el libro", "quiero el libro",
                          "ando buscando el libro", "disponible el libro", "tienen el texto",
                          "el texto de", "esta disponible el", "está disponible el",
                          "buscando el libro", "tienen disponible", "el libro", "libro de",
                          "el texto")

# Gatillos de pregunta + objeto (ej. "esta el netter?", "tienen X", "hay X")
_GATILLOS_PREGUNTA = ("esta el ", "está el ", "esta la ", "está la ", "esta ", "está ",
                      "tienen el ", "tienen la ", "tienen ", "tienes ", "hay el ", "hay ",
                      "encuentro ", "busco ", "buscas ", "quiero ", "necesito ",
                      "me interesa ", "ando buscando ", "estara ", "estará ",
                      "tendran ", "tendrán ", "tienen disponible ")

# Palabras de cortesia / temas que NO son busqueda de libro (para descartar)
_NO_ES_LIBRO = ("hola", "gracias", "buenas", "chao", "adios", "adiós", "ok", "vale",
                "perfecto", "genial", "como estas", "cómo estás", "que tal", "qué tal",
                "ayuda", "informe", "ensayo", "apa", "cita", "horario", "multa",
                "renovar", "renovacion", "renovación", "sala", "taller", "prestamo",
                "préstamo", "bloqueo", "carrera", "sede", "biblioteca", "wifi",
                "computador", "impresora", "casillero")


def _parece_busqueda_libro(tl):
    """Detecta intencion de buscar un libro. Tres vias: (1) frase explicita con
    'libro/texto'; (2) el mensaje coincide con un titulo conocido del catalogo;
    (3) patron 'pregunta + objeto' (ej. 'esta el netter?') donde el objeto no es
    una cortesia ni un tema de servicio. La via 3 es permisiva a proposito: si
    Symphony luego no encuentra nada, el flujo ofrece el Descubridor con naturalidad."""
    tl = tl.strip()
    # via 1: frase explicita con libro/texto
    if any(p in tl for p in _PALABRAS_BUSCAR_LIBRO):
        return True
    limpio = _limpia_consulta_libro(tl)
    palabras = limpio.split()
    tn = _norm_basico(limpio)
    # via 2: coincide con un titulo conocido del indice local
    if 2 <= len(palabras) <= 7:
        if tn in _LIBROS_POR_TITULO:
            return True
        for titulo_norm in _LIBROS_POR_TITULO:
            if titulo_norm.startswith(tn) or tn.startswith(titulo_norm):
                return True
            if all(p in titulo_norm.split() for p in tn.split()):
                return True
    # via 3: patron "pregunta + objeto" (ej. "esta el netter?")
    palabras_tl = set(re.findall(r"\w+", tl.lower()))
    # frases de servicio/cortesia (multi-palabra) que descartan
    _NO_FRASES = ("como estas", "cómo estás", "que tal", "qué tal", "sala de estudio",
                  "abierta la biblioteca")
    es_no = bool(palabras_tl & set(w for w in _NO_ES_LIBRO if " " not in w)) \
            or any(f in tl.lower() for f in _NO_FRASES)
    if any(tl.lower().startswith(g) for g in _GATILLOS_PREGUNTA) or tl.lower().startswith("el ") or tl.lower().startswith("la "):
        if limpio and len(limpio) >= 3 and 1 <= len(palabras) <= 6 and not es_no:
            return True
    return False


def _norm_basico(s):
    s = (s or "").lower().strip()
    s = s.replace("-", " ").replace("/", " ").replace(":", " ").replace(",", " ")
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


# Indice de busqueda de libros por TITULO, construido desde el JSON local
# (que ahora trae titulos reales). Mapea titulo_normalizado -> lista de libros.
_LIBROS_POR_TITULO = {}
for _asig in BIBLIOGRAFIA.values():
    for _libro in _asig.get("libros", []):
        _t = _libro.get("titulo")
        if not _t or _t == "Recurso de la colección":
            continue
        _tn = _norm_basico(_t)
        _entry = dict(_libro)
        _entry["_asignatura"] = _asig.get("asignatura")
        _entry["_carrera"] = _asig.get("carrera")
        _LIBROS_POR_TITULO.setdefault(_tn, []).append(_entry)


def _limpia_consulta_libro(texto):
    """Extrae el termino de busqueda de una frase como
    'quiero el libro de anatomia de netter' -> 'anatomia de netter'."""
    t = texto.lower().strip().strip("¿?¡!.")
    # quitar comillas (rectas y tipograficas)
    t = t.replace('"', " ").replace("“", " ").replace("”", " ").replace("'", " ").replace("«", " ").replace("»", " ")
    for frase in ("hola", "buenas", "por favor", "porfavor", "ando buscando el libro",
                  "tienen el libro", "tienes el libro", "busco el libro", "buscame el libro",
                  "búscame el libro", "necesito el libro", "quiero el libro", "buscando el libro",
                  "el libro de", "esta el libro", "está el libro", "estara el libro",
                  "estará el libro", "disponible el libro", "tienen el texto", "el texto de",
                  "esta disponible el", "está disponible el", "tienen disponible",
                  "el libro", "el texto", "libro de", "libro", "texto de",
                  "se encuentra", "me puedes ayudar a buscar", "ayudame a buscar",
                  "ayúdame a buscar"):
        t = t.replace(frase, " ")
    # quitar palabras de inicio tipo pregunta
    palabras = t.split()
    while palabras and palabras[0] in ("esta", "está", "estara", "estará", "tienen",
                                       "tienes", "hay", "encuentro", "busco", "buscar",
                                       "quiero", "necesito", "el", "la", "los", "las",
                                       "un", "una", "de", "disponible"):
        palabras.pop(0)
    t = " ".join(palabras)
    # quitar "en <sede>" del final (ej. "tokio blues en antonio varas" -> "tokio blues")
    for s in _SEDES:
        for patron in (f" en {s}", f" en la {s}", f" en biblioteca {s}", f" de {s}", f" {s}"):
            if t.endswith(patron):
                t = t[: -len(patron)]
                break
    return re.sub(r"\s+", " ", t).strip(" ,.")


def buscar_libro_por_titulo(termino, limite=8):
    """Busca un libro por titulo en el indice local (titulos reales del CSV).
    Devuelve lista de libros unicos por titulo. Maneja titulos repetidos."""
    tn = _norm_basico(termino)
    if len(tn) < 3:
        return []
    # 1) coincidencia exacta de titulo
    if tn in _LIBROS_POR_TITULO:
        return _dedup_por_titulo(_LIBROS_POR_TITULO[tn])
    # 2) coincidencia parcial: el termino esta contenido en el titulo o viceversa
    encontrados = []
    for titulo_norm, libros in _LIBROS_POR_TITULO.items():
        if tn in titulo_norm or titulo_norm in tn:
            encontrados.extend(libros)
    # 3) por palabras (todas las palabras del termino estan en el titulo)
    if not encontrados:
        palabras = [p for p in tn.split() if len(p) > 2]
        if palabras:
            for titulo_norm, libros in _LIBROS_POR_TITULO.items():
                if all(p in titulo_norm for p in palabras):
                    encontrados.extend(libros)
    return _dedup_por_titulo(encontrados)[:limite]


def _dedup_por_titulo(libros):
    """Quita duplicados por (titulo, autor); conserva el que tenga mas datos."""
    vistos = {}
    for l in libros:
        clave = (_norm_basico(l.get("titulo")), _norm_basico(l.get("autor") or ""))
        if clave not in vistos:
            vistos[clave] = l
    return list(vistos.values())


def _titulo_para_busqueda(titulo):
    """Limpia un titulo para buscarlo en Symphony (que busca textual).
    Quita la editorial pegada al final, el subtitulo tras ':' y ruido,
    para quedarse con el titulo principal. Ej:
    'Marketing 5.0: tecnologia para la humanidad. LID' -> 'Marketing 5.0'."""
    if not titulo:
        return ""
    t = titulo.strip()
    # cortar en el primer ':' (subtitulo) -> nos quedamos con el titulo principal
    if ":" in t:
        t = t.split(":")[0].strip()
    # si aun queda un '. Algo' al final (editorial), cortarlo
    # pero solo si lo que sigue al punto parece editorial (pocas palabras)
    if ". " in t:
        cab, _, cola = t.rpartition(". ")
        if cab and len(cola.split()) <= 3:
            t = cab.strip()
    return t.strip(" .,")


def _titulo_libro_en_historial(historial):
    """Recupera el titulo de libro mencionado por el asistente en mensajes
    recientes. El modelo NO escribe el marcador 'LIBRO ENCONTRADO' al usuario,
    asi que probamos varias formas reales: el bloque interno si quedara, un
    titulo en **negrita**, o 'Encontrado/tenemos <titulo> de <autor>'."""
    for x in reversed(historial):
        if x["role"] != "assistant":
            continue
        c = x["content"]
        # 1) marcador interno (por si el modelo lo repite)
        m = re.search(r'LIBRO ENCONTRADO:\s*"?([^"\n]+?)"?\s*(?: de |\(|$)', c)
        if m and len(m.group(1).strip()) > 2:
            return m.group(1).strip().rstrip('.')
        # 2) primer titulo en **negrita** (lo mas comun: el modelo destaca el titulo)
        m2 = re.search(r'\*\*([^*]{3,})\*\*', c)
        if m2:
            t = m2.group(1).strip().rstrip('.,:')
            # descartar negritas que sean preguntas o frases de UI
            if not any(w in t.lower() for w in ("digital", "fisic", "versión", "version", "sede", "copia")):
                return t
        # 3) 'Encontrado! Titulo de Autor' o 'tenemos Titulo de Autor'
        m3 = re.search(r'(?:encontrado|tenemos|encontr[ée])[^.!\n]*?[:!]?\s+([A-ZÁÉÍÓÚ][^.!\n]{3,}?)\s+de\s', c, re.IGNORECASE)
        if m3:
            return m3.group(1).strip()
    return None


# Respuestas cortas que indican preferencia tras buscar un libro
_RESP_DIGITAL = ("digital", "version digital", "versión digital", "en linea", "en línea", "pdf", "online")
_RESP_FISICO = ("fisico", "físico", "fisica", "física", "copia fisica", "copia física",
                "papel", "impreso", "en la sede", "para retirar")


def _buscar_symphony_robusto(termino, max_resultados=8):
    """Busca un libro en Symphony probando varias estrategias hasta encontrar:
    1) termino completo por TITLE, AUTHOR, GENERAL
    2) si trae 'X de Y', prueba solo 'X' (titulo) y solo 'Y' (autor)
    Devuelve lista de detalles (dict) deduplicada por catkey."""
    termino = (termino or "").strip()
    if len(termino) < 3:
        return []
    vistos = {}
    def _add(consulta, indices):
        for indice in indices:
            try:
                for d in ilsws.buscar_y_detallar(consulta, indice=indice, max_resultados=6):
                    ck = d.get("catkey")
                    if ck and ck not in vistos:
                        vistos[ck] = d
            except Exception:
                pass
    # 1) termino completo
    _add(termino, ("TITLE", "AUTHOR", "GENERAL"))
    # 2) si tiene "X de Y", separar titulo y autor
    if " de " in termino and len(vistos) < 3:
        partes = termino.rsplit(" de ", 1)
        titulo_parte, autor_parte = partes[0].strip(), partes[1].strip()
        if len(titulo_parte) >= 3:
            _add(titulo_parte, ("TITLE", "GENERAL"))
        if len(autor_parte) >= 3:
            _add(autor_parte, ("AUTHOR",))
    return list(vistos.values())


def _contexto_busqueda_libro(historial):
    """Busqueda de un libro por nombre. Implementa el flujo:
    - si el titulo se repite (varios libros distintos): pedir al estudiante que elija
    - si es uno solo: entregar datos y dejar que el modelo pregunte digital o fisico
    Da continuidad si el estudiante responde 'digital'/'fisico'/sede en otro mensaje.
    Usa el indice local (titulos reales) + catalogo Symphony como respaldo."""
    ultimo = ""
    for x in reversed(historial):
        if x["role"] == "user":
            ultimo = x["content"]
            break
    tl = ultimo.lower().strip()

    # ¿Es una respuesta corta de preferencia (digital/fisico) tras buscar un libro?
    es_pref = (any(p == tl or p in tl for p in _RESP_DIGITAL + _RESP_FISICO)
               and len(tl.split()) <= 4)
    termino = None
    if es_pref:
        # SOLO dar continuidad si hay un libro identificado en el historial
        termino = _titulo_libro_en_historial(historial)
        if not termino:
            return None  # "digital" suelto sin libro previo: no hacer nada
    elif _parece_busqueda_libro(tl):
        if any(p in tl for p in ("bibliografia", "bibliografía", "asignatura", "ramo")):
            return None
        termino = _limpia_consulta_libro(ultimo)
    else:
        return None
    if not termino or len(termino) < 3:
        return None

    # === BUSQUEDA EN SYMPHONY PRIMERO (catalogo completo, +300.000 libros de todas
    # las sedes). Esta es la fuente principal. El indice local de bibliografia solo
    # se usa luego para complementar el enlace digital si Symphony no lo trae. ===
    termino_busqueda = _titulo_para_busqueda(termino) or termino
    tn_busca = _norm_basico(termino_busqueda)

    # Busqueda robusta: termino completo + (si trae "X de Y") titulo y autor por
    # separado. Cubre "atlas de netter" (Netter=autor) e "ingenieria de software
    # de sommerville" (Sommerville=autor).
    detalles = _buscar_symphony_robusto(termino_busqueda, max_resultados=8)

    def _cercania(d):
        tt = _norm_basico(d.get("titulo") or "")
        au = _norm_basico(d.get("autor") or "")
        # coincidencia con titulo
        if tt == tn_busca:
            return 0
        if tt.startswith(tn_busca) or tn_busca.startswith(tt):
            return 1
        # alguna palabra del termino esta en el autor (ej "netter")
        if any(p in au for p in tn_busca.split() if len(p) > 3):
            return 2
        if tn_busca in tt or tt in tn_busca:
            return 2
        return 3
    detalles.sort(key=_cercania)

    libros = []
    for d in detalles:
        libros.append({
            "titulo": d.get("titulo"),
            "autor": d.get("autor"),
            "anio": (d.get("edicion") or "")[:4] if (d.get("edicion") or "")[:4].isdigit() else None,
            "catkey": d.get("catkey"),
            "tiene_digital": bool(d.get("enlace_digital")),
            "enlace_digital": d.get("enlace_digital"),
            "enlace": d.get("enlace_digital") or "",
        })
    libros = _dedup_por_titulo(libros)

    # Complemento: si el libro tambien esta en la bibliografia local y trae enlace
    # digital (eLibro 856) que Symphony no devolvio, lo sumamos.
    if libros:
        locales = buscar_libro_por_titulo(termino_busqueda)
        for lib in libros:
            if not lib.get("enlace_digital"):
                tnl = _norm_basico(lib.get("titulo") or "")
                for loc in locales:
                    if _norm_basico(loc.get("titulo") or "").startswith(tnl[:15]) and loc.get("enlace_digital"):
                        lib["enlace_digital"] = loc["enlace_digital"]
                        lib["tiene_digital"] = True
                        break

    # Si Symphony no encontro nada, recien ahi probar el indice local como ultimo recurso
    if not libros:
        libros = buscar_libro_por_titulo(termino_busqueda)
    if not libros:
        # Nada en ningun lado: ofrecer Descubridor en positivo
        q = ilsws._encode_url(termino_busqueda)
        url = (f"https://duoc.primo.exlibrisgroup.com/nde/search?query={q}"
               "&tab=Everything&search_scope=MyInst_and_CI&vid=56SBDU_INST:56SBDU_NDE&lang=es")
        return (f"BUSQUEDA DE LIBRO: ofrece al estudiante ver este título directamente en el catálogo, "
                f"en tono positivo (NUNCA 'no lo encontré'). Enlace ya armado: {url} . "
                "Invítalo a verlo ahí y ofrece ayuda con la versión digital o el staff si la necesita.")
    if len(libros) > 1:
        # titulos repetidos o varias coincidencias: pedir que elija
        lineas = [f"VARIOS LIBROS COINCIDEN con \"{termino}\". Dile al estudiante que hay "
                  "varias coincidencias y muéstrale la lista para que elija cuál busca "
                  "(no consultes disponibilidad aún). Opciones:"]
        for l in libros:
            etiqueta = l.get("titulo")
            if l.get("autor"):
                etiqueta += f" — {l['autor']}"
            if l.get("anio"):
                etiqueta += f" ({l['anio']})"
            lineas.append(f"- {etiqueta}")
        return "\n".join(lineas)
    # un solo libro: entregar datos y guiar el flujo digital/fisico
    l = libros[0]
    cat = l.get("catkey")
    # Si el libro local NO tiene catkey (ej. vino de eLibro), buscarlo en el
    # catalogo Symphony por titulo para conseguir el catkey y poder dar
    # disponibilidad fisica real.
    if not cat:
        titulo_busqueda = _titulo_para_busqueda(l.get("titulo") or termino)
        tn_corto = _norm_basico(titulo_busqueda)
        detalles = ilsws.buscar_y_detallar(titulo_busqueda, indice="TITLE", max_resultados=8)
        mejor = None
        for d in detalles:
            tt = _norm_basico(d.get("titulo") or "")
            if tt == tn_corto or tn_corto in tt:
                mejor = d
                break
        if not mejor and detalles:
            mejor = detalles[0]
        if mejor:
            cat = mejor.get("catkey")
            l["catkey"] = cat
            # si el libro local no traia enlace digital pero Symphony si, usarlo
            if not l.get("enlace_digital") and mejor.get("enlace_digital"):
                l["enlace_digital"] = mejor["enlace_digital"]
                l["tiene_digital"] = True
    tiene_dig = l.get("tiene_digital", False)
    dig = l.get("enlace_digital") or l.get("enlace")
    info = [f"LIBRO ENCONTRADO: \"{l.get('titulo')}\""
            + (f" de {l['autor']}" if l.get("autor") else "")
            + (f" ({l['anio']})" if l.get("anio") else "") + "."]
    info.append("FLUJO OBLIGATORIO: pregunta primero si prefiere la versión DIGITAL o una "
                "copia FÍSICA, y actúa según su respuesta:")
    if tiene_dig and dig and "bibliotecabuscador" not in dig and "SD_ILS" not in dig:
        info.append(f"- Si pide DIGITAL: entrégale este enlace de acceso directo: {ilsws._encode_url(dig)} "
                    "(se entra con credenciales Duoc).")
    else:
        info.append("- Si pide DIGITAL: este título no tiene versión digital en el sistema; "
                    "ofrécele buscarlo en el Descubridor y dale el enlace de búsqueda ya armado.")
    if cat:
        info.append(f"- Si pide FÍSICA: el catkey es {cat}. Pregúntale de qué sede es; cuando responda, "
                    "el sistema consultará disponibilidad en vivo y le dirás en qué sedes hay copias. Si "
                    "está en su sede, que la retire allí. Si no está en su sede pero sí en otra, dile que "
                    "puede pedirla mediante \"Préstamo intersede\" ACERCÁNDOSE a su biblioteca a realizar "
                    "la solicitud (el trámite es presencial en su sede).")
    else:
        info.append("- Si pide FÍSICA: pregúntale su sede igual; el sistema intentará la disponibilidad. "
                    "Si no se puede, ofrécele el Descubridor en positivo para ver copias por sede.")
    return "\n".join(info)


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
    # 1) Si el mensaje nombra explicitamente un libro NUEVO (ej. "esta el libro X
    #    en Antonio Varas?"), NO es continuacion: dejar que lo tome la busqueda de
    #    libro. Solo tratamos como "disponibilidad por sede" (continuacion) si el
    #    mensaje es corto y NO introduce un titulo nuevo.
    nombra_libro_nuevo = any(p in tl for p in (
        "el libro ", "libro llamado", "libro titulado", "el texto ",
        "busco ", "buscar ", "necesito ", "quiero ", "tienen "))
    if not nombra_libro_nuevo:
        # consulta puntual de disponibilidad por sede (continuacion del libro anterior)
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
    if not ctx:
        # si no es bibliografia de ramo, ver si es busqueda de un libro por nombre
        ctx = _contexto_busqueda_libro(historial)
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
                    cuerpo = await r.aread()
                    log.error("Anthropic %s: %s", r.status_code, cuerpo.decode("utf-8", "ignore")[:800])
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
