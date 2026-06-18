"""
ilsws.py - Integracion con SirsiDynix Symphony Web Services (ILSWS)
Chatbot Bibliotecas Duoc UC.

Consulta el catalogo en tiempo real (solo lectura, sin login de usuario):
titulo, autor y disponibilidad por sede de un titulo dado su catalog key.

Resuelve el problema de los recursos del catalogo que en el CSV de bibliografia
aparecen sin titulo legible: aqui se recupera el titulo real desde Symphony.

Config por variables de entorno (en Render):
  ILSWS_BASE_URL   (default: https://duchi.sirsidynix.net/duchi_ilsws)
  ILSWS_CLIENT_ID  (default: SymWSTestClient; en produccion, un client dedicado)

Diseno:
  - Solo lectura de catalogo. No maneja datos de usuarios.
  - Cache en memoria con TTL: la disponibilidad cambia, pero no cada segundo;
    cachear ~10 min evita golpear el Web Service en cada consulta del chatbot.
  - Timeouts cortos y tolerancia a fallos: si el Web Service no responde, el
    chatbot sigue funcionando con los datos que ya tiene (degradacion elegante).
"""

import os
import re
import time
import logging
import unicodedata
import xml.etree.ElementTree as ET

import httpx

log = logging.getLogger("ilsws")

ILSWS_BASE_URL = os.environ.get("ILSWS_BASE_URL", "https://duchi.sirsidynix.net/duchi_ilsws").rstrip("/")
ILSWS_CLIENT_ID = os.environ.get("ILSWS_CLIENT_ID", "SymWSTestClient")
_NS = "{http://schemas.sirsidynix.com/symws/standard}"

_CACHE = {}
_CACHE_TTL = 600  # 10 minutos

_CATKEY_RE = re.compile(r"SD_ILS:(\d+)")


def catkey_desde_url(url):
    """Extrae el catalog key de una URL de catalogo Symphony, o None."""
    if not isinstance(url, str):
        return None
    m = _CATKEY_RE.search(url)
    return m.group(1) if m else None


def _texto(el):
    return el.text.strip() if el is not None and el.text else None


def _encode_url(url):
    """Codifica los espacios y caracteres no seguros de una URL ya formada,
    sin tocar lo que ya esté codificado. Los enlaces del catalogo Duoc vienen
    con espacios sin codificar (ej '.../10175-Diseño acustico ...') que rompen
    el enlace; aqui se arreglan."""
    if not isinstance(url, str):
        return url
    # Codifica solo espacios y caracteres problematicos; deja intactos los ya válidos
    return url.replace(" ", "%20")


def _marc(bib, entry_id):
    """Texto del primer campo MARC con ese entryID (ej '245')."""
    for mei in bib.findall(f"{_NS}MarcEntryInfo"):
        eid = mei.find(f"{_NS}entryID")
        if eid is not None and eid.text == entry_id:
            return _texto(mei.find(f"{_NS}text"))
    return None


def _limpia_titulo(t245):
    """MARC 245: 'Titulo : subtitulo / responsabilidad'. Quita ' / ...'."""
    if not t245:
        return None
    return t245.split(" / ")[0].strip()


def _limpia_autor(a100):
    """MARC 100: 'Apellido, Nombre, 1942-'. Quita fechas finales."""
    if not a100:
        return None
    return re.sub(r",?\s*\d{4}-?\d{0,4}\.?\s*$", "", a100).strip(" ,")


def consultar_titulo(catkey, con_disponibilidad=True):
    """Consulta el Web Service por catalog key. Devuelve dict o None si falla.
    Estructura devuelta:
      {catkey, titulo, autor, edicion, isbn, copias_disponibles,
       sedes_disponibles: [...], holdable, enlace_digital}
    """
    catkey = str(catkey).strip()
    if not catkey.isdigit():
        return None

    # cache
    hit = _CACHE.get(catkey)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]

    url = f"{ILSWS_BASE_URL}/rest/standard/lookupTitleInfo"
    params = {
        "clientID": ILSWS_CLIENT_ID,
        "titleID": catkey,
        "includeAvailabilityInfo": "true" if con_disponibilidad else "false",
        "marcEntryFilter": "ALL",
    }
    try:
        r = httpx.get(url, params=params, timeout=8)
        if r.status_code != 200:
            log.warning("ILSWS %s para catkey=%s", r.status_code, catkey)
            return None
        datos = _parse(r.text, catkey)
        if datos:
            _CACHE[catkey] = (time.time(), datos)
        return datos
    except Exception as e:  # noqa
        log.warning("ILSWS error catkey=%s: %s", catkey, e)
        return None


def _parse(xml_text, catkey):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    ti = root.find(f".//{_NS}TitleInfo")
    if ti is None:
        return None
    bib = ti.find(f"{_NS}BibliographicInfo")

    titulo = autor = edicion = isbn = enlace = None
    if bib is not None:
        titulo = _limpia_titulo(_marc(bib, "245"))
        autor = _limpia_autor(_marc(bib, "100"))
        edicion = _marc(bib, "250")
        isbn_raw = _marc(bib, "020")
        if isbn_raw:
            mi = re.search(r"(\d{10,13})", isbn_raw)
            isbn = mi.group(1) if mi else None
        # enlace(s) digital(es) (MARC 856): puede haber varios (distintas ediciones).
        # Elegimos el del anio mas reciente, leyendo el anio del texto del enlace/etiqueta.
        candidatos = []  # (anio, url)
        for mei in bib.findall(f"{_NS}MarcEntryInfo"):
            eid = mei.find(f"{_NS}entryID")
            if eid is None or eid.text != "856":
                continue
            u = mei.find(f"{_NS}url")
            url_dig = (u.text.strip() if u is not None and u.text else None)
            if not url_dig:
                continue
            # buscar el anio (19xx/20xx) en url + textos, para ordenar por recencia
            contexto = " ".join(filter(None, [
                url_dig,
                _texto(mei.find(f"{_NS}text")),
                _texto(mei.find(f"{_NS}unformattedText")),
            ]))
            anios_full = [int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", contexto)]
            anio = max(anios_full) if anios_full else 0
            candidatos.append((anio, url_dig))
        if candidatos:
            # el de mayor anio; si empatan o no hay anio, el primero encontrado
            candidatos.sort(key=lambda x: x[0], reverse=True)
            enlace = _encode_url(candidatos[0][1])

    av = ti.find(f"{_NS}TitleAvailabilityInfo")
    copias = 0
    sedes = []
    holdable = False
    if av is not None:
        c = _texto(av.find(f"{_NS}totalCopiesAvailable"))
        copias = int(c) if c and c.isdigit() else 0
        sedes = [_texto(s) for s in av.findall(f"{_NS}libraryWithAvailableCopies") if _texto(s)]
        holdable = (_texto(av.find(f"{_NS}holdable")) == "true")

    if not titulo and copias == 0 and not sedes:
        return None

    # anio de la edicion: del campo 250, del enlace digital elegido, o del titulo
    anio_edicion = 0
    fuentes_anio = " ".join(filter(None, [edicion, enlace, titulo]))
    anios_ed = [int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", fuentes_anio or "")]
    if anios_ed:
        anio_edicion = max(anios_ed)

    return {
        "catkey": catkey,
        "titulo": titulo,
        "autor": autor,
        "edicion": edicion,
        "anio_edicion": anio_edicion,
        "isbn": isbn,
        "copias_disponibles": copias,
        "sedes_disponibles": sedes,
        "holdable": holdable,
        "enlace_digital": enlace,
    }


def disponible_en_sede(datos, sede_usuario):
    """Devuelve (disponible_bool, sedes_normalizadas). Compara el nombre de la
    sede del estudiante con la lista de sedes con copias disponibles, de forma
    tolerante (sin tildes, parcial)."""
    if not datos or not datos.get("sedes_disponibles"):
        return (False, [])
    def norm(s):
        s = (s or "").lower().strip()
        return "".join(c for c in unicodedata.normalize("NFD", s)
                       if unicodedata.category(c) != "Mn")
    su = norm(sede_usuario)
    for sede in datos["sedes_disponibles"]:
        sn = norm(sede)
        if su and (su in sn or sn in su):
            return (True, datos["sedes_disponibles"])
    return (False, datos["sedes_disponibles"])


def resumen_disponibilidad(datos, max_sedes=6):
    """Texto corto y claro de disponibilidad para inyectar al modelo."""
    if not datos:
        return None
    n = datos["copias_disponibles"]
    sedes = datos["sedes_disponibles"]
    if n <= 0 or not sedes:
        return "Sin copias disponibles en este momento (consulta con el staff de la biblioteca)."
    muestra = ", ".join(sedes[:max_sedes])
    extra = f" y {len(sedes) - max_sedes} sede(s) mas" if len(sedes) > max_sedes else ""
    return f"{n} copia(s) disponible(s) en: {muestra}{extra}."


# --- Busqueda en el catalogo por texto (endpoint BLUEcloud) ---------------
# Descubierto: GET {base}/catalog/bib/search?clientID=BCCAT&q=INDICE:termino
# Devuelve catalog keys que luego se consultan con consultar_titulo().
ILSWS_SEARCH_CLIENT = os.environ.get("ILSWS_SEARCH_CLIENT", "BCCAT")
_SEARCH_URL = f"{ILSWS_BASE_URL}/catalog/bib/search"


def buscar_catalogo(texto, indice="GENERAL", limite=5):
    """Busca en el catalogo por texto y devuelve una lista de catalog keys.
    indice: GENERAL (todo), TITLE, AUTHOR, SUBJECT.
    Tolerante a fallos: si algo sale mal devuelve lista vacia."""
    texto = (texto or "").strip()
    if not texto:
        return []
    # cache
    clave = f"search:{indice}:{texto.lower()}"
    hit = _CACHE.get(clave)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]
    params = {"clientID": ILSWS_SEARCH_CLIENT, "q": f"{indice}:{texto}",
              "rw": "1", "ct": str(limite)}
    headers = {"sd-originating-app-id": "chatbot", "Accept": "application/json",
               "x-sirs-clientID": ILSWS_SEARCH_CLIENT}
    try:
        r = httpx.get(_SEARCH_URL, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            log.warning("busqueda %s para q=%s", r.status_code, texto)
            return []
        data = r.json()
        keys = [item.get("key") for item in data.get("result", []) if item.get("key")]
        _CACHE[clave] = (time.time(), keys)
        return keys
    except Exception as e:  # noqa
        log.warning("busqueda error q=%s: %s", texto, e)
        return []


def buscar_y_detallar(texto, indice="GENERAL", max_resultados=3):
    """Busca por texto y devuelve los detalles (titulo, autor, disponibilidad,
    enlace digital) de los primeros resultados. Combina busqueda + lookup."""
    keys = buscar_catalogo(texto, indice=indice, limite=max_resultados)
    resultados = []
    for k in keys[:max_resultados]:
        d = consultar_titulo(k)
        if d:
            resultados.append(d)
    return resultados


if __name__ == "__main__":
    # Prueba manual: python ilsws.py 13882   |   python ilsws.py buscar sommerville
    import sys, json
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 2 and sys.argv[1] == "buscar":
        termino = " ".join(sys.argv[2:])
        print(f"Buscando: {termino}")
        for d in buscar_y_detallar(termino):
            print(json.dumps(d, ensure_ascii=False, indent=2))
            print("Resumen:", resumen_disponibilidad(d), "\n")
    else:
        ck = sys.argv[1] if len(sys.argv) > 1 else "13882"
        d = consultar_titulo(ck)
        print(json.dumps(d, ensure_ascii=False, indent=2))
        if d:
            print("\nResumen:", resumen_disponibilidad(d))
