"""
enriquecer_bibliografia.py - Pre-procesamiento (se corre UNA vez, en local)

Recorre los catkeys unicos del bibliografia.json, consulta Symphony Web Services
para cada uno, y guarda en el JSON:
  - titulo real (MARC 245) y autor (MARC 100)
  - tiene_digital + enlace_digital (MARC 856 mas reciente)
  - tiene_fisico (si hay copias en catalogo)

Asi, en produccion, el chatbot sabe al instante que opciones mostrar para cada
recurso (version digital / copias fisicas / ambas) SIN consultar en vivo al
listar. La disponibilidad por SEDE si se consulta en vivo, pero solo cuando el
estudiante pide ver copias de un libro puntual (rapido, una sola consulta).

Uso:
  cd backend
  python enriquecer_bibliografia.py

Genera bibliografia.json enriquecido (respaldo del original en bibliografia.bak.json).
Es idempotente: si lo corres de nuevo, reusa lo ya enriquecido salvo --forzar.
"""

import json
import os
import re
import sys
import time

import ilsws

ORIGEN = os.path.join(os.path.dirname(__file__), "bibliografia.json")
RESPALDO = os.path.join(os.path.dirname(__file__), "bibliografia.bak.json")
PAUSA = 0.3          # segundos entre consultas, para no saturar el Web Service
_CATKEY_RE = re.compile(r"SD_ILS:(\d+)")

forzar = "--forzar" in sys.argv


def main():
    with open(ORIGEN, encoding="utf-8") as f:
        biblio = json.load(f)

    # respaldo del original (solo la primera vez)
    if not os.path.exists(RESPALDO):
        with open(RESPALDO, "w", encoding="utf-8") as f:
            json.dump(biblio, f, ensure_ascii=False)
        print(f"Respaldo guardado en {RESPALDO}")

    # 1) recolectar catkeys unicos
    catkeys = set()
    for asig in biblio.values():
        for libro in asig["libros"]:
            m = _CATKEY_RE.search(libro["enlace"])
            if m:
                catkeys.add(m.group(1))
    catkeys = sorted(catkeys)
    print(f"{len(catkeys)} catkeys unicos a consultar.\n")

    # 2) consultar Symphony para cada catkey
    info = {}   # catkey -> dict con datos
    for i, ck in enumerate(catkeys, 1):
        datos = ilsws.consultar_titulo(ck)
        if datos:
            info[ck] = {
                "titulo": datos.get("titulo"),
                "autor": datos.get("autor"),
                "tiene_digital": bool(datos.get("enlace_digital")),
                "enlace_digital": datos.get("enlace_digital"),
                "tiene_fisico": datos.get("copias_disponibles", 0) > 0
                                or bool(datos.get("sedes_disponibles")),
            }
        else:
            info[ck] = None
        if i % 25 == 0 or i == len(catkeys):
            ok = sum(1 for v in info.values() if v)
            print(f"  {i}/{len(catkeys)} consultados ({ok} con datos)")
        time.sleep(PAUSA)

    # 3) enriquecer cada recurso del JSON con lo encontrado
    enriquecidos = 0
    for asig in biblio.values():
        for libro in asig["libros"]:
            m = _CATKEY_RE.search(libro.get("enlace", ""))
            if not m:
                # recurso que no es de catalogo (eLibro directo): es digital
                libro["tiene_digital"] = True
                libro.setdefault("enlace_digital", libro.get("enlace"))
                libro["tiene_fisico"] = False
                continue
            ck = m.group(1)
            datos = info.get(ck)
            libro["catkey"] = ck
            if datos:
                # El titulo del CSV es el bueno; NO lo pisamos. Solo completamos
                # autor si el CSV no lo trae.
                if not libro.get("autor") and datos.get("autor"):
                    libro["autor"] = datos["autor"]
                # enlace digital: preferir el del CSV; si no hay, usar el de Symphony (856)
                if not libro.get("enlace_digital") and datos.get("enlace_digital"):
                    libro["enlace_digital"] = datos["enlace_digital"]
                libro["tiene_digital"] = bool(libro.get("enlace_digital"))
                libro["tiene_fisico"] = datos["tiene_fisico"]
                enriquecidos += 1
            else:
                # no se pudo consultar; marcar conservador
                libro.setdefault("tiene_digital", bool(libro.get("enlace_digital")))
                libro.setdefault("tiene_fisico", True)  # es de catalogo, probablemente fisico

    with open(ORIGEN, "w", encoding="utf-8") as f:
        json.dump(biblio, f, ensure_ascii=False)
    print(f"\nListo. {enriquecidos} recursos de catalogo enriquecidos.")
    print(f"Guardado en {ORIGEN}")


if __name__ == "__main__":
    main()
