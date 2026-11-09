"""División de texto en chunks para TTS de baja latencia.

El primer chunk es más corto para que Nitshio empiece a hablar antes; los
siguientes respetan ``max_chars``. Se corta por frases y, si una frase es
demasiado larga, por cláusulas y palabras. Nunca se parte una palabra salvo
que ella sola supere el límite.
"""

from __future__ import annotations

import re

_ESPACIOS = re.compile(r"\s+")
_FIN_ORACION = re.compile(r"(?<=[.!?…])\s+|\n+")
_CLAUSULA = re.compile(r"(?<=[,;:—–])\s+")


def normalizar_texto(texto: str) -> str:
    """Colapsa espacios y saltos de línea sin perder la puntuación."""
    return _ESPACIOS.sub(" ", str(texto or "")).strip()


def _partir_por_palabras(texto: str, limite: int) -> list[str]:
    """Parte un texto largo por palabras; si una palabra excede, la trocea."""
    piezas: list[str] = []
    actual = ""
    for palabra in texto.split():
        if len(palabra) > limite:
            if actual:
                piezas.append(actual)
                actual = ""
            for inicio in range(0, len(palabra), limite):
                piezas.append(palabra[inicio : inicio + limite])
            continue
        candidato = f"{actual} {palabra}".strip()
        if actual and len(candidato) > limite:
            piezas.append(actual)
            actual = palabra
        else:
            actual = candidato
    if actual:
        piezas.append(actual)
    return piezas


def _partir_larga(oracion: str, limite: int) -> list[str]:
    """Parte una oración que no cabe en el límite."""
    if len(oracion) <= limite:
        return [oracion]
    piezas: list[str] = []
    for clausula in _CLAUSULA.split(oracion):
        clausula = clausula.strip()
        if not clausula:
            continue
        if len(clausula) <= limite:
            piezas.append(clausula)
        else:
            piezas.extend(_partir_por_palabras(clausula, limite))
    # Reagrupa cláusulas cortas para no generar demasiados chunks.
    agrupadas: list[str] = []
    actual = ""
    for pieza in piezas:
        candidato = f"{actual} {pieza}".strip()
        if actual and len(candidato) > limite:
            agrupadas.append(actual)
            actual = pieza
        else:
            actual = candidato
    if actual:
        agrupadas.append(actual)
    return agrupadas


def dividir_en_chunks(
    texto: str,
    *,
    max_chars: int = 180,
    first_chunk_max_chars: int = 90,
    min_chars: int = 12,
) -> list[str]:
    """Devuelve una lista de chunks listos para sintetizar.

    - El primer chunk respeta ``first_chunk_max_chars``.
    - Ningún chunk supera ``max_chars`` salvo una palabra imposible de partir.
    - Los chunks finales demasiado cortos se fusionan si caben.
    """
    limpio = normalizar_texto(texto)
    if not limpio:
        return []
    max_chars = max(8, int(max_chars))
    first_chunk_max_chars = max(8, int(first_chunk_max_chars))
    min_chars = max(1, int(min_chars))

    oraciones = [parte.strip() for parte in _FIN_ORACION.split(limpio) if parte.strip()]
    if not oraciones:
        oraciones = [limpio]

    unidades: list[str] = []
    for oracion in oraciones:
        unidades.extend(_partir_larga(oracion, max_chars))

    chunks: list[str] = []
    actual = ""
    for unidad in unidades:
        limite = first_chunk_max_chars if not chunks else max_chars
        candidato = f"{actual} {unidad}".strip()
        if actual and len(candidato) > limite:
            chunks.append(actual)
            actual = unidad
        else:
            actual = candidato
    if actual:
        chunks.append(actual)

    # Fusiona chunks finales demasiado cortos con el anterior si cabe.
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        fusion = f"{chunks[-2]} {chunks[-1]}"
        if len(fusion) <= max_chars:
            chunks[-2:] = [fusion]

    # Si el primer chunk quedó diminuto y hay más, intenta fusionarlo con el
    # segundo respetando el límite del primero.
    if len(chunks) >= 2 and len(chunks[0]) < min_chars:
        fusion = f"{chunks[0]} {chunks[1]}"
        if len(fusion) <= max_chars:
            chunks[0:2] = [fusion]

    return [chunk for chunk in chunks if chunk]
