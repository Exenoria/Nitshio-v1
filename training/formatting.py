"""Formato de ejemplos para el fine-tune QLoRA.

Mantiene una sola representación de chat entre el dataset local y el script de
nube. El texto de entrenamiento incluye el system prompt, el turno del usuario
y la respuesta JSON compacta que debe aprender el modelo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from training.schema import DatasetExample, load_dataset


def ejemplo_a_mensajes(
    ejemplo: DatasetExample,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Convierte un ejemplo al formato de mensajes de Hugging Face."""
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": ejemplo.user},
        {"role": "assistant", "content": ejemplo.assistant_json()},
    ]


def renderizar_chat(
    tokenizer: Any,
    mensajes: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Aplica la plantilla del tokenizador o usa un formato Phi-3 mínimo."""
    try:
        texto = tokenizer.apply_chat_template(
            mensajes,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if texto:
            return str(texto)
    except (AttributeError, TypeError, ValueError):
        pass
    # Fallback legible para modelos sin chat template. El script de nube usa
    # el tokenizador real; esto solo evita romper si falta la plantilla.
    partes = []
    for mensaje in mensajes:
        rol = mensaje["role"]
        contenido = mensaje["content"]
        if rol == "system":
            partes.append(f"<|system|>\n{contenido}<|end|>\n")
        elif rol == "user":
            partes.append(f"<|user|>\n{contenido}<|end|>\n")
        elif rol == "assistant":
            partes.append(f"<|assistant|>\n{contenido}<|end|>\n")
    if add_generation_prompt:
        partes.append("<|assistant|>\n")
    return "".join(partes)


def ejemplo_a_texto(
    ejemplo: DatasetExample,
    tokenizer: Any,
    system_prompt: str,
) -> dict[str, str]:
    """Devuelve ``{"text": ...}`` para SFTTrainer."""
    mensajes = ejemplo_a_mensajes(ejemplo, system_prompt)
    return {"text": renderizar_chat(tokenizer, mensajes)}


def cargar_textos(
    path: Path | str,
    emociones: Iterable[str],
    tokenizer: Any,
    system_prompt: str,
    *,
    limite: int | None = None,
) -> list[dict[str, str]]:
    """Carga un JSONL y lo convierte en textos de entrenamiento."""
    resultado = load_dataset(path, emociones)
    if not resultado.ok:
        detalles = "; ".join(str(issue) for issue in resultado.errors[:5])
        raise ValueError(f"El dataset tiene errores: {detalles}")
    ejemplos = resultado.examples[:limite] if limite else resultado.examples
    return [
        ejemplo_a_texto(ejemplo, tokenizer, system_prompt) for ejemplo in ejemplos
    ]


def resumen_dataset(path: Path | str, emociones: Iterable[str]) -> dict[str, Any]:
    """Resumen reproducible para el manifiesto de entrenamiento."""
    from training.schema import contar_emociones, hash_examples, hash_file

    resultado = load_dataset(path, emociones)
    if not resultado.ok:
        raise ValueError(
            "El dataset tiene errores; ejecuta training/validate_dataset.py."
        )
    return {
        "path": str(path),
        "examples": len(resultado.examples),
        "by_emotion": contar_emociones(resultado.examples),
        "dataset_sha256": hash_examples(resultado.examples),
        "file_sha256": hash_file(path),
        "duplicates_discarded": resultado.duplicates,
    }


def escribir_jsonl(
    path: Path | str, registros: Iterable[dict[str, Any]]
) -> None:
    """Escribe registros JSONL sin ordenar."""
    ruta = Path(path)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as fh:
        for registro in registros:
            fh.write(
                json.dumps(registro, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
