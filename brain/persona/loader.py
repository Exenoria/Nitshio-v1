"""Carga de la personalidad, emociones y ejemplos few-shot de Nitshio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from brain.llm.parser import NitshioResponse, ParseError, parse_response

ROOT = Path(__file__).resolve().parents[2]


class PersonaError(RuntimeError):
    """La configuración o los archivos de personalidad son inválidos."""


@dataclass(frozen=True)
class FewShot:
    """Un ejemplo de conversación user -> assistant en formato Nitshio."""

    user: str
    response: str
    emotion: str

    def assistant_json(self) -> str:
        """Devuelve el JSON compacto que vería el modelo como respuesta."""
        return json.dumps(
            {"response": self.response, "emotion": self.emotion},
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class Persona:
    """Personalidad completa cargada desde disco."""

    name: str
    system_prompt: str
    few_shots: tuple[FewShot, ...]
    emotions: tuple[str, ...]
    default_emotion: str
    include_few_shots_in_runtime: bool = True


def _ruta(root: Path, relativa: str) -> Path:
    """Resuelve una ruta relativa a la raíz del proyecto."""
    path = Path(relativa)
    return path if path.is_absolute() else root / path


def cargar_emociones(path: Path) -> tuple[list[str], str]:
    """Lee las emociones permitidas y la emoción por defecto."""
    if not path.exists():
        raise PersonaError(f"No existe el archivo de emociones: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    emociones = list((data.get("emotions") or {}).keys())
    if not emociones:
        raise PersonaError(f"No hay emociones declaradas en {path}")
    default = str(data.get("default", emociones[0]))
    if default not in emociones:
        raise PersonaError(
            f"La emoción por defecto {default!r} no está en {path}"
        )
    return emociones, default


def cargar_few_shots(
    path: Path, emociones: Iterable[str]
) -> tuple[FewShot, ...]:
    """Carga y valida cada línea JSONL de ejemplos few-shot."""
    if not path.exists():
        raise PersonaError(f"No existe el archivo de few-shots: {path}")
    ejemplos: list[FewShot] = []
    with path.open("r", encoding="utf-8") as fh:
        for numero, linea in enumerate(fh, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                data = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise PersonaError(
                    f"Few-shot línea {numero}: JSON inválido ({exc})"
                ) from exc
            if not isinstance(data, dict):
                raise PersonaError(
                    f"Few-shot línea {numero}: se esperaba un objeto."
                )
            user = str(data.get("user", "")).strip()
            if not user:
                raise PersonaError(
                    f"Few-shot línea {numero}: falta 'user'."
                )
            assistant = {
                "response": data.get("response", ""),
                "emotion": data.get("emotion", ""),
            }
            try:
                parsed = parse_response(
                    json.dumps(assistant, ensure_ascii=False), emociones
                )
            except ParseError as exc:
                raise PersonaError(
                    f"Few-shot línea {numero}: {exc}"
                ) from exc
            ejemplos.append(
                FewShot(
                    user=user,
                    response=parsed.data.response,
                    emotion=parsed.data.emotion,
                )
            )
    if not ejemplos:
        raise PersonaError(f"El archivo de few-shots está vacío: {path}")
    return tuple(ejemplos)


def cargar_persona(
    config: dict[str, Any], root: Path | None = None
) -> Persona:
    """Carga personalidad, emociones y few-shots desde la configuración."""
    root = root or ROOT
    persona_cfg = config.get("persona") or {}
    system_rel = persona_cfg.get("system_file")
    few_rel = persona_cfg.get("few_shots_file")
    emotions_rel = persona_cfg.get("emotions_file")
    if not system_rel or not few_rel or not emotions_rel:
        raise PersonaError(
            "config.persona debe declarar system_file, few_shots_file y "
            "emotions_file."
        )

    system_path = _ruta(root, system_rel)
    if not system_path.exists():
        raise PersonaError(f"No existe el prompt de personalidad: {system_path}")
    system_prompt = system_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise PersonaError(f"El prompt de personalidad está vacío: {system_path}")

    emociones, default = cargar_emociones(_ruta(root, emotions_rel))
    few_shots = cargar_few_shots(_ruta(root, few_rel), emociones)

    return Persona(
        name=str((config.get("project") or {}).get("name", "Nitshio")),
        system_prompt=system_prompt,
        few_shots=few_shots,
        emotions=tuple(emociones),
        default_emotion=default,
        include_few_shots_in_runtime=bool(
            persona_cfg.get("include_few_shots_in_runtime", True)
        ),
    )


def construir_system_prompt(
    persona: Persona,
    contexto: str | None = None,
    max_context_chars: int = 12000,
) -> str:
    """Añade contexto dinámico (memoria, herramientas, autoconocimiento)."""
    base = persona.system_prompt
    if not contexto:
        return base

    contexto = contexto.strip()
    margen = max(0, int(max_context_chars) - len(base) - 200)
    if margen <= 0:
        return base
    if len(contexto) > margen:
        contexto = contexto[:margen].rsplit("\n", 1)[0] + "\n[...contexto recortado...]"

    return (
        f"{base}\n\n"
        "## Contexto de esta conversación\n"
        "Usa este contexto si es relevante. No inventes datos que no estén aquí.\n\n"
        f"{contexto}"
    )


def construir_mensajes(
    persona: Persona,
    user_text: str,
    *,
    history: Iterable[dict[str, str]] | None = None,
    contexto: str | None = None,
    max_context_chars: int = 12000,
) -> list[dict[str, str]]:
    """Construye la lista de mensajes para la API de chat de Ollama."""
    mensajes: list[dict[str, str]] = [
        {
            "role": "system",
            "content": construir_system_prompt(
                persona, contexto=contexto, max_context_chars=max_context_chars
            ),
        }
    ]

    if persona.include_few_shots_in_runtime:
        for ejemplo in persona.few_shots:
            mensajes.append({"role": "user", "content": ejemplo.user})
            mensajes.append(
                {"role": "assistant", "content": ejemplo.assistant_json()}
            )

    for mensaje in history or []:
        rol = mensaje.get("role")
        contenido = mensaje.get("content")
        if rol in {"user", "assistant"} and contenido:
            mensajes.append({"role": rol, "content": str(contenido)})

    mensajes.append({"role": "user", "content": user_text})
    return mensajes


def validar_few_shots(persona: Persona) -> list[str]:
    """Devuelve una lista de problemas detectados en los few-shots."""
    problemas: list[str] = []
    for indice, ejemplo in enumerate(persona.few_shots, start=1):
        try:
            parse_response(ejemplo.assistant_json(), persona.emotions)
        except ParseError as exc:
            problemas.append(f"Few-shot {indice}: {exc}")
    return problemas


def respuesta_a_json(respuesta: NitshioResponse) -> str:
    """Serializa una respuesta validada al formato compacto del proyecto."""
    return json.dumps(
        respuesta.model_dump(), ensure_ascii=False, separators=(",", ":")
    )
