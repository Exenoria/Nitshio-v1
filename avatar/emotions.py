"""Emociones configurables y transiciones de blendshapes VRM.

La lista de emociones vive en ``config/emotions.yaml``; este módulo no conoce
nombres hardcodeados. El controlador interpola valores para que VSeeFace no
reciba saltos bruscos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


class EmotionError(RuntimeError):
    """La configuración de emociones es inválida."""


@dataclass(frozen=True)
class EmotionSpec:
    """Definición de una emoción y su blendshape."""

    name: str
    display_name: str
    blendshape: str
    intensity: float
    fade_in_ms: int
    hold_ms: int
    fade_out_ms: int
    description: str = ""


@dataclass(frozen=True)
class EmotionConfig:
    """Configuración completa de emociones."""

    default: str
    emotions: dict[str, EmotionSpec]
    tick_hz: int = 60
    return_to_neutral_ms: int = 1200
    crossfade_ms: int = 180

    @property
    def blendshapes(self) -> tuple[str, ...]:
        return tuple(sorted({spec.blendshape for spec in self.emotions.values()}))


def cargar_emociones(path: Path) -> EmotionConfig:
    """Carga y valida ``config/emotions.yaml``."""
    if not path.exists():
        raise EmotionError(f"No existe el archivo de emociones: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    crudas = data.get("emotions") or {}
    if not isinstance(crudas, dict) or not crudas:
        raise EmotionError(f"No hay emociones declaradas en {path}")
    emociones: dict[str, EmotionSpec] = {}
    for nombre, valores in crudas.items():
        valores = valores or {}
        blendshape = str(valores.get("vrm_blendshape", nombre))
        emociones[str(nombre)] = EmotionSpec(
            name=str(nombre),
            display_name=str(valores.get("display_name", nombre)),
            blendshape=blendshape,
            intensity=max(0.0, min(1.0, float(valores.get("intensity", 1.0)))),
            fade_in_ms=max(0, int(valores.get("fade_in_ms", 200))),
            hold_ms=max(0, int(valores.get("hold_ms", 0))),
            fade_out_ms=max(0, int(valores.get("fade_out_ms", 500))),
            description=str(valores.get("description", "")),
        )
    default = str(data.get("default", next(iter(emociones))))
    if default not in emociones:
        raise EmotionError(
            f"La emoción por defecto {default!r} no está declarada."
        )
    transicion = data.get("transition") or {}
    return EmotionConfig(
        default=default,
        emotions=emociones,
        tick_hz=max(1, int(transicion.get("tick_hz", 60))),
        return_to_neutral_ms=max(
            0, int(transicion.get("return_to_neutral_ms", 1200))
        ),
        crossfade_ms=max(0, int(transicion.get("crossfade_ms", 180))),
    )


def _interpolar(
    desde: dict[str, float], hasta: dict[str, float], t: float
) -> dict[str, float]:
    t = max(0.0, min(1.0, float(t)))
    claves = set(desde) | set(hasta)
    return {
        clave: float(desde.get(clave, 0.0))
        + (float(hasta.get(clave, 0.0)) - float(desde.get(clave, 0.0))) * t
        for clave in claves
    }


class EmotionController:
    """Máquina de estados de una emoción con fade in/hold/fade out."""

    def __init__(self, config: EmotionConfig) -> None:
        self.config = config
        self._specs = config.emotions
        self._neutral = self._objetivo(config.default)
        self._actual = dict(self._neutral)
        self._desde = dict(self._neutral)
        self._hasta = dict(self._neutral)
        self._spec: EmotionSpec = self._specs[config.default]
        self._activa = config.default
        self._fase = "idle"
        self._transcurrido = 0.0

    # ------------------------------------------------------------------
    # Objetivos
    # ------------------------------------------------------------------
    def _objetivo(self, nombre: str) -> dict[str, float]:
        spec = self._specs[nombre]
        valores = {blendshape: 0.0 for blendshape in self.config.blendshapes}
        valores[spec.blendshape] = spec.intensity
        return valores

    def _neutral_objetivo(self) -> dict[str, float]:
        return self._objetivo(self.config.default)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    @property
    def active_emotion(self) -> str:
        return self._activa

    @property
    def phase(self) -> str:
        return self._fase

    def current_values(self) -> dict[str, float]:
        return {clave: round(valor, 4) for clave, valor in self._actual.items()}

    def set_emotion(self, nombre: str) -> None:
        """Inicia una transición hacia la emoción indicada."""
        clave = str(nombre).strip().lower()
        if clave not in self._specs:
            raise EmotionError(
                f"Emoción desconocida: {nombre!r}. "
                f"Disponibles: {', '.join(self._specs)}"
            )
        if clave == self._activa and self._fase in {"in", "hold"}:
            return
        self._spec = self._specs[clave]
        self._activa = clave
        self._desde = dict(self._actual)
        self._hasta = self._objetivo(clave)
        self._fase = "in"
        self._transcurrido = 0.0

    def set_neutral(self) -> None:
        self.set_emotion(self.config.default)

    def tick(self, dt_ms: float) -> dict[str, float]:
        """Avanza la animación y devuelve los valores actuales."""
        dt = max(0.0, float(dt_ms))
        if self._fase == "idle":
            return self.current_values()

        self._transcurrido += dt
        if self._fase == "in":
            duracion = max(1, self._spec.fade_in_ms)
            t = min(1.0, self._transcurrido / duracion)
            self._actual = _interpolar(self._desde, self._hasta, t)
            if t >= 1.0:
                self._fase = "hold"
                self._transcurrido = 0.0
                if self._spec.hold_ms <= 0:
                    self._iniciar_salida()
        elif self._fase == "hold":
            if self._transcurrido >= self._spec.hold_ms:
                self._iniciar_salida()
        elif self._fase == "out":
            duracion = max(1, self._spec.fade_out_ms)
            t = min(1.0, self._transcurrido / duracion)
            self._actual = _interpolar(self._desde, self._hasta, t)
            if t >= 1.0:
                self._fase = "idle"
                self._activa = self.config.default
                self._actual = dict(self._neutral)
        return self.current_values()

    def _iniciar_salida(self) -> None:
        self._desde = dict(self._actual)
        self._hasta = self._neutral_objetivo()
        self._fase = "out"
        self._transcurrido = 0.0

    def tick_seconds(self, dt_s: float) -> dict[str, float]:
        return self.tick(float(dt_s) * 1000.0)


def emociones_disponibles(config: EmotionConfig) -> Iterable[str]:
    return config.emotions.keys()
