"""Esquema, validación, deduplicación y partición del dataset de Nitshio.

Formato de una línea JSONL:

```json
{"user": "Hola, Nitshio.", "response": "Hmph. ¿Qué quieres?", "emotion": "angry"}
```

El módulo no llama a Ollama. Solo trabaja con archivos locales y es
determinista, para que los tests y los hashes sean reproducibles.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Límites razonables para que un ejemplo no rompa el contexto de 2048 tokens.
MIN_USER_CHARS = 1
MAX_USER_CHARS = 1000
MIN_RESPONSE_CHARS = 1
MAX_RESPONSE_CHARS = 4000
MAX_EMOTION_CHARS = 40


class DatasetError(RuntimeError):
    """El dataset no se pudo leer, validar o escribir."""


class DatasetExample(BaseModel):
    """Un ejemplo de conversación user -> assistant en formato Nitshio."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user: str = Field(min_length=MIN_USER_CHARS, max_length=MAX_USER_CHARS)
    response: str = Field(
        min_length=MIN_RESPONSE_CHARS, max_length=MAX_RESPONSE_CHARS
    )
    emotion: str = Field(min_length=1, max_length=MAX_EMOTION_CHARS)

    def assistant_json(self) -> str:
        """JSON compacto que el modelo debe aprender a devolver."""
        return json.dumps(
            {"response": self.response, "emotion": self.emotion},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def to_record(self) -> dict[str, str]:
        """Representación estable para guardar en JSONL."""
        return {
            "user": self.user,
            "response": self.response,
            "emotion": self.emotion,
        }

    def canonical(self) -> str:
        """Cadena canónica usada para hashes y deduplicación exacta."""
        return json.dumps(
            self.to_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True)
class ValidationIssue:
    """Un problema detectado en una línea del dataset."""

    line: int
    kind: str
    message: str
    raw: str = ""

    def __str__(self) -> str:
        etiqueta = "duplicado" if self.kind == "duplicate" else "error"
        return f"línea {self.line} [{etiqueta}]: {self.message}"


@dataclass
class DatasetLoadResult:
    """Resultado de cargar y validar un JSONL."""

    path: Path
    examples: list[DatasetExample] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    total_lines: int = 0
    blank_lines: int = 0
    duplicates: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.kind == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.kind != "error"]

    @property
    def ok(self) -> bool:
        """True si no hay errores de formato (los duplicados son avisos)."""
        return not self.errors

    def counts_by_emotion(self) -> dict[str, int]:
        return contar_emociones(self.examples)


@dataclass
class DatasetSplit:
    """Partición estratificada y reproducible del dataset."""

    train: list[DatasetExample]
    val: list[DatasetExample]
    test: list[DatasetExample]
    seed: int
    ratios: dict[str, float]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def manifest(self, *, source_hash: str | None = None) -> dict[str, Any]:
        """Metadatos que acompañan a los archivos de la partición."""
        return {
            "seed": self.seed,
            "ratios": dict(self.ratios),
            "counts": {
                "train": len(self.train),
                "val": len(self.val),
                "test": len(self.test),
                "total": self.total,
            },
            "by_emotion": {
                "train": contar_emociones(self.train),
                "val": contar_emociones(self.val),
                "test": contar_emociones(self.test),
            },
            "source_hash": source_hash,
            "hashes": {
                "train": hash_examples(self.train),
                "val": hash_examples(self.val),
                "test": hash_examples(self.test),
            },
        }


def _normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y solo caracteres alfanuméricos."""
    descompuesto = unicodedata.normalize("NFKD", str(texto).casefold())
    sin_acentos = "".join(
        caracter for caracter in descompuesto if not unicodedata.combining(caracter)
    )
    limpio = "".join(
        caracter if caracter.isalnum() else " " for caracter in sin_acentos
    )
    return " ".join(limpio.split())


def _firma(texto: str) -> frozenset[str]:
    """Firma de shingles de palabras para detectar casi duplicados."""
    tokens = _normalizar(texto).split()
    if not tokens:
        return frozenset()
    if len(tokens) <= 3:
        return frozenset({" ".join(tokens)})
    return frozenset(
        " ".join(tokens[indice : indice + 3])
        for indice in range(len(tokens) - 2)
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _clave_exacta(ejemplo: DatasetExample) -> str:
    return _normalizar(f"{ejemplo.user} || {ejemplo.response}")


def _leer_lineas(path: Path) -> list[tuple[int, str]]:
    try:
        texto = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"No se pudo leer {path}: {exc}") from exc
    return [
        (numero, linea.strip())
        for numero, linea in enumerate(texto.splitlines(), start=1)
        if linea.strip()
    ]


def validar_emocion(
    emocion: str, emociones_permitidas: Iterable[str]
) -> None:
    """Lanza ``ValueError`` si la emoción no está declarada."""
    permitidas = list(emociones_permitidas)
    if emocion not in permitidas:
        raise ValueError(
            f"Emoción inválida {emocion!r}. Permitidas: {', '.join(permitidas)}"
        )


def parse_example(
    data: Mapping[str, Any] | str,
    emociones_permitidas: Iterable[str],
    *,
    line: int = 0,
) -> DatasetExample:
    """Valida un objeto o una línea JSON y devuelve un ``DatasetExample``.

    Se usa también desde la UI y el generador sintético.
    """
    if isinstance(data, str):
        try:
            cargado = json.loads(data)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"JSON inválido: {exc}") from exc
        data = cargado
    if not isinstance(data, Mapping):
        raise DatasetError("Cada ejemplo debe ser un objeto JSON.")
    try:
        ejemplo = DatasetExample.model_validate(dict(data))
    except ValidationError as exc:
        detalles = "; ".join(
            f"{'.'.join(str(parte) for parte in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        prefijo = f"línea {line}: " if line else ""
        raise DatasetError(f"{prefijo}{detalles}") from exc
    try:
        validar_emocion(ejemplo.emotion, emociones_permitidas)
    except ValueError as exc:
        prefijo = f"línea {line}: " if line else ""
        raise DatasetError(f"{prefijo}{exc}") from exc
    return ejemplo


def _es_casi_duplicado(
    firma: frozenset[str],
    tokens: int,
    previos: Sequence[tuple[int, frozenset[str], int]],
    umbral: float,
) -> int | None:
    """Devuelve la línea del primer casi duplicado o ``None``."""
    for linea_previa, firma_previa, tokens_previos in previos:
        # Evita comparar textos de longitudes muy distintas.
        if tokens_previos and max(tokens, tokens_previos) > min(tokens, tokens_previos) * 3:
            continue
        if _jaccard(firma, firma_previa) >= umbral:
            return linea_previa
    return None


def load_dataset(
    path: Path | str,
    emociones_permitidas: Iterable[str],
    *,
    near_duplicate_threshold: float = 0.9,
    detect_near_duplicates: bool = True,
) -> DatasetLoadResult:
    """Carga un JSONL y valida formato, emoción y duplicados.

    Los duplicados exactos o casi exactos se descartan de ``examples`` y se
    reportan como avisos. Los errores de formato se reportan y la línea se
    omite; ``ok`` será ``False``.
    """
    ruta = Path(path)
    if not ruta.exists():
        raise DatasetError(f"No existe el dataset: {ruta}")
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold debe estar en (0, 1].")

    emociones = list(emociones_permitidas)
    resultado = DatasetLoadResult(path=ruta)
    exactos: dict[str, int] = {}
    firmas: list[tuple[int, frozenset[str], int]] = []

    lineas = _leer_lineas(ruta)
    total = len(ruta.read_text(encoding="utf-8").splitlines())
    resultado.total_lines = total
    resultado.blank_lines = max(0, total - len(lineas))

    for numero, linea in lineas:
        try:
            ejemplo = parse_example(linea, emociones, line=numero)
        except DatasetError as exc:
            resultado.issues.append(
                ValidationIssue(numero, "error", str(exc), raw=linea)
            )
            continue

        clave = _clave_exacta(ejemplo)
        if clave in exactos:
            resultado.duplicates += 1
            resultado.issues.append(
                ValidationIssue(
                    numero,
                    "duplicate",
                    f"duplicado exacto de la línea {exactos[clave]}.",
                    raw=linea,
                )
            )
            continue

        if detect_near_duplicates:
            texto = f"{ejemplo.user} {ejemplo.response}"
            firma = _firma(texto)
            tokens = len(_normalizar(texto).split())
            previa = _es_casi_duplicado(
                firma, tokens, firmas, near_duplicate_threshold
            )
            if previa is not None:
                resultado.duplicates += 1
                resultado.issues.append(
                    ValidationIssue(
                        numero,
                        "duplicate",
                        f"casi duplicado de la línea {previa}.",
                        raw=linea,
                    )
                )
                continue
            firmas.append((numero, firma, tokens))

        exactos[clave] = numero
        resultado.examples.append(ejemplo)

    return resultado


def save_jsonl(
    path: Path | str,
    examples: Iterable[DatasetExample | Mapping[str, Any]],
) -> Path:
    """Escribe ejemplos en UTF-8, uno por línea, sin reordenar."""
    ruta = Path(path)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    lineas: list[str] = []
    for ejemplo in examples:
        if isinstance(ejemplo, DatasetExample):
            registro = ejemplo.to_record()
        else:
            registro = dict(ejemplo)
        lineas.append(
            json.dumps(registro, ensure_ascii=False, separators=(",", ":"))
        )
    ruta.write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
    return ruta


def contar_emociones(examples: Iterable[DatasetExample]) -> dict[str, int]:
    """Cuenta ejemplos por emoción, ordenado alfabéticamente."""
    conteo: dict[str, int] = {}
    for ejemplo in examples:
        conteo[ejemplo.emotion] = conteo.get(ejemplo.emotion, 0) + 1
    return dict(sorted(conteo.items()))


def hash_examples(examples: Iterable[DatasetExample]) -> str:
    """SHA-256 estable del contenido, independiente del orden."""
    lineas = sorted(ejemplo.canonical() for ejemplo in examples)
    digest = hashlib.sha256()
    for linea in lineas:
        digest.update(linea.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_file(path: Path | str) -> str:
    """SHA-256 del archivo tal cual."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as fh:
            for bloque in iter(lambda: fh.read(65536), b""):
                digest.update(bloque)
    except OSError as exc:
        raise DatasetError(f"No se pudo leer {path}: {exc}") from exc
    return digest.hexdigest()


def _repartir(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Reparte ``n`` elementos con el método del mayor resto.

    Si hay al menos 3 ejemplos, se garantiza al menos uno en cada partición
    activa; el resto se reparte según las proporciones.
    """
    asignado = {clave: 0 for clave in ratios}
    if n <= 0:
        return asignado
    activos = [clave for clave, ratio in ratios.items() if ratio > 0]
    if not activos:
        return asignado
    suma = sum(ratios[clave] for clave in activos)
    crudos = {clave: n * ratios[clave] / suma for clave in activos}
    for clave in activos:
        asignado[clave] = int(math.floor(crudos[clave]))
    sobrante = n - sum(asignado.values())
    orden = sorted(
        activos,
        key=lambda clave: (
            crudos[clave] - math.floor(crudos[clave]),
            clave == "train",
            clave == "val",
        ),
        reverse=True,
    )
    for indice in range(max(0, sobrante)):
        asignado[orden[indice % len(orden)]] += 1

    # Cobertura mínima de validación/prueba en grupos pequeños.
    if n >= 3:
        for clave in activos:
            if asignado[clave] > 0:
                continue
            donante = max(
                activos,
                key=lambda candidato: (
                    asignado[candidato],
                    candidato == "train",
                ),
            )
            if asignado[donante] > 1:
                asignado[donante] -= 1
                asignado[clave] = 1
    return asignado


def split_dataset(
    examples: Iterable[DatasetExample],
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    """Divide por emoción de forma estratificada y determinista.

    Las proporciones se normalizan si no suman exactamente 1. Para grupos muy
    pequeños se prioriza que ``train`` no quede vacío.
    """
    ratios_originales = {"train": float(train), "val": float(val), "test": float(test)}
    if any(valor < 0 for valor in ratios_originales.values()):
        raise ValueError("Las proporciones no pueden ser negativas.")
    suma = sum(ratios_originales.values())
    if suma <= 0:
        raise ValueError("Al menos una proporción debe ser mayor que cero.")
    ratios = {clave: valor / suma for clave, valor in ratios_originales.items()}

    grupos: dict[str, list[DatasetExample]] = {}
    for ejemplo in examples:
        grupos.setdefault(ejemplo.emotion, []).append(ejemplo)

    rng = random.Random(seed)
    resultado: dict[str, list[DatasetExample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for emocion in sorted(grupos):
        grupo = list(grupos[emocion])
        rng.shuffle(grupo)
        reparto = _repartir(len(grupo), ratios)
        # Un grupo no puede quedar sin train salvo que no haya ejemplos.
        if len(grupo) > 0 and reparto["train"] == 0:
            donante = max(reparto, key=lambda clave: (reparto[clave], clave != "test"))
            if reparto[donante] > 0:
                reparto[donante] -= 1
                reparto["train"] += 1
        inicio = 0
        for clave in ("train", "val", "test"):
            cantidad = reparto[clave]
            resultado[clave].extend(grupo[inicio : inicio + cantidad])
            inicio += cantidad

    for clave in ("train", "val", "test"):
        rng.shuffle(resultado[clave])

    return DatasetSplit(
        train=resultado["train"],
        val=resultado["val"],
        test=resultado["test"],
        seed=seed,
        ratios={clave: round(valor, 6) for clave, valor in ratios.items()},
    )


def write_split(
    split: DatasetSplit,
    output_dir: Path | str,
    *,
    source_hash: str | None = None,
) -> dict[str, Path]:
    """Escribe ``train.jsonl``, ``val.jsonl``, ``test.jsonl`` y ``manifest.json``."""
    destino = Path(output_dir)
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {
        "train": save_jsonl(destino / "train.jsonl", split.train),
        "val": save_jsonl(destino / "val.jsonl", split.val),
        "test": save_jsonl(destino / "test.jsonl", split.test),
    }
    manifest = destino / "manifest.json"
    manifest.write_text(
        json.dumps(split.manifest(source_hash=source_hash), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    rutas["manifest"] = manifest
    return rutas


def merge_examples(
    *groups: Iterable[DatasetExample],
    near_duplicate_threshold: float = 0.9,
) -> list[DatasetExample]:
    """Une grupos y descarta duplicados conservando el primero."""
    resultado: list[DatasetExample] = []
    exactos: set[str] = set()
    firmas: list[tuple[int, frozenset[str], int]] = []
    for grupo in groups:
        for ejemplo in grupo:
            clave = _clave_exacta(ejemplo)
            if clave in exactos:
                continue
            texto = f"{ejemplo.user} {ejemplo.response}"
            firma = _firma(texto)
            tokens = len(_normalizar(texto).split())
            if _es_casi_duplicado(
                firma, tokens, firmas, near_duplicate_threshold
            ) is not None:
                continue
            exactos.add(clave)
            firmas.append((len(resultado) + 1, firma, tokens))
            resultado.append(ejemplo)
    return resultado
