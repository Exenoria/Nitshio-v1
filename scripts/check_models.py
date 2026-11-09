#!/usr/bin/env python
"""Comprueba Ollama, los modelos configurados y el rendimiento del hardware.

Uso:
    python scripts/check_models.py
    python scripts/check_models.py --pull
    python scripts/check_models.py --skip-perf
    python scripts/check_models.py --json

El script NO hardcodea nombres de modelo: todos salen de config/config.yaml.
Sale con código 0 si todo está presente y el rendimiento es aceptable, 1 si
falta Ollama o algún modelo, y 2 si el rendimiento es insuficiente.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
def cargar_config(path: Path) -> dict[str, Any]:
    """Carga el YAML de configuración y valida las claves mínimas."""
    if not path.exists():
        raise FileNotFoundError(f"No existe la configuración: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for key in ("ollama", "hardware"):
        if key not in data:
            raise ValueError(f"Falta la sección '{key}' en {path}")
    if "models" not in data["ollama"]:
        raise ValueError(f"Falta 'ollama.models' en {path}")
    return data


def cargar_emociones(config: dict[str, Any]) -> list[str]:
    """Lee las emociones permitidas desde el archivo configurado."""
    rel = config.get("persona", {}).get("emotions_file", "config/emotions.yaml")
    path = ROOT / rel
    if not path.exists():
        return ["neutral"]
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return list((data.get("emotions") or {}).keys()) or ["neutral"]


# ---------------------------------------------------------------------------
# Cliente Ollama
# ---------------------------------------------------------------------------
class OllamaError(RuntimeError):
    """Error de comunicación o de respuesta de Ollama."""


class Ollama:
    """Cliente HTTP mínimo para la API local de Ollama."""

    def __init__(self, host: str, timeout: float = 180.0) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        try:
            resp = self.session.get(
                self._url(path), timeout=timeout or self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise OllamaError(str(exc)) from exc

    def post(
        self, path: str, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any]:
        try:
            resp = self.session.post(
                self._url(path),
                json=payload,
                timeout=timeout or self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise OllamaError(str(exc)) from exc

    def version(self) -> str:
        return str(self.get("/api/version", timeout=10).get("version", "desconocida"))

    def tags(self) -> list[dict[str, Any]]:
        return list(self.get("/api/tags", timeout=15).get("models", []))

    def loaded(self) -> list[dict[str, Any]]:
        return list(self.get("/api/ps", timeout=15).get("models", []))

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        format_schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str = "5m",
        think: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "keep_alive": keep_alive,
            "options": options or {},
        }
        if format_schema is not None:
            payload["format"] = format_schema
        return self.post("/api/generate", payload)

    def embeddings(self, model: str, prompt: str) -> dict[str, Any]:
        return self.post("/api/embeddings", {"model": model, "prompt": prompt})


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def info_gpu() -> dict[str, Any] | None:
    """Devuelve datos de la primera GPU NVIDIA usando nvidia-smi, si existe."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        if not out:
            return None
        first = out.splitlines()[0]
        name, total, used, util = [p.strip() for p in first.split(",")]
        return {
            "name": name,
            "vram_total_mb": int(float(total)),
            "vram_used_mb": int(float(used)),
            "util_percent": int(float(util)),
        }
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def info_ram() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 1),
        "used_gb": round(vm.used / (1024**3), 1),
        "percent": vm.percent,
    }


def _normalizar_tag(tag: str) -> str:
    """Ollama agrega ':latest' a los modelos sin tag explícito."""
    return tag if ":" in tag else f"{tag}:latest"


def _esta_instalado(modelo: str, tags: list[str]) -> bool:
    """Compara un modelo con los tags instalados aceptando ':latest' implícito."""
    objetivo = _normalizar_tag(modelo)
    return any(_normalizar_tag(t) == objetivo for t in tags)


def parar_modelo(modelo: str) -> None:
    """Descarga un modelo de Ollama para medir el siguiente sin competencia."""
    exe = shutil.which("ollama")
    if not exe:
        return
    try:
        subprocess.run(
            [exe, "stop", modelo],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        pass


def modelos_cargados(ollama: Ollama) -> list[dict[str, Any]]:
    """Resume qué modelos tiene cargados Ollama y si están en GPU."""
    resultado = []
    try:
        for m in ollama.loaded():
            size = int(m.get("size", 0) or 0)
            vram = int(m.get("size_vram", 0) or 0)
            resultado.append(
                {
                    "name": m.get("name", "?"),
                    "size_mb": round(size / (1024**2), 1),
                    "vram_mb": round(vram / (1024**2), 1),
                    "processor": "GPU" if vram > 0 else "CPU",
                }
            )
    except OllamaError:
        pass
    return resultado


# ---------------------------------------------------------------------------
# Pruebas de rendimiento
# ---------------------------------------------------------------------------
def _tps(data: dict[str, Any], wall_seconds: float) -> float:
    """Calcula tokens/s usando eval_duration; si no existe, usa el wall time."""
    eval_count = int(data.get("eval_count", 0) or 0)
    eval_ns = int(data.get("eval_duration", 0) or 0)
    if eval_count and eval_ns > 0:
        return eval_count / (eval_ns / 1e9)
    if eval_count and wall_seconds > 0:
        return eval_count / wall_seconds
    return 0.0


def _json_valido(texto: str) -> tuple[bool, str | None]:
    """Comprueba que el texto sea un objeto JSON parseable."""
    try:
        valor = json.loads(texto)
    except (json.JSONDecodeError, TypeError) as exc:
        return False, str(exc)
    if not isinstance(valor, dict):
        return False, "la raíz no es un objeto"
    return True, None


def probar_modelo(
    ollama: Ollama,
    rol: str,
    modelo: str,
    *,
    schema: dict[str, Any] | None,
    prompt: str,
    options: dict[str, Any],
    keep_alive: str,
    validar_json: bool,
) -> dict[str, Any]:
    """Mide carga, latencia y tokens/s de un modelo con una petición real."""
    # Calentamiento: carga el modelo y deja lista la GPU. No se mide.
    t0 = time.perf_counter()
    calentamiento = ollama.generate(
        modelo,
        "Responde solo: OK",
        options={"num_predict": 4, "temperature": 0},
        keep_alive=keep_alive,
    )
    wall_calentamiento = time.perf_counter() - t0
    load_ns = int(calentamiento.get("load_duration", 0) or 0)

    # Medición real.
    t0 = time.perf_counter()
    data = ollama.generate(
        modelo,
        prompt,
        format_schema=schema,
        options=options,
        keep_alive=keep_alive,
    )
    wall = time.perf_counter() - t0
    texto = str(data.get("response", ""))
    tps = _tps(data, wall)
    json_ok, json_error = (None, None)
    if validar_json:
        json_ok, json_error = _json_valido(texto)

    return {
        "rol": rol,
        "modelo": modelo,
        "carga_s": round(wall_calentamiento, 2),
        "load_duration_s": round(load_ns / 1e9, 2),
        "latencia_s": round(wall, 2),
        "tokens_salida": int(data.get("eval_count", 0) or 0),
        "tokens_entrada": int(data.get("prompt_eval_count", 0) or 0),
        "tokens_por_segundo": round(tps, 1),
        "json_valido": json_ok,
        "json_error": json_error,
        "respuesta": texto[:400],
    }


def probar_embeddings(ollama: Ollama, modelo: str) -> dict[str, Any]:
    """Mide la latencia y dimensionalidad del modelo de embeddings."""
    texto = "Nitshio recuerda que su usuario prefiere café por la mañana."
    ollama.embeddings(modelo, "calentamiento")
    t0 = time.perf_counter()
    data = ollama.embeddings(modelo, texto)
    wall = time.perf_counter() - t0
    emb = data.get("embedding") or []
    return {
        "modelo": modelo,
        "latencia_s": round(wall, 2),
        "dimensiones": len(emb),
    }


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------
def imprimir_informe(
    *,
    ollama_ok: bool,
    version: str,
    tags: list[str],
    roles: dict[str, str],
    faltantes: list[str],
    gpu: dict[str, Any] | None,
    ram: dict[str, Any],
    cargados: list[dict[str, Any]],
    perfiles: list[dict[str, Any]],
    embeddings: dict[str, Any] | None,
    veredicto: str,
    sugerencias: list[str],
) -> None:
    print("=" * 72)
    print("  NITSHIO — comprobación de modelos y hardware")
    print("=" * 72)
    print(f"Ollama: {'OK' if ollama_ok else 'NO DISPONIBLE'}"
          f"{f' (v{version})' if version else ''}")
    print(f"Modelos instalados: {', '.join(tags) if tags else '(ninguno)'}")
    print()
    print("Roles configurados:")
    for rol, modelo in roles.items():
        estado = "OK" if _esta_instalado(modelo, tags) else "FALTA"
        print(f"  - {rol:12s} {modelo:48s} [{estado}]")
    if faltantes:
        print()
        print("Faltantes:")
        for modelo in faltantes:
            print(f"  - {modelo}")
        print("  Descárgalos con:  python scripts/check_models.py --pull")
    print()
    if gpu:
        print(
            f"GPU: {gpu['name']} | VRAM {gpu['vram_used_mb']}/"
            f"{gpu['vram_total_mb']} MB | uso {gpu['util_percent']}%"
        )
    else:
        print("GPU: no detectada por nvidia-smi (se usará CPU/RAM).")
    print(
        f"RAM: {ram['used_gb']}/{ram['total_gb']} GB "
        f"({ram['percent']}% en uso)"
    )
    if cargados:
        print("Modelos cargados ahora mismo:")
        for m in cargados:
            print(
                f"  - {m['name']}: {m['size_mb']} MB, "
                f"VRAM {m['vram_mb']} MB -> {m['processor']}"
            )
    print()
    if perfiles:
        print("Rendimiento medido (segunda petición, modelo caliente):")
        for p in perfiles:
            extra = ""
            if p.get("json_valido") is not None:
                extra = " | JSON: " + ("válido" if p["json_valido"] else "inválido")
            print(
                f"  - {p['rol']:12s} {p['modelo']:48s} "
                f"{p['tokens_por_segundo']:6.1f} tok/s | "
                f"{p['latencia_s']:6.2f} s{extra}"
            )
    if embeddings:
        print(
            f"  - embeddings   {embeddings['modelo']:48s} "
            f"{embeddings['latencia_s']:6.2f} s | "
            f"{embeddings['dimensiones']} dims"
        )
    print()
    print(f"VEREDICTO: {veredicto}")
    if sugerencias:
        print("Sugerencias:")
        for s in sugerencias:
            print(f"  - {s}")
    print("=" * 72)


def veredicto_y_sugerencias(
    *,
    faltantes: list[str],
    perfiles: list[dict[str, Any]],
    embeddings: dict[str, Any] | None,
    config: dict[str, Any],
    gpu: dict[str, Any] | None,
    ram: dict[str, Any],
    cargados: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Calcula el veredicto global y las alternativas más ligeras."""
    hw = config.get("hardware", {})
    min_primary = float(hw.get("min_tokens_per_second", 8))
    comfortable = float(hw.get("comfortable_tokens_per_second", 20))
    min_router = float(hw.get("min_router_tokens_per_second", 20))
    max_vram = float(hw.get("max_vram_percent", 95))
    max_ram = float(hw.get("max_ram_percent", 90))

    if faltantes:
        return "NO SOPORTADO — faltan modelos", [
            "Ejecuta `python scripts/check_models.py --pull` para descargarlos."
        ]

    problemas: list[str] = []
    sugerencias: list[str] = []
    primary = next((p for p in perfiles if p["rol"] == "active"), None)
    if primary is None:
        primary = next((p for p in perfiles if p["rol"] == "base"), None)
    router = next((p for p in perfiles if p["rol"] == "router"), None)

    if primary:
        tps = primary["tokens_por_segundo"]
        if tps < min_primary:
            problemas.append(
                f"El modelo principal va a {tps:.1f} tok/s "
                f"(mínimo {min_primary:.0f})."
            )
            alternativas = hw.get("lighter_primary_alternatives", [])
            if alternativas:
                sugerencias.append(
                    "Alternativas más ligeras para el modelo principal: "
                    + ", ".join(str(a) for a in alternativas)
                )
        elif tps < comfortable:
            problemas.append(
                f"El modelo principal va a {tps:.1f} tok/s: usable, pero ajustado."
            )

    if router:
        tps = router["tokens_por_segundo"]
        if tps < min_router:
            problemas.append(
                f"El router va a {tps:.1f} tok/s (mínimo {min_router:.0f})."
            )
            alternativas = hw.get("lighter_router_alternatives", [])
            if alternativas:
                sugerencias.append(
                    "Alternativas para el router: "
                    + ", ".join(str(a) for a in alternativas)
                )

    if embeddings and embeddings["latencia_s"] > 2.0:
        sugerencias.append(
            "Los embeddings tardan más de 2 s; considera un modelo de embeddings "
            "más pequeño o reutilizar el mismo proveedor con cache."
        )

    if gpu:
        vram_pct = 100 * gpu["vram_used_mb"] / max(gpu["vram_total_mb"], 1)
        if vram_pct > max_vram:
            problemas.append(
                f"La VRAM está al {vram_pct:.0f}% (límite {max_vram:.0f}%)."
            )
            sugerencias.append(
                "Cierra aplicaciones con GPU o usa una cuantización más pequeña."
            )
    else:
        sugerencias.append(
            "No hay GPU NVIDIA visible para Ollama; la inferencia irá más lenta "
            "en CPU. Revisa los drivers y `ollama ps`."
        )

    if ram["percent"] > max_ram:
        problemas.append(
            f"La RAM está al {ram['percent']}% (límite {max_ram:.0f}%)."
        )

    if cargados and all(m["processor"] == "CPU" for m in cargados):
        sugerencias.append(
            "Ollama tiene los modelos en CPU. Si hay GPU NVIDIA, revisa "
            "`nvidia-smi` y la variable OLLAMA_GPU_OVERHEAD."
        )

    if any(m["processor"] == "GPU" for m in cargados) and gpu:
        sugerencias.append(
            "En 4 GB de VRAM no mantengas router y modelo principal cargados a la "
            "vez; usa `OLLAMA_MAX_LOADED_MODELS=1` o un keep_alive corto."
        )

    if not problemas:
        return "SOPORTADO — rendimiento aceptable", sugerencias
    if primary and primary["tokens_por_segundo"] >= min_primary:
        return "AJUSTADO — funciona, pero con margen justo", problemas + sugerencias
    return "NO SOPORTADO — rendimiento insuficiente", problemas + sugerencias


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comprueba Ollama, modelos y rendimiento para Nitshio."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pull", action="store_true", help="Descarga los modelos que falten."
    )
    parser.add_argument(
        "--skip-perf",
        action="store_true",
        help="Solo comprueba presencia; no mide latencia ni tokens/s.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Salida JSON (para automatización)."
    )
    args = parser.parse_args(argv)

    try:
        config = cargar_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR de configuración: {exc}", file=sys.stderr)
        return 1

    host = config["ollama"].get("host", "http://localhost:11434")
    timeout = float(config["ollama"].get("request_timeout_seconds", 180))
    keep_alive = str(config["ollama"].get("keep_alive", "5m"))
    modelos = config["ollama"]["models"]
    emociones = cargar_emociones(config)
    ollama = Ollama(host, timeout)

    roles = {
        "base": modelos.get("base", ""),
        "active": modelos.get("active", modelos.get("base", "")),
        "router": modelos.get("router", ""),
        "embeddings": modelos.get("embeddings", ""),
        "vision": modelos.get("vision", ""),
    }
    roles = {k: v for k, v in roles.items() if v}

    # 1. Servicio y tags.
    try:
        version = ollama.version()
        tags_data = ollama.tags()
        tags = [m.get("name", "") for m in tags_data]
    except OllamaError as exc:
        print("=" * 72)
        print("  NITSHIO — comprobación de modelos y hardware")
        print("=" * 72)
        print(f"ERROR: no se pudo hablar con Ollama en {host}")
        print(f"Detalle: {exc}")
        print()
        print("Qué hacer:")
        print("  1. Abre Ollama o ejecuta `ollama serve` en otra terminal.")
        print("  2. Comprueba que http://localhost:11434/api/tags responde.")
        print("  3. Si usas otro puerto, cámbialo en config/config.yaml.")
        return 1

    faltantes = [m for m in roles.values() if not _esta_instalado(m, tags)]
    # El modelo de visión es opcional; si falta, no bloquea el resto.
    faltantes_obligatorios = [
        m for m in faltantes if m != roles.get("vision")
    ]

    # 2. Descarga opcional.
    if args.pull and faltantes:
        if not shutil.which("ollama"):
            print("ERROR: `ollama` no está en PATH; no se pueden descargar modelos.")
            return 1
        for modelo in faltantes:
            print(f"Descargando {modelo} ...")
            try:
                subprocess.run(["ollama", "pull", modelo], check=True)
            except subprocess.CalledProcessError as exc:
                print(f"ERROR al descargar {modelo}: {exc}")
                return 1
        try:
            tags = [m.get("name", "") for m in ollama.tags()]
        except OllamaError as exc:
            print(f"ERROR al refrescar modelos: {exc}")
            return 1
        faltantes = [m for m in roles.values() if not _esta_instalado(m, tags)]
        faltantes_obligatorios = [
            m for m in faltantes if m != roles.get("vision")
        ]

    # 3. Hardware.
    gpu = info_gpu()
    ram = info_ram()

    perfiles: list[dict[str, Any]] = []
    embeddings_info: dict[str, Any] | None = None

    if not args.skip_perf and not faltantes_obligatorios:
        response_schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "emotion": {"type": "string", "enum": emociones},
            },
            "required": ["response", "emotion"],
            "additionalProperties": False,
        }
        router_schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "arguments": {"type": "object"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["tool", "arguments", "confidence", "reason"],
            "additionalProperties": False,
        }

        opciones_primary = dict(
            config["ollama"].get("options", {}).get("primary", {})
        )
        opciones_primary["num_predict"] = min(
            int(opciones_primary.get("num_predict", 512)), 64
        )
        opciones_router = dict(
            config["ollama"].get("options", {}).get("router", {})
        )
        # El router necesita espacio para argumentos + razón; con 48 tokens
        # el JSON queda truncado y el test da un falso negativo.
        opciones_router["num_predict"] = min(
            int(opciones_router.get("num_predict", 256)), 160
        )

        probados: set[str] = set()
        # El modelo activo se mide primero y se descarga antes del siguiente:
        # en 4 GB de VRAM, dos modelos compitiendo falsean el resultado.
        for rol in ("active", "base"):
            modelo = roles.get(rol)
            if not modelo or modelo in probados:
                continue
            probados.add(modelo)
            try:
                perfiles.append(
                    probar_modelo(
                        ollama,
                        rol,
                        modelo,
                        schema=response_schema,
                        prompt=(
                            "Responde únicamente con un objeto JSON que tenga "
                            'las claves "response" (texto) y "emotion" (una de: '
                            + ", ".join(emociones)
                            + "). Di una frase corta en español."
                        ),
                        options=opciones_primary,
                        keep_alive=keep_alive,
                        validar_json=True,
                    )
                )
            except OllamaError as exc:
                print(f"ERROR midiendo {rol} ({modelo}): {exc}")
            finally:
                parar_modelo(modelo)

        if roles.get("router") and roles["router"] not in probados:
            try:
                perfiles.append(
                    probar_modelo(
                        ollama,
                        "router",
                        roles["router"],
                        schema=router_schema,
                        prompt=(
                            "Eres un router de herramientas. El usuario dice: "
                            '"Hola, ¿cómo estás?". Decide si hace falta una '
                            'herramienta. Devuelve JSON con las claves "tool" '
                            '(usa "none" si no hace falta), "arguments" (objeto), '
                            '"confidence" (0 a 1) y "reason" (texto).'
                        ),
                        options=opciones_router,
                        keep_alive=keep_alive,
                        validar_json=True,
                    )
                )
            except OllamaError as exc:
                print(f"ERROR midiendo router ({roles['router']}): {exc}")
            finally:
                parar_modelo(roles["router"])

        if roles.get("embeddings") and roles["embeddings"] not in faltantes:
            try:
                embeddings_info = probar_embeddings(
                    ollama, roles["embeddings"]
                )
            except OllamaError as exc:
                print(f"ERROR midiendo embeddings: {exc}")
            finally:
                parar_modelo(roles["embeddings"])

    cargados = modelos_cargados(ollama)
    veredicto, sugerencias = veredicto_y_sugerencias(
        faltantes=faltantes_obligatorios,
        perfiles=perfiles,
        embeddings=embeddings_info,
        config=config,
        gpu=gpu,
        ram=ram,
        cargados=cargados,
    )

    if args.json:
        salida = {
            "ollama": {"ok": True, "version": version, "host": host},
            "roles": roles,
            "instalados": tags,
            "faltantes": faltantes,
            "hardware": {"gpu": gpu, "ram": ram, "cargados": cargados},
            "perfiles": perfiles,
            "embeddings": embeddings_info,
            "veredicto": veredicto,
            "sugerencias": sugerencias,
        }
        print(json.dumps(salida, ensure_ascii=False, indent=2))
    else:
        imprimir_informe(
            ollama_ok=True,
            version=version,
            tags=tags,
            roles=roles,
            faltantes=faltantes,
            gpu=gpu,
            ram=ram,
            cargados=cargados,
            perfiles=perfiles,
            embeddings=embeddings_info,
            veredicto=veredicto,
            sugerencias=sugerencias,
        )

    if faltantes_obligatorios:
        return 1
    if veredicto.startswith("NO SOPORTADO"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
