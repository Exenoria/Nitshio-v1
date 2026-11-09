"""Herramienta de archivos con rutas permitidas, papelera y confirmación.

Reglas de seguridad (no negociables):

- Solo se opera dentro de ``tools.allowed_paths``.
- Se resuelven symlinks y se rechazan si ``allow_symlinks`` es falso.
- Borrar pide confirmación por defecto (``confirm_delete: true``).
- El borrado va a la papelera con ``send2trash`` salvo configuración explícita.
- Cada borrado se registra en un JSONL.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolContext, ToolError, ToolResult


class FilesTool(Tool):
    """Lista, lee, escribe, crea y borra archivos dentro del workspace."""

    name = "files"
    description = (
        "Gestiona archivos y carpetas dentro de las rutas permitidas: listar, "
        "leer, escribir, crear carpetas, mover y borrar (con confirmación y "
        "papelera por defecto)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list",
                    "read",
                    "write",
                    "mkdir",
                    "delete",
                    "exists",
                    "move",
                ],
                "description": "Operación a realizar.",
            },
            "path": {
                "type": "string",
                "description": "Ruta relativa al proyecto o absoluta dentro de allowed_paths.",
            },
            "destination": {
                "type": "string",
                "description": "Destino para action=move.",
            },
            "content": {
                "type": "string",
                "description": "Contenido para action=write.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Para list/delete de carpetas.",
            },
            "max_entries": {
                "type": "integer",
                "description": "Máximo de entradas al listar.",
            },
        },
        "required": ["action", "path"],
        "additionalProperties": False,
    }

    def __init__(self, context: ToolContext) -> None:
        self.context = context
        tools_cfg = context.config.get("tools") or {}
        security = tools_cfg.get("security") or {}
        self.confirm_delete = bool(security.get("confirm_delete", True))
        self.use_recycle_bin = bool(security.get("use_recycle_bin", True))
        self.delete_log = context.ruta(
            security.get("delete_log", "logs/deletions.jsonl")
        )
        self.max_read_bytes = int(security.get("max_read_bytes", 200_000))
        self.max_write_bytes = int(security.get("max_write_bytes", 200_000))
        self.allow_symlinks = bool(security.get("allow_symlinks", False))
        rutas = tools_cfg.get("allowed_paths") or ["workspace"]
        self.allowed_roots = [self._real(context.ruta(r)) for r in rutas]
        if not self.allowed_roots:
            raise ToolError("tools.allowed_paths está vacío; no se permite acceso.")

    # ------------------------------------------------------------------
    # Seguridad de rutas
    # ------------------------------------------------------------------
    @staticmethod
    def _real(path: Path) -> Path:
        return Path(os.path.realpath(str(path)))

    def _resolver(self, ruta: str, *, debe_existir: bool = False) -> Path:
        if not ruta or not str(ruta).strip():
            raise ToolError("La ruta está vacía.")
        candidato = Path(ruta)
        if not candidato.is_absolute():
            candidato = self.context.root / candidato
        if not self.allow_symlinks:
            for parte in [candidato, *candidato.parents]:
                try:
                    if parte.is_symlink():
                        raise ToolError(
                            f"Symlink no permitido en la ruta: {parte}"
                        )
                except OSError:
                    continue
        real = self._real(candidato)
        if not any(
            real == raiz or raiz in real.parents for raiz in self.allowed_roots
        ):
            raise ToolError(
                f"Ruta fuera de allowed_paths: {ruta!r}"
            )
        if debe_existir and not real.exists():
            raise ToolError(f"No existe: {ruta}")
        return real

    def _comprobar_tamano(self, path: Path, limite: int) -> None:
        if path.is_file() and path.stat().st_size > limite:
            raise ToolError(
                f"El archivo supera el límite de {limite} bytes: {path}"
            )

    # ------------------------------------------------------------------
    # Operaciones
    # ------------------------------------------------------------------
    def _listar(self, path: Path, recursive: bool, max_entries: int) -> ToolResult:
        if not path.exists():
            raise ToolError(f"No existe: {path}")
        entradas: list[dict[str, Any]] = []
        if path.is_file():
            entradas.append(self._info(path))
        else:
            iterador = path.rglob("*") if recursive else path.iterdir()
            for item in sorted(iterador):
                entradas.append(self._info(item))
                if len(entradas) >= max_entries:
                    break
        return ToolResult.exito(
            json.dumps(entradas, ensure_ascii=False, indent=2),
            entries=entradas,
            truncated=len(entradas) >= max_entries,
        )

    @staticmethod
    def _info(path: Path) -> dict[str, Any]:
        try:
            stat = path.stat()
            return {
                "name": path.name,
                "path": str(path),
                "type": "dir" if path.is_dir() else "file",
                "size": stat.st_size if path.is_file() else None,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        except OSError as exc:
            return {"path": str(path), "error": str(exc)}

    def _leer(self, path: Path) -> ToolResult:
        if not path.is_file():
            raise ToolError(f"No es un archivo: {path}")
        self._comprobar_tamano(path, self.max_read_bytes)
        datos = path.read_bytes()
        if b"\x00" in datos:
            raise ToolError("El archivo parece binario; no se puede leer como texto.")
        texto = datos.decode("utf-8", errors="replace")
        return ToolResult.exito(texto, path=str(path), bytes=len(datos))

    def _escribir(self, path: Path, content: str) -> ToolResult:
        datos = str(content).encode("utf-8")
        if len(datos) > self.max_write_bytes:
            raise ToolError(
                f"El contenido supera el límite de {self.max_write_bytes} bytes."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(datos)
        return ToolResult.exito(
            f"Escrito {path} ({len(datos)} bytes).",
            path=str(path),
            bytes=len(datos),
        )

    def _borrar(
        self, path: Path, recursive: bool, context: ToolContext
    ) -> ToolResult:
        if not path.exists():
            raise ToolError(f"No existe: {path}")
        if path.is_dir() and not recursive:
            raise ToolError(
                "Es una carpeta; usa recursive=true para borrarla."
            )
        if self.confirm_delete:
            if context.confirm is None:
                return ToolResult.fallo(
                    "Borrado cancelado: se requiere confirmación del usuario "
                    "(confirm_delete=true)."
                )
            if not context.confirm(f"¿Borrar definitivamente {path}?"):
                self._registrar_borrado(path, "cancelado", "confirmación rechazada")
                return ToolResult.fallo("Borrado cancelado por el usuario.")

        modo = "recycle_bin" if self.use_recycle_bin else "permanente"
        try:
            if self.use_recycle_bin:
                self._enviar_a_papelera(path)
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except Exception as exc:
            self._registrar_borrado(path, "error", str(exc), modo=modo)
            raise ToolError(f"No se pudo borrar {path}: {exc}") from exc
        self._registrar_borrado(path, "ok", "", modo=modo)
        return ToolResult.exito(
            f"Borrado ({modo}): {path}", path=str(path), mode=modo
        )

    def _enviar_a_papelera(self, path: Path) -> None:
        try:
            from send2trash import send2trash
        except ImportError as exc:  # pragma: no cover
            raise ToolError(
                "send2trash no está instalado; no se borra permanentemente "
                "por seguridad."
            ) from exc
        send2trash(str(path))

    def _registrar_borrado(
        self, path: Path, estado: str, detalle: str, *, modo: str = ""
    ) -> None:
        try:
            self.delete_log.parent.mkdir(parents=True, exist_ok=True)
            registro = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(path),
                "status": estado,
                "mode": modo,
                "detail": detalle,
            }
            with self.delete_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _mover(self, origen: Path, destino: Path) -> ToolResult:
        if not origen.exists():
            raise ToolError(f"No existe el origen: {origen}")
        if destino.exists():
            raise ToolError(f"El destino ya existe: {destino}")
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(origen), str(destino))
        return ToolResult.exito(
            f"Movido {origen} -> {destino}",
            source=str(origen),
            destination=str(destino),
        )

    # ------------------------------------------------------------------
    # Tool
    # ------------------------------------------------------------------
    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Ejecuta la acción y convierte los errores en resultados fallidos."""
        try:
            return self._ejecutar(arguments, context)
        except ToolError as exc:
            return ToolResult.fallo(str(exc))
        except OSError as exc:
            return ToolResult.fallo(f"Error del sistema de archivos: {exc}")

    def _ejecutar(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        accion = str(arguments.get("action", "")).strip().lower()
        ruta = str(arguments.get("path", ""))
        recursive = bool(arguments.get("recursive", False))
        max_entries = max(1, int(arguments.get("max_entries", 200)))

        if accion == "list":
            return self._listar(self._resolver(ruta, debe_existir=True), recursive, max_entries)
        if accion == "read":
            return self._leer(self._resolver(ruta, debe_existir=True))
        if accion == "write":
            return self._escribir(
                self._resolver(ruta), str(arguments.get("content", ""))
            )
        if accion == "mkdir":
            path = self._resolver(ruta)
            path.mkdir(parents=True, exist_ok=True)
            return ToolResult.exito(f"Carpeta creada: {path}", path=str(path))
        if accion == "delete":
            return self._borrar(
                self._resolver(ruta, debe_existir=True), recursive, context
            )
        if accion == "exists":
            path = self._resolver(ruta)
            return ToolResult.exito(
                f"{path} {'existe' if path.exists() else 'no existe'}",
                exists=path.exists(),
                path=str(path),
            )
        if accion == "move":
            destino = self._resolver(str(arguments.get("destination", "")))
            return self._mover(
                self._resolver(ruta, debe_existir=True), destino
            )
        raise ToolError(f"Acción desconocida: {accion!r}")


def create_tool(context: ToolContext) -> FilesTool:
    return FilesTool(context)
