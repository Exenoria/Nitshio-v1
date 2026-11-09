"""Tests de las herramientas de navegador, búsqueda, medios y pantalla."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pytest

from tools.base import ToolContext
from tools.open_google import OpenGoogleTool
from tools.open_youtube import OpenYoutubeTool
from tools.play_song import PlaySongTool
from tools.read_own_code import ReadOwnCodeTool
from tools.screen import ScreenTool
from tools.search_google import SearchGoogleTool
from tools.search_youtube import SearchYoutubeTool


def _context(tmp_path: Path, **extra: Any) -> ToolContext:
    config = {
        "tools": {
            "search": {"max_results": 3, "timeout_seconds": 1},
            "media": {
                "cache_dir": "data/cache/audio",
                "player": "ffplay",
                "max_duration_seconds": 300,
            },
            "screen": {
                "monitor": 1,
                "max_width": 640,
                "save_dir": "data/cache/screens",
            },
            "security": {"max_read_bytes": 10_000},
        },
        "ollama": {
            "host": "http://localhost:11434",
            "models": {"vision": "moondream:v2"},
            "request_timeout_seconds": 5,
        },
    }
    config["tools"].update(extra)
    return ToolContext(root=tmp_path, config=config)


# ---------------------------------------------------------------------------
# Navegador
# ---------------------------------------------------------------------------
def test_open_google_abre_url_codificada(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    abiertas: list[str] = []
    monkeypatch.setattr(
        "tools.open_google.webbrowser.open",
        lambda url: abiertas.append(url) or True,
    )
    resultado = OpenGoogleTool().execute(
        {"query": "hola mundo"}, _context(tmp_path)
    )
    assert resultado.ok
    assert abiertas == [
        "https://www.google.com/search?q=" + quote_plus("hola mundo")
    ]
    assert resultado.data["opened"] is True


def test_open_google_vacio(tmp_path: Path) -> None:
    resultado = OpenGoogleTool().execute({"query": "  "}, _context(tmp_path))
    assert not resultado.ok


def test_open_youtube_con_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    abiertas: list[str] = []
    monkeypatch.setattr(
        "tools.open_youtube.webbrowser.open",
        lambda url: abiertas.append(url) or True,
    )
    resultado = OpenYoutubeTool().execute(
        {"query": "lofi"}, _context(tmp_path)
    )
    assert resultado.ok
    assert abiertas[0].startswith(
        "https://www.youtube.com/results?search_query="
    )


def test_open_youtube_con_url_valida(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    abiertas: list[str] = []
    monkeypatch.setattr(
        "tools.open_youtube.webbrowser.open",
        lambda url: abiertas.append(url) or True,
    )
    resultado = OpenYoutubeTool().execute(
        {"url": "https://youtu.be/abc123"}, _context(tmp_path)
    )
    assert resultado.ok
    assert abiertas == ["https://youtu.be/abc123"]


def test_open_youtube_rechaza_dominio_externo(tmp_path: Path) -> None:
    resultado = OpenYoutubeTool().execute(
        {"url": "https://evil.example/video"}, _context(tmp_path)
    )
    assert not resultado.ok
    assert "YouTube" in (resultado.error or "")


def test_open_youtube_sin_argumentos(tmp_path: Path) -> None:
    resultado = OpenYoutubeTool().execute({}, _context(tmp_path))
    assert not resultado.ok


# ---------------------------------------------------------------------------
# Búsqueda web
# ---------------------------------------------------------------------------
class RespuestaFalsa:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


def test_search_google_usa_cse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CSE_KEY", "clave")
    monkeypatch.setenv("GOOGLE_CSE_CX", "cx")
    llamadas: list[dict[str, Any]] = []

    def get_falso(url: str, **kwargs: Any) -> RespuestaFalsa:
        llamadas.append({"url": url, **kwargs})
        return RespuestaFalsa(
            {
                "items": [
                    {
                        "title": "Resultado",
                        "link": "https://ejemplo.com",
                        "snippet": "Un fragmento",
                    }
                ]
            }
        )

    monkeypatch.setattr("tools.search_google.requests.get", get_falso)
    resultado = SearchGoogleTool(_context(tmp_path)).execute(
        {"query": "clima"}, _context(tmp_path)
    )
    assert resultado.ok
    assert resultado.data["results"][0]["url"] == "https://ejemplo.com"
    assert llamadas[0]["params"]["q"] == "clima"


def test_search_google_cae_a_ddg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_CSE_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)
    monkeypatch.setattr(
        SearchGoogleTool,
        "_ddg",
        lambda self, consulta, limite: [
            {
                "title": "DDG",
                "url": "https://ddg.example",
                "snippet": "texto",
            }
        ],
    )
    resultado = SearchGoogleTool(_context(tmp_path)).execute(
        {"query": "noticias"}, _context(tmp_path)
    )
    assert resultado.ok
    assert resultado.data["results"][0]["title"] == "DDG"


def test_search_google_sin_resultados_abre_navegador(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_CSE_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_CX", raising=False)
    monkeypatch.setattr(
        SearchGoogleTool, "_ddg", lambda self, consulta, limite: []
    )
    abiertas: list[str] = []
    monkeypatch.setattr(
        "tools.search_google.webbrowser.open",
        lambda url: abiertas.append(url) or True,
    )
    resultado = SearchGoogleTool(_context(tmp_path)).execute(
        {"query": "x"}, _context(tmp_path)
    )
    assert resultado.ok
    assert resultado.data["opened"] is True
    assert abiertas


def test_search_google_query_vacia(tmp_path: Path) -> None:
    resultado = SearchGoogleTool(_context(tmp_path)).execute(
        {"query": " "}, _context(tmp_path)
    )
    assert not resultado.ok


# ---------------------------------------------------------------------------
# Búsqueda de YouTube
# ---------------------------------------------------------------------------
def test_search_youtube_formatea_resultados(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        SearchYoutubeTool,
        "_buscar_ytdlp",
        lambda self, consulta, limite: [
            {
                "title": "Video",
                "url": "https://youtu.be/x",
                "channel": "Canal",
                "duration_s": 65,
            }
        ],
    )
    resultado = SearchYoutubeTool(_context(tmp_path)).execute(
        {"query": "música"}, _context(tmp_path)
    )
    assert resultado.ok
    assert "Video" in resultado.content
    assert "65 s" in resultado.content


def test_search_youtube_falla_y_abre_navegador(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def falla(self, consulta: str, limite: int):
        raise RuntimeError("sin red")

    monkeypatch.setattr(SearchYoutubeTool, "_buscar_ytdlp", falla)
    abiertas: list[str] = []
    monkeypatch.setattr(
        "tools.search_youtube.webbrowser.open",
        lambda url: abiertas.append(url) or True,
    )
    resultado = SearchYoutubeTool(_context(tmp_path)).execute(
        {"query": "x"}, _context(tmp_path)
    )
    assert not resultado.ok
    assert resultado.data["opened"] is True
    assert abiertas


# ---------------------------------------------------------------------------
# Reproducción de canciones
# ---------------------------------------------------------------------------
def test_play_song_rechaza_duracion_excesiva(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        PlaySongTool,
        "_buscar",
        lambda self, consulta: {
            "title": "Larga",
            "url": "https://youtu.be/larga",
            "duration_s": 999,
        },
    )
    monkeypatch.setattr(
        PlaySongTool,
        "_descargar",
        lambda self, entrada: pytest.fail("no debía descargar"),
    )
    resultado = PlaySongTool(_context(tmp_path)).execute(
        {"query": "larga", "max_duration_seconds": 60},
        _context(tmp_path),
    )
    assert not resultado.ok
    assert "límite" in (resultado.error or "")


def test_play_song_reproduce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        PlaySongTool,
        "_buscar",
        lambda self, consulta: {
            "title": "Corta",
            "url": "https://youtu.be/corta",
            "duration_s": 30,
        },
    )
    ruta = tmp_path / "audio.mp3"
    ruta.write_bytes(b"audio")
    monkeypatch.setattr(
        PlaySongTool, "_descargar", lambda self, entrada: ruta
    )
    reproducidas: list[Path] = []
    monkeypatch.setattr(
        PlaySongTool,
        "_reproducir",
        lambda self, path: reproducidas.append(path),
    )
    resultado = PlaySongTool(_context(tmp_path)).execute(
        {"query": "corta"}, _context(tmp_path)
    )
    assert resultado.ok
    assert reproducidas == [ruta]
    assert resultado.data["title"] == "Corta"


def test_play_song_sin_resultados(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PlaySongTool, "_buscar", lambda self, consulta: None)
    resultado = PlaySongTool(_context(tmp_path)).execute(
        {"query": "nada"}, _context(tmp_path)
    )
    assert not resultado.ok


# ---------------------------------------------------------------------------
# Pantalla
# ---------------------------------------------------------------------------
def test_screen_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ruta = tmp_path / "screen.png"
    ruta.write_bytes(b"png")
    tool = ScreenTool(_context(tmp_path))
    monkeypatch.setattr(tool, "_capturar", lambda monitor: ruta)
    resultado = tool.execute({"action": "capture"}, _context(tmp_path))
    assert resultado.ok
    assert resultado.data["path"] == str(ruta)


def test_screen_analyze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ruta = tmp_path / "screen.png"
    ruta.write_bytes(b"png")
    tool = ScreenTool(_context(tmp_path))
    monkeypatch.setattr(tool, "_capturar", lambda monitor: ruta)
    monkeypatch.setattr(
        tool, "_analizar", lambda path, prompt: "Hay un error en pantalla."
    )
    resultado = tool.execute(
        {"action": "analyze", "prompt": "¿qué ves?"}, _context(tmp_path)
    )
    assert resultado.ok
    assert "error" in resultado.content


def test_screen_error_de_captura(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = ScreenTool(_context(tmp_path))
    monkeypatch.setattr(
        tool,
        "_capturar",
        lambda monitor: (_ for _ in ()).throw(RuntimeError("sin monitor")),
    )
    resultado = tool.execute({"action": "capture"}, _context(tmp_path))
    assert not resultado.ok
    assert "sin monitor" in (resultado.error or "")


# ---------------------------------------------------------------------------
# Autoconocimiento
# ---------------------------------------------------------------------------
def test_read_own_code_lista_y_lee(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "modulo.py").write_text(
        "linea uno\nlinea dos\nlinea tres\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("hola\n", encoding="utf-8")
    (tmp_path / "binario.exe").write_bytes(b"\x00\x01")
    tool = ReadOwnCodeTool(_context(tmp_path))

    listado = tool.execute({"action": "list", "path": "."}, _context(tmp_path))
    assert listado.ok
    assert "src/modulo.py" in listado.data["files"]
    assert "binario.exe" not in listado.data["files"]

    lectura = tool.execute(
        {
            "action": "read",
            "path": "src/modulo.py",
            "start_line": 2,
            "end_line": 3,
        },
        _context(tmp_path),
    )
    assert lectura.ok
    assert "linea dos" in lectura.content
    assert "linea uno" not in lectura.content
    assert lectura.data["total_lines"] == 3


def test_read_own_code_rechaza_fuera(tmp_path: Path) -> None:
    fuera = tmp_path.parent / "fuera.py"
    fuera.write_text("secreto", encoding="utf-8")
    tool = ReadOwnCodeTool(_context(tmp_path))
    resultado = tool.execute(
        {"action": "read", "path": "../fuera.py"}, _context(tmp_path)
    )
    assert not resultado.ok
    assert "fuera" in (resultado.error or "")


def test_read_own_code_rechaza_extension(tmp_path: Path) -> None:
    (tmp_path / "binario.exe").write_bytes(b"\x00")
    tool = ReadOwnCodeTool(_context(tmp_path))
    resultado = tool.execute(
        {"action": "read", "path": "binario.exe"}, _context(tmp_path)
    )
    assert not resultado.ok
    assert "Extensión" in (resultado.error or "")
