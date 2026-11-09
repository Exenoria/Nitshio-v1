# Nitshio

Waifu IA **100 % local** para Windows: personalidad tsundere en español
mexicano, voz, memoria, herramientas, avatar VRM en VSeeFace y un pipeline de
entrenamiento pensado para la nube.

El cerebro, el STT, el TTS local, la memoria y el avatar funcionan sin internet.
Solo necesitan red la búsqueda web, la reproducción de YouTube y el proveedor
opcional Fish Audio.

> **Estado actual:** Paso 2 (pipeline completo) **cerrado**. La cadena
> STT → Ollama → TTS → avatar está validada con Moonshine, Piper, VB-Cable y
> VSeeFace reales, incluido el lip-sync por audio. El fine-tune (Paso 3)
> todavía no está implementado.

## Requisitos

- Windows 10/11.
- Python 3.11.
- [Ollama](https://ollama.com/) con los modelos declarados en
  `config/config.yaml`.
- **Opcional para voz:** VB-Cable, VSeeFace y un avatar `.vrm`.
- **Opcional para canciones:** `ffplay` (viene con FFmpeg) en el `PATH`.

No se instala PyTorch CUDA local. La inferencia usa Ollama y ONNX Runtime CPU
para Moonshine.

## Instalación rápida

```powershell
cd "$env:USERPROFILE\Nitshio"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Descarga la voz de Piper (solo la primera vez):

```powershell
.\.venv\Scripts\python.exe scripts\download_voice.py
```

Comprueba el entorno:

```powershell
.\.venv\Scripts\python.exe run.py --mode diagnose
```

## Uso

```powershell
# Chat de texto interactivo
.\.venv\Scripts\python.exe run.py --mode text

# Un solo mensaje, salida legible o JSON
.\.venv\Scripts\python.exe run.py --mode text --once "Hola, Nitshio"
.\.venv\Scripts\python.exe run.py --mode text --once "Hola" --json

# Texto con voz (TTS Piper)
.\.venv\Scripts\python.exe run.py --mode text --speak

# Conversación por micrófono (VAD + Moonshine)
.\.venv\Scripts\python.exe run.py --mode voice

# Modo llamada con barge-in
.\.venv\Scripts\python.exe run.py --mode call

# Prueba integral sin hablar al micrófono: Piper -> Moonshine -> Ollama -> TTS
.\.venv\Scripts\python.exe scripts\test_voice_roundtrip.py --no-tools --json
```

También puedes usar `run.bat`:

```powershell
.\run.bat --mode text --once "Hola"
```

### Banderas útiles

| Bandera | Efecto |
|---|---|
| `--config RUTA` | Usa otro YAML de configuración. |
| `--model TAG` | Sobrescribe `ollama.models.active`. |
| `--no-memory` | Desactiva ChromaDB para ese arranque. |
| `--no-tools` | Desactiva el router y las herramientas. |
| `--no-avatar` | No envía VMC a VSeeFace. |
| `--no-tts` | No sintetiza voz. |
| `--speak` | Lee las respuestas en modo texto. |
| `--json` | Salida JSON en `--once`. |
| `--debug` | Muestra intentos, memoria y herramientas. |

## Cómo funciona

```text
Texto / micrófono
      │
      ▼
STT Moonshine + VAD (solo voz)
      │
      ▼
Router qwen3.5:0.8b ──► herramienta (plugin) ──► resultado
      │
      ▼
LLM de personalidad (nitshio-vN en Ollama)
      │
      ├── memoria ChromaDB (RAG)
      ├── parser JSON + reintento + fallback
      ├── TTS Piper por chunks ──► CABLE Input
      └── avatar VMC ──► VSeeFace
```

- La personalidad vive en `brain/persona/nitshio.md` y los ejemplos en
  `brain/persona/few_shots.jsonl`.
- La salida del modelo es siempre
  `{"response": "...", "emotion": "..."}`; las emociones permitidas se leen de
  `config/emotions.yaml`.
- Los nombres de modelo, rutas, dispositivos y herramientas están en
  `config/config.yaml`. **No hay nombres de modelo hardcodeados.**
- Los secretos van en `.env` (ver `.env.example`), nunca en el YAML ni en el
  código.

## Memoria

ChromaDB persiste en `data/chroma`. Las colecciones son:

- `memories`: hechos y preferencias.
- `conversations`: turnos de conversación.
- `self_knowledge`: código y estructura del proyecto.
- `documentation`: documentación indexada.

Los embeddings se calculan con Ollama (`nomic-embed-text`) fuera de Chroma.
Así se evita que Chroma descargue su propio modelo oculto.

## Herramientas

Habilitadas por defecto en `config/config.yaml`:

| Herramienta | Función |
|---|---|
| `search_google` | Busca en Google CSE o DuckDuckGo. |
| `search_youtube` | Busca videos con yt-dlp. |
| `play_song` | Descarga el audio y lo reproduce con `ffplay`. |
| `open_google` | Abre una búsqueda en el navegador. |
| `open_youtube` | Abre YouTube o una URL válida. |
| `files` | Lee, escribe, lista, mueve y borra archivos. |
| `screen` | Captura la pantalla y opcionalmente la analiza. |
| `read_own_code` | Lee el código y la documentación propios. |

### Seguridad de archivos

- Solo se opera dentro de `tools.allowed_paths` (por defecto `workspace/`).
- Se resuelven symlinks y se rechazan por defecto.
- Borrar pide confirmación (`confirm_delete: true`).
- El borrado va a la papelera (`send2trash`) y se registra en
  `logs/deletions.jsonl`.
- Se limitan tamaño de lectura y escritura.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_docs.py
```

La suite cubre configuración, parser, persona, cliente Ollama, router, memoria,
herramientas, seguridad de archivos, VAD, STT/TTS, avatar, orquestador, pipeline
de voz, micrófono y el propio `check_docs.py` (**192 tests**).

## Documentación

- [Plan y fases](docs/PLAN.md)
- [Arquitectura](docs/ARCHITECTURE.md)
- [Decisiones](docs/DECISIONS.md)
- [Changelog](docs/CHANGELOG.md)
- [Entrenamiento](docs/TRAINING.md)
- [Extender Nitshio](docs/EXTENDING.md)
- [Solución de problemas](docs/TROUBLESHOOTING.md)
- [VSeeFace](docs/VSEEFACE.md)

## Limitaciones conocidas

- El modelo activo rinde ~15.6 tok/s en la T1200 de 4 GB; es usable, pero
  conviene mantener `num_ctx: 2048`.
- La calidad de personalidad del modelo base todavía es irregular (tercera
  persona, respuestas fuera de tono). Se ajustará con el fine-tune del Paso 3;
  el parser y el formato JSON ya están validados.
- La conversación humana por `--mode voice` queda como confirmación opcional:
  la cadena completa se prueba con `scripts/test_voice_roundtrip.py` y el
  micrófono real ya se validó por separado.
- El lip-sync de VSeeFace requiere el **proveedor viejo** (“Usar el modelo viejo
  para lip sync”). Con el nuevo, el audio llega pero la boca no se mueve. Ver
  [VSeeFace](docs/VSEEFACE.md).
- El fine-tune solo se ejecutará en Kaggle/Colab; la T1200 no entrena.
- Discord está previsto como adaptador opcional y aún no está conectado.
