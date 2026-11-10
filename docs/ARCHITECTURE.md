# Arquitectura — Nitshio

> Documento vivo. Describe los módulos implementados y cómo se conectan.
> Última actualización: 2026-11-09 (Paso 3, GGUF y Colab).

## 1. Principios

1. **Todo local por defecto.** Ollama, Moonshine, Piper, ChromaDB y VMC corren
   en la máquina. La red solo se usa para búsquedas, YouTube y Fish Audio
   opcional.
2. **Configuración central.** Nombres de modelo, rutas, dispositivos, umbrales
   y herramientas viven en `config/config.yaml`. El código no hardcodea tags.
3. **Degradación elegante.** Si memoria, router, avatar o TTS fallan, el turno
   de texto sigue funcionando y se registra el aviso.
4. **Contratos pequeños.** STT, TTS, memoria y herramientas tienen interfaces
   mínimas para poder cambiar de proveedor sin tocar el orquestador.
5. **Seguridad antes que comodidad.** Las herramientas de archivos validan
   rutas, symlinks, tamaño, confirmación y papelera.

## 2. Flujo de un turno

```text
                 ┌──────────────────────────────┐
                 │  run.py / TextPipeline        │
                 └──────────────┬───────────────┘
                                │ texto
                                ▼
                 ┌──────────────────────────────┐
                 │  Orchestrator.process()       │
                 └──────────────┬───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 Memoria (RAG)           Router de tools         Historial
 ChromaDB                qwen3.5:0.8b            (memoria de sesión)
        │                       │
        │                       ▼
        │               ToolRegistry.execute()
        │                       │
        └───────────┬───────────┘
                    ▼
        construir_mensajes(persona + contexto)
                    │
                    ▼
        OllamaClient.chat_structured()
        JSON Schema + reintentos + reparación + fallback
                    │
                    ▼
        ParsedResponse {response, emotion}
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
   TTSManager   Avatar VMC   Memoria (guardar)
   Piper/Fish   VSeeFace     conversación
```

### Modo voz

```text
Micrófono ──► Microphone.frames() ──► VADSegmenter ──► utterance
                                                         │
                                                         ▼
                                              MoonshineProvider.transcribe()
                                                         │
                                                         ▼
                                              Orchestrator.process()
                                                         │
                                                         ▼
                                              TTSManager.speak()
```

En modo llamada, `MicrophoneListener` captura en un hilo y deja utterances en
una cola. Mientras el TTS habla en otro hilo, el bucle sigue leyendo la cola:
si llega una utterance nueva, `TTSManager.stop()` cancela la locución
(barge-in) y se procesa el mensaje nuevo.

## 3. Módulos

### `brain/`

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Carga YAML + `.env`; resuelve rutas y `OLLAMA_HOST`. |
| `llm/ollama_client.py` | Única puerta a Ollama. `chat()` y `chat_structured()` con JSON Schema, reintentos y fallback. |
| `llm/parser.py` | Extrae JSON de texto, repara comas/comillas/saltos, valida con Pydantic y comprueba emociones. |
| `llm/orchestrator.py` | Une memoria, router, herramientas, LLM, avatar y TTS. |
| `persona/loader.py` | Carga prompt, few-shots y emociones; construye mensajes. |
| `router/tool_router.py` | Pide al router una decisión `{tool, arguments, confidence, reason}` y la valida. |

**Contrato de salida del LLM:**

```json
{"response": "texto en personaje", "emotion": "neutral"}
```

`response` tiene entre 1 y 4000 caracteres. `emotion` debe estar declarada en
`config/emotions.yaml`. Cualquier otro campo se rechaza (`extra="forbid"`).

### `stt/`

| Archivo | Responsabilidad |
|---|---|
| `base.py` | `STTProvider`, `Transcript`, normalización de audio y fábrica `crear_stt()`. |
| `vad.py` | Segmentación por WebRTC VAD con fallback por energía. |
| `microphone.py` | Stream de `sounddevice`, frames mono y utterances. |
| `moonshine_provider.py` | Adaptador real a `moonshine-voice` en español. |
| `whisper_provider.py` | Fallback opcional con faster-whisper. |

Todo proveedor recibe audio mono `float32` en `[-1, 1]` y devuelve un
`Transcript` con texto normalizado.

### `tts/`

| Archivo | Responsabilidad |
|---|---|
| `base.py` | `TTSProvider`, `AudioChunk` y fábrica `crear_tts()`. |
| `chunker.py` | Divide el texto respetando frases y hace el primer chunk más corto para bajar la latencia. |
| `piper_provider.py` | Sintetiza WAV con Piper. |
| `fish_provider.py` | Proveedor opcional Fish Audio (requiere `FISH_API_KEY`). |
| `player.py` | Reproduce WAV con `sounddevice`, resuelve el dispositivo por nombre/índice y soporta `stop()`. |
| `manager.py` | Orquesta chunker + proveedor + reproductor y expone `speak()`/`stop()`. |

### `memory/`

| Archivo | Responsabilidad |
|---|---|
| `embeddings.py` | Embeddings con Ollama (`/api/embed`, fallback `/api/embeddings`). |
| `store.py` | ChromaDB persistente, CRUD, búsqueda y construcción de contexto RAG. |

Chroma solo almacena vectores; los embeddings se calculan externamente con
Ollama para no depender de la función por defecto de Chroma.

### `tools/`

| Archivo | Responsabilidad |
|---|---|
| `base.py` | `Tool`, `ToolContext`, `ToolResult` y validación básica de JSON Schema. |
| `registry.py` | Carga dinámica de plugins habilitados y ejecución sin propagar excepciones. |
| `files.py` | Operaciones de archivos con `allowed_paths`, symlinks, papelera, confirmación y log. |
| `search_google.py` | Google CSE si hay claves; si no, DuckDuckGo; si falla, abre el navegador. |
| `search_youtube.py` | Búsqueda de videos con yt-dlp. |
| `play_song.py` | Busca, descarga audio y reproduce con `ffplay`, con límite de duración. |
| `open_google.py` / `open_youtube.py` | Abren URLs en el navegador. |
| `screen.py` | Captura con `mss` y análisis opcional con el modelo de visión. |
| `read_own_code.py` | Lectura de código/documentación con extensión y tamaño limitados. |

Un plugin nuevo solo necesita:

```python
class MiTool(Tool):
    name = "mi_tool"
    description = "..."
    parameters = {...}

    def execute(self, arguments, context):
        return ToolResult.exito("...")

def create_tool(context):
    return MiTool(context)
```

y añadir `mi_tool` a `tools.enabled`.

### `avatar/`

| Archivo | Responsabilidad |
|---|---|
| `emotions.py` | Carga `config/emotions.yaml`; máquina de estados fade in/hold/fade out. |
| `vmc.py` | Cliente OSC y `AvatarController`; envía blendshapes a VSeeFace. |
| `lip_sync.py` | Utilidad opcional de amplitud → visema (no se activa sola). |

El lip-sync activo es el de VSeeFace leyendo `CABLE Output` con
`AudioLipSync=1` y `OldLipSyncProvider=1`. Los visemas VMC no se usan porque en
las pruebas no movieron la boca; ver `docs/VSEEFACE.md`.

Mensajes VMC:

```text
/VMC/Ext/Blend/Val   <nombre_blendshape> <valor 0..1>
/VMC/Ext/Blend/Apply
```

### `pipeline/`

| Archivo | Responsabilidad |
|---|---|
| `text.py` | Bucle de chat, comandos `/salir`, `/limpiar`, `/herramientas`, `/memoria`, `/emocion`. |
| `voice.py` | `VoicePipeline`, `MicrophoneListener` y `CallMode` con barge-in. |

### `training/`

| Archivo | Responsabilidad |
|---|---|
| `schema.py` | Esquema Pydantic estricto, validación, deduplicación, hashes y split estratificado. |
| `validate_dataset.py` | CLI de validación, conteos, duplicados y partición train/val/test. |
| `generate_synthetic.py` | Generación de lotes por emoción con Ollama, filtrado, deduplicación y manifiesto. |
| `formatting.py` | Conversión de ejemplos a mensajes y texto de entrenamiento. |
| `finetune_qlora.py` | QLoRA con Unsloth/TRL en la nube; `--dry-run` funciona en local. |
| `export_gguf.py` | Fusión de adaptadores LoRA, conversión GGUF y cuantización. |
| `evaluate.py` | Métricas de JSON, emoción, similitud, latencia, juez opcional y reportes. |
| `versions.py` | Registro canónico, activación, rollback y métricas por versión. |
| `build_model.py` | Render del Modelfile (tag o `--gguf` local) y `ollama create`. |
| `service.py` | Lógica de la UI: chat, feedback, correcciones, dataset y versiones. |
| `ui.py` | Interfaz Gradio con pestañas Chat, Dataset, Versiones y Nube. |
| `datasets/seed.jsonl` | 348 ejemplos semilla (48 originales + 300 importados), 58 por emoción. |
| `datasets/nitshio_dataset_300.jsonl` | 300 ejemplos importados, 50 por emoción. |
| `datasets/train.jsonl` | Combinado deduplicado de 348 ejemplos; entrada del fine-tune. |
| `eval/prompts.yaml` | 18 prompts fijos con emoción esperada y referencia. |
| `cloud/Nitshio_QLoRA.ipynb` | Notebook de Colab: `/content`, unzip, dataset combinado y GGUF. |

**Flujo de entrenamiento:**

```text
seed + nitshio_dataset_300 + synthetic/corrections JSONL
   │
   ▼
schema + validate_dataset ──► deduplicación + train.jsonl + split estratificado
   │
   ▼
generate_synthetic (opcional) ──► más ejemplos + manifiesto
   │
   ▼
finetune_qlora (Colab) ──► adaptadores LoRA
   │
   ▼
export_gguf ──► GGUF f16 ──► q4_K_M
   │
   ▼
build_model --gguf ──► FROM local + TEMPLATE Phi-3.5 + type=gguf
   │
   ▼
ollama create nitshio-vN ──► evaluate.py ──► versions.py (activar/rollback)
```

La UI nunca lanza QLoRA en la T1200: muestra el comando y el notebook. Las
correcciones se guardan en JSONL y en ChromaDB para que el cambio sea inmediato
sin reentrenar.

### `scripts/`

| Script | Función |
|---|---|
| `check_models.py` | Mide Ollama, modelos, tokens/s, VRAM/RAM y emite veredicto. |
| `download_voice.py` | Descarga la voz Piper configurada. |
| `test_emotions.py` | Recorre emociones por VMC; `--dry-run` no envía UDP. |
| `check_docs.py` | Verifica changelog y documentos obligatorios. |

## 4. Manejo de errores

| Capa | Estrategia |
|---|---|
| Ollama | `OllamaError`; el orquestador puede reintentar el turno completo con `process_with_retry()`. |
| JSON | Reintento con instrucción correctiva; reparación conservadora; fallback neutral. |
| Memoria | Si Chroma o los embeddings fallan, se avisa y se continúa sin contexto. |
| Router | Si falla, se continúa sin herramienta. |
| Herramienta | `ToolRegistry.execute()` captura excepciones y devuelve `ToolResult.fallo()`. |
| TTS | Si falla, la respuesta de texto ya está lista; se registra el aviso. |
| Avatar | Si VMC falla, no se interrumpe la conversación. |
| STT | `STTError`/`MicrophoneError`; en modo continuo se informa y se sigue. |

## 5. Seguridad

- `FilesTool._resolver()` resuelve la ruta real y comprueba que esté dentro de
  `allowed_paths`.
- Los symlinks se rechazan antes de resolver la ruta, salvo
  `allow_symlinks: true`.
- Lectura y escritura tienen límites de bytes.
- Los archivos con byte nulo se consideran binarios y no se leen como texto.
- Borrar exige confirmación por defecto y usa papelera.
- Cada intento de borrado se registra en JSONL con timestamp, ruta, estado,
  modo y detalle.
- `read_own_code` solo acepta extensiones de texto dentro de la raíz.

## 6. Configuración y secretos

- `config/config.yaml`: todo lo no secreto.
- `config/emotions.yaml`: emociones, blendshapes y tiempos.
- `config/prompts/router.md`: prompt del router con `__TOOLS__`.
- `.env`: `FISH_API_KEY`, `GOOGLE_CSE_KEY`, `GOOGLE_CSE_CX`, `DISCORD_*`,
  `HF_TOKEN`, `OLLAMA_HOST`.

## 7. Estado de pruebas

- Suite unitaria con dobles para audio, VMC, Ollama, Chroma y navegador:
  **255 tests en verde** (192 del núcleo + 62 de training + 1 de persona).
- Round-trip real Piper → Moonshine validado en este equipo.
- Pipeline real de texto con Ollama validado (JSON válido y memoria activa).
- Micrófono real `USB MIC PRO` abre y captura.
- TTS real a `CABLE Input` confirmado en `CABLE Output` (RMS 0.123407).
- Router real con rutas exactas (`config/config.yaml`, listado de `config`,
  `brain/persona/nitshio.md`).
- Seis emociones VMC confirmadas visualmente y vuelta a neutral.
- Lip-sync visual confirmado con el proveedor viejo de VSeeFace.
- Prueba integral de voz (`scripts/test_voice_roundtrip.py`): STT Moonshine,
  orquestador real, TTS por chunks y audio en `CABLE Output` (RMS 0.068506,
  pico 0.283783).
- Datasets validados: 300 importados y 348 combinados, 0 errores y 0 duplicados.
- Split real del combinado con ratios 0.8/0.1/0.1: 276 train / 36 val / 36 test.
- Dry-run real de QLoRA: 348 ejemplos → 312 train / 36 val.
- Dry-run real de exportación GGUF: cuantización `q4_K_M`.
- Carga de GGUF local con `--gguf` validada: `FROM` absoluto, `TEMPLATE` de
  Phi-3.5 y registro `type=gguf`.
- Notebook Colab validado como JSON con 10 celdas y rutas `/content`.
- UI Gradio construida y arrancada en `http://127.0.0.1:7861/`.
- Evaluación real de `nitshio-v1` contra el modelo base con
  `--compare-base --no-judge` (18 prompts, embeddings, sin juez): score
  **0.5452** vs **0.4289**; reporte en
  `training/eval/reports/eval_20261109_051355.md` y métricas en
  `models/versions.json`.
