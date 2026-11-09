# Plan de construcción — Nitshio

> Documento vivo. Se actualiza al cerrar cada fase.
> Última actualización: 2026-11-08

## 1. Resumen

Nitshio será una compañera IA 100 % local para Windows, con personalidad tsundere extrema,
voz, memoria, herramientas, avatar VRM en VSeeFace y un pipeline de entrenamiento con
versionado, evaluación y interfaz web. El único componente que puede requerir red es la
búsqueda web, la reproducción de YouTube y el proveedor opcional Fish Audio; el cerebro,
STT, TTS local, memoria y avatar funcionan sin conexión.

## 2. Entorno de validación (2026-11-07)

| Elemento | Estado |
|---|---|
| SO | Windows 10/11 |
| Python | 3.11.0 |
| Ollama | API local respondiendo |
| Modelos | Se configuran en `config/config.yaml` |
| Avatar | VRM con grupos de blendshapes, opcional |
| Audio virtual | VB-Cable, opcional |

### Proyecto previo encontrado

Existe un borrador independiente (Node.js) con una personalidad similar y un
dataset. **No se borrará ni modificará**. Nitshio se construirá aparte. Si el
usuario lo autoriza, se importarán ejemplos de ese dataset como semilla.

### Viabilidad de hardware

- **Inferencia:** Phi-3.5 Q4_K_M con contexto 2048 cabe en los 4 GB de VRAM; el
  router 0.8B también. El resto puede usar RAM. Medición real: 15.6 tok/s en el
  modelo activo y 73.2 tok/s en el router (veredicto AJUSTADO).
- **Fine-tuning Phi-3.5:** decisión confirmada: **solo cloud** (Kaggle/Colab con GPU
  gratuita). El entorno local no se usará para entrenar. El pipeline de entrenamiento,
  exportación GGUF y evaluación se ejecutará en la nube; Ollama local solo hace
  inferencia y recibe el modelo resultante.
- **PyTorch local:** no se instala build CUDA. La inferencia local usa Ollama (GPU vía
  su propio runtime) y ONNX Runtime CPU para STT. `requirements-train.txt` queda para
  el entorno cloud.

## 3. Decisiones confirmadas

1. **Raíz del proyecto:** carpeta independiente del borrador Nova y de la
   instalación global de npm.
2. **Modelo principal:** `phi3.5:3.8b-mini-instruct-q4_K_M` como base de inferencia y
   fine-tune. Nombres en `config/config.yaml`, nunca en código.
3. **Router:** `qwen3.5:0.8b` (ya existe). Configurable para cambiar a otro tag.
4. **STT por defecto:** Moonshine en español (`moonshine-voice`, modelo streaming
   small/tiny es). Hay soporte oficial de español en 2026. Fallback: faster-whisper.
5. **TTS local por defecto:** Piper con voz española (es_MX). Fish Audio opcional por
   `.env`. Alternativa local futura: Moonshine TTS.
6. **Borrado de archivos:** por defecto pide confirmación (`confirm_delete: true`),
   manda a papelera con `send2trash`, limita a `allowed_paths` y registra en log.
   El usuario puede poner `false` explícitamente.
7. **Búsqueda Google:** sin API key se abre Google en el navegador y se usa
   DuckDuckGo/HTML como fuente de resultados. Si el usuario aporta clave de Google
   Custom Search o SerpAPI, se usa esa.
8. **Discord:** adaptador opcional; se implementa y se prueba la lógica local, pero la
   conexión real requiere token y servidor del usuario.
9. **Audio:** salida a `CABLE Input` (VB-Cable) para que VSeeFace lea de
   `CABLE Output`; dispositivo configurable por nombre o índice.
10. **VSeeFace:** VMC/OSC en `127.0.0.1:39539`; blendshapes VRM estándar.
11. **ChromaDB:** colecciones `memories`, `conversations`, `self_knowledge` (código)
    y `documentation` (docs). Embeddings vía Ollama `nomic-embed-text`.
12. **Idioma:** español mexicano por defecto; configurable.

## 4. Arquitectura objetivo

```text
Usuario
  │  (texto o micrófono)
  ▼
STT Moonshine + VAD ──────────────┐
  ▼                               │
Tool Router (qwen3.5:0.8b)        │
  │                               │
  ├── Herramienta (plugin) ───────┤
  │                               ▼
  └──────────────────► LLM personalidad (Phi-3.5 / nitshio-vN)
                                  │
                                  ▼
                        Memoria ChromaDB (RAG)
                                  │
                                  ▼
                     Parser JSON + reintento/reparación
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
       TTS por chunks        Avatar VMC          Acciones agente
       (Piper/Fish)          (VSeeFace)          (plugins)
             │                    │
             ▼                    ▼
       VB-Cable ──────────► VSeeFace (lip-sync + emociones)
```

## 5. Estructura de carpetas

```text
Nitshio/
├─ AGENTS.md
├─ README.md
├─ requirements.txt
├─ requirements-train.txt
├─ requirements-dev.txt
├─ .env.example
├─ .gitignore
├─ install.ps1
├─ run.py
├─ run.bat
├─ config/
│  ├─ config.yaml
│  ├─ emotions.yaml
│  └─ prompts/router.md
├─ brain/
│  ├─ persona/nitshio.md
│  ├─ persona/few_shots.jsonl
│  ├─ llm/ollama_client.py
│  ├─ llm/parser.py
│  ├─ llm/orchestrator.py
│  └─ router/tool_router.py
├─ stt/
│  ├─ base.py
│  ├─ microphone.py
│  ├─ moonshine_provider.py
│  ├─ whisper_provider.py
│  └─ vad.py
├─ tts/
│  ├─ base.py
│  ├─ piper_provider.py
│  ├─ fish_provider.py
│  ├─ chunker.py
│  ├─ manager.py
│  └─ player.py
├─ pipeline/
│  ├─ text.py
│  └─ voice.py
├─ memory/
│  ├─ store.py
│  ├─ embeddings.py
│  └─ indexer.py
├─ tools/
│  ├─ base.py
│  ├─ registry.py
│  ├─ search_google.py
│  ├─ search_youtube.py
│  ├─ play_song.py
│  ├─ open_google.py
│  ├─ open_youtube.py
│  ├─ files.py
│  ├─ screen.py
│  └─ read_own_code.py
├─ avatar/
│  ├─ vmc.py
│  ├─ emotions.py
│  └─ lip_sync.py
├─ training/
│  ├─ datasets/
│  ├─ build_model.py
│  ├─ validate_dataset.py
│  ├─ generate_synthetic.py
│  ├─ finetune_qlora.py
│  ├─ export_gguf.py
│  ├─ evaluate.py
│  ├─ versions.py
│  ├─ ui.py
│  └─ eval/prompts.yaml
├─ scripts/
│  ├─ check_models.py
│  ├─ check_docs.py
│  ├─ download_voice.py
│  ├─ test_emotions.py
│  ├─ test_voice_roundtrip.py
│  ├─ build_manifest.py
│  └─ index_self.py
├─ models/
│  ├─ Modelfile.template
│  └─ versions.json
├─ tests/
├─ docs/
│  ├─ PLAN.md
│  ├─ ARCHITECTURE.md
│  ├─ CHANGELOG.md
│  ├─ DECISIONS.md
│  ├─ TRAINING.md
│  ├─ EXTENDING.md
│  ├─ TROUBLESHOOTING.md
│  └─ VSEEFACE.md
├─ data/
│  ├─ chroma/
│  ├─ audio/
│  └─ cache/
└─ logs/
```

## 6. Fases de ejecución

Cada fase termina con: pruebas ejecutadas, resumen de lo hecho, cómo probarlo y qué sigue.
No se avanza si la fase anterior falla.

### Estado de fases

| Fase | Estado | Evidencia |
|---|---|---|
| A — Paso 0: Ollama/hardware | ✅ Completada | `check_models.py` verde; 20.9 tok/s principal, 92.1 tok/s router, 0.12 s embeddings; JSON válido en ambos |
| B — Paso 1: cerebro | ✅ Completada | 40 tests; `nitshio-v1` con JSON válido y personalidad; `check_models.py` AJUSTADO (15.6 tok/s) |
| C — Paso 2: pipeline | ✅ Completada | 192 tests; prueba integral de voz Piper→Moonshine→Ollama→TTS→CABLE Output (RMS 0.0685); lip-sync visual confirmado; seis emociones VMC; router real |
| D — Paso 3: entrenamiento | ⏳ Pendiente | — |
| E — Paso 4: extensibilidad | ⏳ Pendiente | — |
| F — Paso 5: documentación | ⏳ Pendiente | — |
| G — Paso 6: emociones | ⏳ Pendiente | — |
| H — Paso 7: autoconciencia | ⏳ Pendiente | — |
| I — Cierre | ⏳ Pendiente | — |

### Fase A — Paso 0: base Ollama y hardware

- Crear estructura, `config/config.yaml`, `.env.example`, `requirements.txt`.
- Verificar GPU y el runtime GPU de Ollama (sin instalar PyTorch CUDA local).
- Descargar `phi3.5:3.8b-mini-instruct-q4_K_M`.
- Implementar `scripts/check_models.py`:
  - comprueba Ollama y tags,
  - mide latencia y tokens/s de router, principal y embeddings,
  - lee VRAM/RAM,
  - emite veredicto (bien/ajustado/no soportado) y alternativa más ligera.
- Prueba real con ambos modelos.
- **Aceptación:** script en verde, phi3.5 responde JSON, router responde JSON, GPU visible.

### Fase B — Paso 1: cerebro y personalidad

- `brain/persona/nitshio.md` con personalidad, doble fachada y reglas.
- Few-shots en JSONL.
- `Modelfile.template` con SYSTEM, TEMPLATE Phi-3, parámetros y MESSAGE.
- `brain/llm/parser.py`: modelo Pydantic, validación de emociones, reintento con
  instrucción correctiva, reparación de JSON y fallback neutral.
- `brain/llm/ollama_client.py`: API chat con `format` = JSON Schema, `think=false`.
- CLI mínima de chat para probar personalidad.
- Tests: JSON válido, emoción inválida, texto extra, JSON roto, reintento.
- **Aceptación:** 10/10 prompts de prueba devuelven JSON válido; el fallback funciona.

### Fase C — Paso 2: pipeline completo

- STT Moonshine español + VAD; entrada de texto.
- Router con esquema `{tool, arguments, confidence, reason}` y catálogo dinámico.
- Memoria ChromaDB: guardar/recuperar recuerdos, hechos y resúmenes; embeddings Ollama.
- Herramientas plugin: Google, YouTube, reproducir, abrir, archivos, pantalla,
  autoconocimiento.
- Seguridad de archivos: `allowed_paths`, resolución de symlinks, papelera, log,
  confirmación por defecto.
- TTS Piper por chunks + reproductor con selección de dispositivo; Fish Audio opcional.
- Avatar VMC: emociones, transiciones y lip-sync (VSeeFace micrófono virtual).
- Modo llamada local con barge-in; adaptador Discord opcional.
- Tests unitarios por módulo e integración de texto; prueba de micrófono y VSeeFace.
- **Aceptación:** conversación por texto y voz; herramientas funcionan; audio sale por
  VB-Cable; VSeeFace recibe emoción; borrado protegido.

#### Estado real del Paso 2 (2026-11-07)

Implementado y validado:

- [x] STT Moonshine español + VAD WebRTC con fallback por energía.
- [x] Micrófono con `sounddevice` y segmentación de utterances.
- [x] Router con esquema validado, umbral de confianza y catálogo dinámico.
- [x] Memoria ChromaDB con embeddings de Ollama y RAG.
- [x] Ocho herramientas plugin y registro con validación de argumentos.
- [x] Seguridad de archivos: `allowed_paths`, symlinks, tamaño, papelera, log y
      confirmación por defecto.
- [x] TTS Piper por chunks, Fish opcional, reproductor con selección de
      dispositivo y barge-in.
- [x] Avatar VMC con seis emociones y transiciones.
- [x] Orquestador con degradación elegante.
- [x] Pipelines de texto, voz y modo llamada.
- [x] `run.py`, `run.bat`, `download_voice.py`, `test_emotions.py`,
      `test_voice_roundtrip.py` y `check_docs.py`.
- [x] 192 tests en verde (incluye memoria, herramientas, router, avatar,
      orquestador, voz, micrófono y check_docs).
- [x] Round-trip real Piper → Moonshine: audio de 2.21 s transcrito como
      “hola soy nichio cómo está?”.
- [x] Pipeline real de texto con Ollama: JSON válido, memoria activa y router
      disponible.
- [x] Micrófono real abre y captura: `[1] Micrófono (USB MIC PRO)`,
      RMS 0.001737, pico 0.019348.
- [x] TTS real a `CABLE Input`: 1 chunk de 123948 bytes; audio confirmado en
      `CABLE Output` con RMS 0.123407 y pico 0.985046.
- [x] Router real con tres rutas exactas (`config/config.yaml`, listado de
      `config`, `brain/persona/nitshio.md`).
- [x] Seis emociones VMC confirmadas visualmente y vuelta a neutral.
- [x] Lip-sync visual confirmado con `OldLipSyncProvider=1`; el ajuste
      persiste tras cerrar y reiniciar VSeeFace.
- [x] Prueba integral de voz (`scripts/test_voice_roundtrip.py`): frase Piper
      inyectada en micrófono falso → Moonshine real
      (`"hola nizio cómo está?"`) → `nitshio-v1` → 2 chunks TTS (9.58 s) →
      `CABLE Output` con RMS 0.068506 y pico 0.283783; el avatar recibió la
      emoción del turno.

Confirmación humana opcional (no bloquea la fase):

- [ ] Hablar por el micrófono real en `run.py --mode voice` y confirmar de oído
      la respuesta. La cadena ya está cubierta por la prueba integral y el
      micrófono real se probó por separado.

### Fase D — Paso 3: entrenamiento real

- Dataset JSONL con validación estricta (JSON, emoción, roles, longitudes).
- Generador de datos sintéticos con filtrado y deduplicación.
- Construcción de Modelfile y `ollama create nitshio-vN`.
- Fine-tune QLoRA con Unsloth en Kaggle/Colab (GPU gratuita), exportación a GGUF y
  carga en Ollama local.
- Versionado `models/versions.json`, activar/rollback.
- Evaluación antes/después con set fijo: JSON válido, emoción, consistencia, similitud
  semántica y juicio opcional del router; reporte Markdown lado a lado.
- UI Gradio: chat, 👍/👎, corrección de respuesta/emoción, editor de dataset,
  reconstruir Modelfile, lanzar fine-tune, activar versión.
- Correcciones guardadas también en ChromaDB.
- **Aceptación:** crear `nitshio-v2`, comparar contra `v1`, ver mejora o diferencia
  medible, activar y volver atrás.

### Fase E — Paso 4: extensibilidad

- Interfaces abstractas para LLM, STT, TTS, memoria y avatar.
- Proveedores intercambiables por config: Piper/Fish, Moonshine/Whisper.
- Emociones nuevas por `config/emotions.yaml` + blendshape.
- Herramientas nuevas por archivo en `tools/`.
- Discord opcional y modo llamada.
- `docs/EXTENDING.md` con ejemplos completos.
- **Aceptación:** añadir un proveedor/tool/emoción de prueba sin tocar el núcleo.

### Fase F — Paso 5: documentación y control

- README, ARCHITECTURE, CHANGELOG, DECISIONS, TRAINING, EXTENDING, TROUBLESHOOTING,
  VSEEFACE.
- `AGENTS.md` con la regla obligatoria de changelog + docs en la misma tarea.
- `scripts/check_docs.py`: compara cambios de código con entradas del changelog.
- Indexación de docs en ChromaDB.
- **Aceptación:** `check_docs.py` falla si se modifica código sin changelog y pasa
  cuando se documenta.

### Fase G — Paso 6: emociones en VSeeFace

- Mapeo en `config/emotions.yaml`: neutral, fun, angry, joy, sorrow, surprised.
- Envío VMC `/VMC/Ext/Blend/Val` + `/VMC/Ext/Blend/Apply`.
- Fade in/out configurable, duración y retorno a neutral.
- `scripts/test_emotions.py` recorre todas las emociones.
- Guía VSeeFace: VMC, VRM, micrófono virtual, lip-sync.
- **Aceptación:** las seis emociones se ven en VSeeFace y vuelven a neutral.

### Fase H — Paso 7: autoconciencia

- Contexto de arranque con raíz, SO, versión, modelos activos y herramientas.
- `MANIFEST.json` automático con árbol, módulos y propósito.
- Indexación de código y docs en ChromaDB; reindexado por hash al iniciar/cambiar.
- Herramienta `read_own_code` con validación de rutas.
- **Aceptación:** preguntar “¿dónde está tu config?” o “¿cómo funcionas?” responde en
  personaje y con datos correctos.

### Fase I — Cierre

- Suite completa de tests, instalación limpia, `run.bat`.
- Revisión de criterios de aceptación.
- Resumen final y guía de uso.

## 7. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Memoria GPU local insuficiente para QLoRA | Entrenamiento solo en Kaggle/Colab |
| PyTorch CUDA no instalado | No hace falta localmente: inferencia con Ollama + ONNX |
| Moonshine español cambia de API | Adaptador aislado + fallback faster-whisper |
| Piper no tiene voz es_MX | Descargar voz es_ES/es_MX; fallback Fish o SAPI |
| VMC no llega a VSeeFace | Script de diagnóstico OSC + guía de puerto/firewall |
| Latencia de Phi-3.5 en 4 GB | Contexto 4K, KV cache q8, router separado, TTS por chunks |
| Borrado accidental | Papelera, allowed_paths, confirmación, log, tests |
| Respuestas JSON inválidas | JSON Schema, reintento, reparación, fallback neutral |
| Discord requiere token | Adaptador opcional, no bloquea el resto |
| Búsqueda Google bloqueada | DuckDuckGo + abrir navegador; API key opcional |
| Dataset escaso | Generación sintética + correcciones desde la UI |

## 8. Criterios de aceptación globales

- [x] Ollama local con Phi-3.5 y router, sin nombres hardcodeados.
- [ ] Personalidad tsundere en español, salida JSON válida y parser robusto.
      El parser y el JSON ya están validados; la calidad de personalidad se
      ajusta en el Paso 3.
- [x] Pipeline texto + voz + VAD + barge-in.
- [ ] Memoria RAG funcional y correcciones recordadas al instante. La memoria
      RAG funciona; las correcciones desde la UI llegan en el Paso 3.
- [x] Herramientas con plugins y borrado seguro.
- [x] TTS por chunks, salida a VB-Cable, Fish opcional.
- [x] VSeeFace recibe emociones y lip-sync.
- [ ] Fine-tune QLoRA + GGUF + Ollama, versionado y rollback.
- [ ] Evaluación antes/después con reporte.
- [ ] UI Gradio de entrenamiento y correcciones.
- [x] Documentación completa, changelog obligatorio y `check_docs.py`.
- [ ] Autoconciencia de ruta/código con `MANIFEST.json` y `read_own_code`.
      `read_own_code` ya existe; falta el manifiesto y la indexación completa.
- [x] Tests mínimos por módulo y scripts de instalación/arranque.

## 9. Orden de trabajo inmediato

1. Paso 2 cerrado. Iniciar el Paso 3: dataset JSONL, generador sintético,
   fine-tune QLoRA en Kaggle/Colab, exportación GGUF, evaluación y versionado.
2. Mantener `pytest -q` + `check_docs.py` como gate de cada subfase.
3. La confirmación humana de `run.py --mode voice` queda como prueba opcional
   cuando el usuario quiera hablar por el micrófono.
