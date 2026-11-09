# Decisiones de arquitectura — Nitshio

Registro de decisiones importantes (ADR ligero). Cada una incluye contexto,
decisión y consecuencias.

---

## D-001 — Raíz del proyecto: carpeta independiente

**Contexto:** la sesión inició en una carpeta global de npm, que no es un lugar
adecuado para un proyecto de aplicación.

**Decisión:** crear Nitshio en una carpeta propia y mover ahí la sesión.

**Consecuencias:** rutas limpias, independientes de npm; el borrador Nova no se
toca. Los scripts resuelven la raíz con `Path(__file__).resolve().parents[...]`,
nunca con una ruta absoluta hardcodeada.

---

## D-002 — Empezar de cero; no reutilizar Nova

**Contexto:** Existe un borrador funcional independiente (Node.js) con personalidad y dataset.

**Decisión:** Nitshio se construye desde cero. Nova se deja intacto. Si en el
futuro se quiere importar un ejemplo, se copia explícitamente, nunca se
modifica el original.

**Consecuencias:** no hay deuda técnica heredada; la personalidad y el pipeline
quedan diseñados para los requisitos actuales.

---

## D-003 — Phi-3.5 como modelo principal y qwen3.5:0.8b como router

**Contexto:** el requisito pide Phi-3.5 mini como base del fine-tune y un router
pequeño. Ambos ya están disponibles en Ollama (Phi-3.5 hay que descargarlo).

**Decisión:** `phi3.5:3.8b-mini-instruct-q4_K_M` para personalidad e inferencia;
`qwen3.5:0.8b` para decidir herramientas. Los tags viven en
`config/config.yaml`.

**Consecuencias:** 4 GB de VRAM alcanzan para inferencia, pero no conviene
mantener ambos modelos cargados a la vez. Se usará `OLLAMA_MAX_LOADED_MODELS=1`
o `keep_alive` corto para el router.

---

## D-004 — Fine-tuning solo en la nube (Kaggle/Colab)

**Contexto:** los recursos de GPU disponibles localmente son limitados. Unsloth
reporta 3.5 GB como mínimo absoluto para QLoRA de 3B; entrenar localmente es
posible en teoría pero frágil y lento.

**Decisión:** el fine-tune real se ejecuta en una GPU gratuita de Kaggle o
Colab. El proyecto incluye el notebook, los scripts de preparación/exportación y
la integración con Ollama local; no se instala PyTorch CUDA localmente.

**Consecuencias:** entrenamiento reproducible y sin riesgo de OOM local. La
inferencia y todo el resto siguen siendo 100 % locales. El usuario necesita una
cuenta gratuita de Kaggle o Google Colab para entrenar.

---

## D-005 — STT Moonshine en español

**Contexto:** Moonshine ya soporta español (modelos streaming tiny/small) en
2026, a diferencia de la versión inglesa original. La personalidad es mexicana,
así que el STT debe entender español.

**Decisión:** `moonshine-voice` con modelo streaming small en español, VAD
`webrtcvad` y fallback opcional a faster-whisper.

**Consecuencias:** STT local, CPU-only y de baja latencia. El adaptador queda
aislado para poder cambiar de proveedor sin tocar el pipeline.

---

## D-006 — TTS local Piper, Fish Audio opcional

**Contexto:** se necesita TTS local por chunks, con salida a VB-Cable y
posibilidad de usar Fish Audio.

**Decisión:** Piper (`piper-tts`) con voz `es_MX-claude-high` como proveedor por
defecto; Fish Audio como proveedor opcional con API key en `.env`.

**Consecuencias:** voz local rápida y sin red. La calidad de Piper es buena pero
no al nivel de Fish; se puede cambiar con una línea de configuración.

---

## D-007 — Salida JSON con esquema, reintento y reparación

**Contexto:** el modelo puede devolver texto extra, comillas mal escapadas o una
emoción inválida.

**Decisión:** validar con Pydantic; forzar JSON Schema en Ollama; hasta dos
reintentos con un mensaje correctivo; reparación conservadora; último recurso
`{"response": "...", "emotion": "neutral"}`.

**Consecuencias:** el pipeline nunca se rompe por una respuesta mal formada. El
parser registra cada fallo para poder evaluar y entrenar.

---

## D-008 — Borrado seguro por defecto

**Contexto:** una herramienta de borrado manejada por un LLM puede destruir
archivos si el modelo alucina una ruta.

**Decisión:** `allowed_paths` obligatorio, resolución de symlinks, papelera
(`send2trash`), log JSONL y confirmación por defecto
(`confirm_delete: true`). El usuario puede poner `false` explícitamente.

**Consecuencias:** un error del modelo no borra fuera del workspace ni de forma
permanente. La confirmación añade un paso, pero es el default seguro pedido.

---

## D-009 — Memoria ChromaDB con colecciones separadas

**Contexto:** memorias del usuario, conversaciones, código y documentación
tienen ciclos de vida distintos.

**Decisión:** colecciones `memories`, `conversations`, `self_knowledge` y
`documentation`, con embeddings de Ollama (`nomic-embed-text`).

**Consecuencias:** se puede reindexar el código sin tocar recuerdos, y recuperar
documentación por separado. El RAG inyecta solo lo relevante y limita caracteres.

---

## D-010 — VMC/OSC a VSeeFace en `127.0.0.1:39539`

**Contexto:** VSeeFace acepta VMC para blendshapes y seguimiento.

**Decisión:** enviar `/VMC/Ext/Blend/Val` + `/VMC/Ext/Blend/Apply` por UDP al
puerto configurado. Las emociones se mapean a blendshapes VRM estándar en
`config/emotions.yaml`.

**Consecuencias:** no hace falta un SDK propietario; cualquier avatar VRM con
los blendshapes estándar funciona. El lip-sync se delega al micrófono virtual de
VSeeFace (CABLE Output), no a los visemas VMC (ver D-016).

---

## D-011 — Documentación obligatoria en la misma tarea

**Contexto:** el proyecto se modifica por fases y es fácil perder el rastro.

**Decisión:** `AGENTS.md` obliga a actualizar `docs/CHANGELOG.md` y la doc
correspondiente en la misma tarea; `scripts/check_docs.py` lo verifica.

**Consecuencias:** cada cambio queda justificado y es reproducible por otra
sesión. El check puede dar falsos positivos; se documenta cómo resolverlos.

---

## D-012 — Embeddings calculados fuera de Chroma

**Contexto:** la función de embeddings por defecto de ChromaDB puede descargar
un modelo oculto y duplicar el que ya tenemos en Ollama.

**Decisión:** Chroma solo almacena vectores. `memory/embeddings.py` calcula los
embeddings con Ollama (`nomic-embed-text`), usando `/api/embed` y cayendo a
`/api/embeddings` si la versión es antigua.

**Consecuencias:** una sola fuente de embeddings, control explícito del modelo
y tests con un embedder falso sin red.

---

## D-013 — Moonshine en CPU con ONNX Runtime

**Contexto:** no se instala PyTorch CUDA localmente.

**Decisión:** Moonshine corre en CPU con ONNX Runtime. El adaptador usa
`get_model_for_language()` y `Transcriber.transcribe_without_streaming()`.

**Consecuencias:** el STT no compite por VRAM con Ollama. La latencia es buena
para utterances cortas; si el modelo cambia de API, el adaptador queda aislado.

---

## D-014 — `FilesTool.execute()` devuelve fallos, no excepciones

**Contexto:** el registro captura excepciones, pero llamar a la herramienta
directamente también debe ser seguro y predecible.

**Decisión:** `FilesTool.execute()` envuelve la operación y convierte
`ToolError`/`OSError` en `ToolResult.fallo()`. La validación de symlinks ocurre
antes de resolver la ruta real para no confundir un symlink con una ruta
externa.

**Consecuencias:** el orquestador y los tests reciben siempre un resultado
uniforme; los mensajes de error distinguen symlink, ruta fuera de
`allowed_paths`, tamaño y binario.

---

## D-015 — `check_docs.py` basado en fechas, no en git

**Contexto:** el proyecto puede no tener git inicializado y la regla de
documentación debe comprobarse igualmente.

**Decisión:** comparar la fecha de modificación de los archivos de código con
la del changelog. La última entrada debe mencionar cada archivo modificado por
ruta, carpeta (`brain/`) o nombre, y debe incluir `### Archivos afectados`.

**Consecuencias:** funciona sin git. Si hay falsos positivos por marcas de
tiempo iguales, se usa `--tolerance` o se actualiza el changelog. En el futuro
se puede añadir un modo basado en `git diff` sin romper el actual.

---

## D-016 — Lip-sync por audio de VSeeFace, no por visemas VMC

**Contexto:** Nitshio puede enviar visemas `A/I/U/E/O` por VMC, pero en las
pruebas VSeeFace no movió la boca con ellos. El proveedor nuevo de lip-sync por
audio tampoco reaccionó al audio de `CABLE Output`; el proveedor viejo sí.

**Decisión:** el lip-sync activo es el de VSeeFace leyendo `CABLE Output`, con
`AudioLipSync=1` y `OldLipSyncProvider=1`. Nitshio no envía visemas por defecto;
`avatar/lip_sync.py` queda como utilidad opcional y no se activa sola.

**Consecuencias:** una sola ruta de lip-sync, sin peleas entre VMC y audio. El
ajuste debe hacerse en la UI de VSeeFace y persiste en `settings.ini`. Si una
versión futura de VSeeFace arregla el proveedor nuevo, se puede volver a
`OldLipSyncProvider=0` sin tocar el código de Nitshio.
