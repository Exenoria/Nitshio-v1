# Solución de problemas — Nitshio

> Última actualización: 2026-11-07 (cierre del Paso 2).

## Ollama

### `No se pudo conectar con Ollama`

1. Comprueba que Ollama esté corriendo:

   ```powershell
   ollama list
   ollama serve
   ```

2. Revisa `OLLAMA_HOST` en `.env` y `ollama.host` en `config/config.yaml`.
3. Si usas otra máquina, asegúrate de que el puerto `11434` esté accesible.

### `model 'nitshio-v1' not found`

El modelo activo se declara en `config/config.yaml`. Si no existe:

```powershell
ollama list
ollama create nitshio-v1 -f models/Modelfile
```

O cambia temporalmente `ollama.models.active` al modelo base. Nunca cambies el
nombre en el código: el proyecto lee el tag del YAML.

### Respuestas lentas

- Si la GPU tiene memoria limitada, mantén `num_ctx: 2048`.
- No dejes cargados a la vez el modelo principal y el router si no hace falta.
- Usa `--no-tools` para saltarte el router en pruebas de personalidad.
- `scripts/check_models.py` mide tokens/s y sugiere alternativas más ligeras.

### El modelo devuelve texto fuera del JSON o una emoción inválida

El parser reintenta y repara. Si aun así falla, usa el fallback neutral. Para
diagnosticar:

```powershell
.\.venv\Scripts\python.exe run.py --mode text --once "Hola" --debug
```

Revisa `attempts`, `fallback` y `reparado` en la salida de depuración.

## Memoria / ChromaDB

### `No se pudo abrir ChromaDB`

- Comprueba permisos sobre `data/chroma`.
- Borra la carpeta solo si aceptas perder recuerdos:
  `Remove-Item -Recurse data\chroma`.
- Asegúrate de que `chromadb` esté instalado:
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### `Fallo de embeddings en Ollama`

- Verifica que `nomic-embed-text:latest` esté descargado.
- Comprueba que Ollama responda en `/api/embed`.
- Si usas una versión antigua de Ollama, el cliente cae a `/api/embeddings`
  automáticamente.

### La memoria no recuerda nada

- La primera conversación solo guarda; la recuperación aparece en turnos
  posteriores.
- `min_score` en `config/config.yaml` puede ser demasiado alto.
- Usa `/memoria <texto>` en el chat para ver el contexto recuperado.

## Voz

### `No existe la voz Piper`

```powershell
.\.venv\Scripts\python.exe scripts\download_voice.py
```

Verifica que existan:

- `data/models/piper/es_MX-claude-high.onnx`
- `data/models/piper/es_MX-claude-high.onnx.json`

### `No se pudo cargar la voz Piper`

- La ruta debe apuntar a un `.onnx`, no a la raíz del proyecto.
- El archivo `.onnx.json` debe estar junto al `.onnx`.
- Revisa `tts.piper.model_path` y `tts.piper.config_path`.

### No se oye audio

1. Ejecuta `run.py --mode diagnose` y copia el nombre exacto del dispositivo.
2. Ajusta `tts.audio.output_device` o usa el índice.
3. Si usas VB-Cable, recuerda: Nitshio escribe en `CABLE Input`; VSeeFace lee
   de `CABLE Output`.
4. Prueba sin VSeeFace con `--no-avatar`.

### El micrófono no captura

- Revisa el dispositivo de entrada en Windows.
- Prueba `python run.py --mode voice` y habla después de que el VAD detecte
  silencio inicial.
- Si `webrtcvad` falla, el segmentador usa el fallback por energía.
- Ajusta `stt.vad.aggressiveness`, `min_speech_ms` y `end_silence_ms`.

### Moonshine no transcribe bien

- El modelo español se descarga la primera vez en `data/models/moonshine`.
- Comprueba `stt.language: es`.
- Habla cerca del micrófono y evita ruido de fondo.
- Como fallback, cambia `stt.provider` a `whisper` si tienes
  `faster-whisper` instalado.

### La prueba integral de voz falla o no detecta audio

`scripts/test_voice_roundtrip.py` inyecta audio Piper en un micrófono falso y
comprueba la cadena completa. Si falla:

1. Ejecuta sin herramientas para descartar el router:
   `.\.venv\Scripts\python.exe scripts\test_voice_roundtrip.py --no-tools --json`
2. Si la transcripción sale vacía, revisa Moonshine y la voz Piper.
3. Si `CABLE Output` da RMS casi cero:
   - comprueba que `tts.audio.output_device` sea `CABLE Input`;
   - cierra otras aplicaciones que estén usando el cable;
   - ejecuta `run.py --mode diagnose` y confirma que el dispositivo existe.
4. Si el TTS no genera chunks, revisa `tts.piper.model_path` y el log
   `logs/nitshio.log`.
5. Si el avatar no recibe la emoción, revisa el puerto VMC y que VSeeFace esté
   abierto. La prueba lo avisa, pero no falla si VSeeFace está cerrado.

## Herramientas

### `search_google` no devuelve resultados

- Sin claves de Google usa DuckDuckGo; si también falla, abre Google en el
  navegador.
- Para usar Google CSE, rellena `GOOGLE_CSE_KEY` y `GOOGLE_CSE_CX` en `.env`.

### `play_song` no reproduce

- Comprueba que `ffplay` esté en el `PATH`:

  ```powershell
  ffplay -version
  ```

- Ajusta `tools.media.player` si el ejecutable tiene otro nombre.
- El límite `max_duration_seconds` evita descargar audio demasiado largo.

### `No se encontró el reproductor`

Instala FFmpeg o cambia `tools.media.player` a un reproductor disponible.

### Borrado cancelado

Es el comportamiento seguro por defecto. `confirm_delete: true` exige
confirmación. Si quieres desactivarlo, hazlo de forma explícita en
`config/config.yaml` y asume el riesgo.

### `Ruta fuera de allowed_paths`

La herramienta solo opera dentro de `tools.allowed_paths`. Añade la carpeta
necesaria al YAML o mueve el archivo a `workspace/`.

### `Symlink no permitido`

Por seguridad, los symlinks se rechazan. Usa una ruta real o activa
`allow_symlinks: true` solo si confías en el origen.

## Avatar / VSeeFace

Consulta [VSEEFACE.md](VSEEFACE.md). Los fallos más comunes son el puerto VMC,
el dispositivo de audio y los nombres de blendshape.

### Se oye el audio pero el avatar no mueve la boca

Este fue el fallo real en este equipo. El audio llegaba a `CABLE Output`, pero
VSeeFace usaba un proveedor de lip-sync que no reaccionaba. Solución:

1. Abre **Configuración General** en VSeeFace.
2. Activa **Lipsync por audio** (`AudioLipSync=1`).
3. Selecciona **`CABLE Output (VB-Audio Virtual Cable)`** como micrófono.
4. Baja a **Opciones avanzadas** y activa
   **“Usar el modelo viejo para lip sync (menor rendimiento)”**
   (`OldLipSyncProvider=1`).
5. Reinicia VSeeFace y verifica en el archivo `settings.ini` de VSeeFace,
   bajo `%USERPROFILE%\AppData\LocalLow\<proveedor>\VSeeFace\`:

   ```ini
   Microphone=CABLE Output (VB-Audio Virtual Cable)
   AudioLipSync=1
   OldLipSyncProvider=1
   VMCReceiveEnable=1
   ```

No envíes visemas `A/I/U/E/O` por VMC a la vez: en las pruebas no movieron la
boca y el lip-sync por audio los sobrescribía.

### El avatar no cambia de emoción

- Comprueba que `VMCReceiveEnable=1` y el puerto `39539`.
- Usa un VRM que declare grupos de blendshapes; un avatar sin esos grupos no
  mostrará las emociones aunque se reciba el mensaje VMC.
- Ejecuta `scripts/test_emotions.py --all` y revisa los valores impresos.

### La captura de VSeeFace sale en blanco

`PrintWindow` devuelve una imagen blanca si la ventana está minimizada.
Restaura VSeeFace antes de capturar. No es un fallo del avatar ni de Nitshio.

## Tests y documentación

### `pytest` falla por audio o VMC

Los tests usan dobles y no deberían abrir dispositivos reales. Si falla algo
del entorno, revisa que estés usando el Python del `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### `check_docs.py` dice que falta documentación

1. Añade la entrada en `docs/CHANGELOG.md` con `### Archivos afectados`.
2. Incluye la ruta, la carpeta (`brain/`) o el nombre del archivo.
3. Actualiza el documento correspondiente.
4. Vuelve a ejecutar:

   ```powershell
   .\.venv\Scripts\python.exe scripts\check_docs.py
   ```

### `check_docs.py` da un falso positivo por fecha

El script compara fechas de modificación. Si acabas de tocar un archivo y el
changelog tiene la misma marca de tiempo, añade un margen:

```powershell
.\.venv\Scripts\python.exe scripts\check_docs.py --tolerance 5
```

## Logs

- `logs/nitshio.log`: log rotativo general.
- `logs/deletions.jsonl`: registro de borrados.
- `data/cache/screens`: capturas de pantalla.
- `data/cache/audio`: audio descargado por `play_song`.
