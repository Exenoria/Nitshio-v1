# Extender Nitshio

> Última actualización: 2026-11-07 (Paso 2). Los ejemplos son contratos reales
> del código actual.

## 1. Añadir una herramienta (plugin)

Toda herramienta vive en `tools/` y expone `create_tool(context)` o `TOOL`.

```python
# tools/weather.py
from __future__ import annotations

from typing import Any

from tools.base import Tool, ToolContext, ToolResult


class WeatherTool(Tool):
    name = "weather"
    description = "Consulta el clima actual de una ciudad."
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Ciudad a consultar."}
        },
        "required": ["city"],
        "additionalProperties": False,
    }

    def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        ciudad = str(arguments["city"]).strip()
        if not ciudad:
            return ToolResult.fallo("La ciudad está vacía.")
        # Aquí llamarías a tu API. Nunca inventes datos si falla.
        return ToolResult.exito(
            f"Clima de {ciudad}: 24 °C, despejado.",
            city=ciudad,
            temperature_c=24,
        )


def create_tool(context: ToolContext) -> WeatherTool:
    del context
    return WeatherTool()
```

Después añade el nombre a `config/config.yaml`:

```yaml
tools:
  enabled:
    - weather
```

El `ToolRegistry` lo importa como `tools.weather`, valida los argumentos contra
`parameters` y captura excepciones. El router lo verá automáticamente en su
catálogo.

### Reglas para herramientas

- Devuelve siempre `ToolResult.exito(...)` o `ToolResult.fallo(...)`.
- No lances excepciones para errores esperados; el registro las captura, pero
  un mensaje claro es mejor.
- Valida rutas si tocas archivos. No dupliques la seguridad de `files.py` sin
  motivo.
- Si la herramienta tarda, el turno espera: mantén timeouts.
- Añade tests en `tests/` con dobles; no dependas de red en la suite.

## 2. Añadir una emoción

Edita `config/emotions.yaml`:

```yaml
emotions:
  shy:
    display_name: Tímida
    vrm_blendshape: Shy
    intensity: 0.85
    fade_in_ms: 300
    hold_ms: 900
    fade_out_ms: 600
    description: "Se sonroja y evita la mirada."
```

El parser lee la lista de ahí; no hay que tocar código. Comprueba que el
blendshape exista en el avatar VRM y pruébalo:

```powershell
.\.venv\Scripts\python.exe scripts\test_emotions.py --emotion shy --dry-run
.\.venv\Scripts\python.exe scripts\test_emotions.py --emotion shy
```

## 3. Cambiar de proveedor STT

`stt/base.py` expone `crear_stt(config)`. Para añadir uno:

1. Implementa `STTProvider.transcribe(audio, sample_rate)` y devuelve un
   `Transcript`.
2. Añade el caso en `crear_stt()`.
3. Configura `stt.provider` en el YAML.

Contrato de audio: mono `float32` en `[-1, 1]`. Usa `preparar_audio()` si tu
motor espera otro formato.

## 4. Cambiar de proveedor TTS

1. Implementa `TTSProvider.synthesize(text) -> AudioChunk`.
2. Añade el caso en `tts/base.py::crear_tts()`.
3. Configura `tts.provider`.

`AudioChunk` lleva `data` (WAV/MP3), `sample_rate`, `format` y `duration_s`.
El `TTSManager` se encarga del chunking y del reproductor.

## 5. Cambiar la memoria

`MemoryStore` acepta un `embedder` y un `client` inyectables. Para pruebas usa
`chromadb.EphemeralClient()` y un embedder determinista, como en
`tests/test_memory.py`.

Si añades una colección:

```yaml
memory:
  collections:
    recipes: recipes
```

y usa `store.add("recipes", texto)`, `store.query("recipes", consulta)`.

## 6. Cambiar el router

El prompt está en `config/prompts/router.md` y el catálogo se inyecta en
`__TOOLS__`. El esquema de decisión es:

```json
{
  "tool": "nombre_o_none",
  "arguments": {},
  "confidence": 0.0,
  "reason": "por qué"
}
```

Si cambias el esquema, actualiza `router_schema()` en
`brain/router/tool_router.py` y sus tests.

## 7. Añadir un proveedor de LLM

Todo el proyecto llama a `OllamaClient` a través de `chat()` y
`chat_structured()`. Para otro backend, mantén esa interfaz o crea un adaptador
que la cumpla; el orquestador no conoce detalles HTTP.

## 8. Añadir pruebas

- Unitarias: dobles para red, audio, VMC y Chroma.
- Integración real: márcala o ejecútala a mano; no debe romper la suite.
- Después de cualquier cambio:

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe scripts\check_docs.py
  ```

- Actualiza `docs/CHANGELOG.md` y este documento si cambia el contrato.
