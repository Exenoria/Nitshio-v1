# Entrenamiento — Nitshio

> **Estado: planificado para el Paso 3. Todavía no implementado.**
> Este documento describe el diseño acordado para que la siguiente fase no
> improvise. No ejecutes estos comandos como si existieran: los scripts se
> crearán y probarán en su fase.

## 1. Decisión principal

El fine-tune se hace **solo en la nube** (Kaggle o Colab con GPU gratuita).
Localmente solo se hace inferencia con Ollama y se recibe el GGUF resultante.

Motivos:

- QLoRA de un modelo de 3B no cabe con comodidad en 4 GB.
- El entrenamiento local bloquea la máquina durante horas.
- Kaggle/Colab ofrecen GPU gratuita suficiente para QLoRA.

## 2. Flujo previsto

```text
dataset JSONL
   │
   ▼
validate_dataset.py  ──► errores de formato/emoción/roles
   │
   ▼
generate_synthetic.py ──► ejemplos nuevos + deduplicación + filtrado
   │
   ▼
finetune_qlora.py (Kaggle/Colab) ──► adaptadores LoRA
   │
   ▼
export_gguf.py ──► modelo GGUF cuantizado q4_K_M
   │
   ▼
ollama create nitshio-vN -f Modelfile
   │
   ▼
evaluate.py ──► reporte antes/después
   │
   ▼
versions.py ──► activar / rollback
```

## 3. Formato del dataset

Cada línea es un objeto JSON con un mensaje de usuario y la respuesta exacta
que debe aprender el modelo:

```json
{"user":"Hola, ¿quién eres?","response":"Hmph. Soy Nitshio, no que te importe.","emotion":"angry"}
```

Reglas de validación:

- JSON válido, una línea por ejemplo.
- `user` no vacío.
- `response` no vacío y con longitud razonable.
- `emotion` declarada en `config/emotions.yaml`.
- Sin duplicados exactos ni casi duplicados.
- Mezcla equilibrada de emociones; no todo `angry`.
- Separar entrenamiento, validación y prueba.

## 4. Fine-tune QLoRA

Parámetros de partida en `config/config.yaml`:

```yaml
training:
  qlora:
    max_seq_length: 2048
    load_in_4bit: true
    lora_r: 16
    lora_alpha: 16
    lora_dropout: 0.0
    batch_size: 2
    grad_accum: 4
    epochs: 3
    learning_rate: 0.0002
```

El notebook vivirá en `training/cloud/Nitshio_QLoRA.ipynb` y usará Unsloth o
TRL + PEFT. Los adaptadores se guardarán junto con el tokenizador y la
configuración exacta para poder reproducir el entrenamiento.

## 5. Exportación a GGUF y Ollama

1. Fusionar los adaptadores con el modelo base.
2. Convertir a GGUF.
3. Cuantizar a `q4_K_M`.
4. Crear el modelo en Ollama con `models/Modelfile.template`.
5. Verificar que responde JSON válido y que la personalidad se mantiene.

El nombre del modelo se registra en `models/versions.json`, nunca en el código.

## 6. Versionado y rollback

`models/versions.json` guardará, por versión:

- nombre del tag de Ollama;
- modelo base;
- fecha;
- dataset usado (hash);
- parámetros de entrenamiento;
- métricas de evaluación;
- ruta del GGUF;
- estado (`active`, `candidate`, `archived`).

`versions.py` permitirá activar una versión y volver a la anterior sin editar
el YAML a mano.

## 7. Evaluación

Set fijo de prompts en `training/eval/prompts.yaml`. Métricas:

- **JSON válido:** porcentaje de respuestas parseables.
- **Emoción correcta:** coincidencia con la emoción esperada.
- **Consistencia de personaje:** rúbrica o juicio del router.
- **Similitud semántica:** comparación con respuestas de referencia.
- **Longitud y latencia:** para no degradar la experiencia.

El reporte Markdown se guarda en `training/eval/reports` y compara la versión
candidata contra la activa, lado a lado.

## 8. UI de entrenamiento (Gradio)

Prevista para el Paso 3:

- chat de prueba;
- botones 👍/👎;
- corrección de respuesta y emoción;
- editor del dataset;
- reconstrucción del Modelfile;
- lanzamiento del fine-tune en la nube;
- activación de versión y rollback.

Las correcciones también se guardarán en ChromaDB para que Nitshio las
recuerde sin reentrenar.

## 9. Qué falta antes de empezar

- Cerrar la validación real de voz y VSeeFace del Paso 2.
- Crear el dataset semilla y el validador.
- Implementar `training/` y sus tests.
- Probar el notebook en Kaggle/Colab con una versión pequeña.
- Definir el set de evaluación y el umbral de mejora aceptable.
