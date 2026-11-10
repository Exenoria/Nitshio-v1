#!/usr/bin/env python
"""UI Gradio de entrenamiento y correcciones de Nitshio.

Ejecuta:
    python training/ui.py
    python training/ui.py --port 7860

La UI no entrena en local. Prepara datos, prueba el modelo activo, guarda
correcciones, reconstruye el Modelfile y activa o revierte versiones. El
fine-tune QLoRA se lanza desde el notebook de Kaggle/Colab.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.service import TrainingService, TrainingServiceError


def _estado(texto: str) -> str:
    return f"**{texto}**"


def build_ui(service: TrainingService | None = None) -> Any:
    """Construye la interfaz; importa Gradio solo al abrirla."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "Gradio no está instalado. Ejecuta "
            "`.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        ) from exc

    servicio = service or TrainingService()

    def enviar(mensaje: str, historial: list[dict[str, str]] | None, ultimo: dict | None):
        if not str(mensaje or "").strip():
            return historial or [], ultimo, ""
        try:
            turno = servicio.chat(mensaje, history=historial or [])
        except (TrainingServiceError, Exception) as exc:
            nuevo = list(historial or []) + [
                {"role": "user", "content": mensaje},
                {"role": "assistant", "content": f"ERROR: {exc}"},
            ]
            return nuevo, ultimo, ""
        nuevo = list(historial or []) + [
            {"role": "user", "content": mensaje},
            {"role": "assistant", "content": f"{turno.response}\n\n_[{turno.emotion}]_"},
        ]
        estado = {
            "user": turno.user,
            "response": turno.response,
            "emotion": turno.emotion,
            "attempts": turno.attempts,
            "fallback": turno.fallback,
        }
        return nuevo, estado, ""

    def feedback(
        ultimo: dict | None,
        rating: str,
        respuesta_corregida: str,
        emocion_corregida: str,
    ) -> str:
        if not ultimo:
            return "Primero envía un mensaje."
        try:
            resultado = servicio.save_feedback(
                ultimo.get("user", ""),
                ultimo.get("response", ""),
                ultimo.get("emotion", ""),
                rating,
                corrected_response=respuesta_corregida or None,
                corrected_emotion=emocion_corregida or None,
            )
        except (TrainingServiceError, Exception) as exc:
            return f"ERROR: {exc}"
        extra = f" Memoria: `{resultado.memory_id}`." if resultado.memory_id else ""
        return resultado.message + extra

    def cargar_dataset(ruta: str) -> tuple[str, str]:
        try:
            texto = servicio.dataset_text(ruta or None)
        except Exception as exc:
            return "", f"ERROR: {exc}"
        return texto, f"Cargado: {ruta or 'seed.jsonl'}"

    def validar_dataset(texto: str) -> str:
        errores = servicio.validate_dataset_text(texto)
        if errores:
            return "Errores:\n" + "\n".join(f"- {error}" for error in errores)
        return "Dataset válido."

    def guardar_dataset(texto: str, ruta: str) -> str:
        try:
            destino = servicio.save_dataset_text(texto, path=ruta or None)
        except Exception as exc:
            return f"ERROR: {exc}"
        return f"Guardado en `{destino}`."

    def refrescar_versiones() -> list[list[str]]:
        try:
            return servicio.version_rows()
        except Exception as exc:
            return [["ERROR", str(exc), "", "", ""]]

    def activar_version(nombre: str) -> str:
        try:
            servicio.activate(nombre.strip())
        except Exception as exc:
            return f"ERROR: {exc}"
        return f"Versión activa: `{nombre.strip()}`. Reinicia el pipeline para usarla."

    def hacer_rollback() -> str:
        try:
            entrada = servicio.rollback()
        except Exception as exc:
            return f"ERROR: {exc}"
        return f"Rollback a `{entrada.get('name')}`. Reinicia el pipeline."

    def construir(version: str, crear: bool, activar: bool, dataset: str) -> str:
        try:
            resultado = servicio.build_modelfile(
                version.strip(),
                create=bool(crear),
                activate=bool(activar),
                dataset_path=dataset or None,
            )
        except Exception as exc:
            return f"ERROR: {exc}"
        return (
            f"Modelfile: `{resultado['modelfile']}`\n\n"
            f"Versión: `{resultado['version']}` | creada: {resultado['created']} | "
            f"activa: {resultado['activated']}"
        )

    plan = servicio.cloud_plan()

    with gr.Blocks(title="Nitshio — entrenamiento") as demo:
        gr.Markdown(
            "# Nitshio — entrenamiento y correcciones\n"
            "Prueba el modelo activo, guarda feedback y prepara el dataset. "
            "El fine-tune QLoRA se ejecuta en Kaggle/Colab."
        )
        ultimo = gr.State(None)
        with gr.Tab("Chat"):
            # Gradio 6 usa mensajes por defecto y ya no acepta ``type``;
            # Gradio 5 necesita el argumento explícito.
            try:
                chatbot = gr.Chatbot(type="messages", height=420)
            except TypeError:
                chatbot = gr.Chatbot(height=420)
            with gr.Row():
                mensaje = gr.Textbox(
                    label="Mensaje",
                    placeholder="Hola, Nitshio...",
                    scale=4,
                )
                enviar_btn = gr.Button("Enviar", variant="primary", scale=1)
            estado_chat = gr.Markdown("Sin turno todavía.")
            with gr.Row():
                bien = gr.Button("👍")
                mal = gr.Button("👎")
            respuesta_corregida = gr.Textbox(
                label="Corrección de la respuesta (opcional)",
                lines=3,
            )
            emocion_corregida = gr.Dropdown(
                choices=list(servicio.persona.emotions),
                value=servicio.persona.default_emotion,
                label="Emoción corregida (opcional)",
            )
            guardar_feedback = gr.Button("Guardar feedback/corrección")
            estado_feedback = gr.Markdown("")
            enviar_btn.click(
                enviar,
                inputs=[mensaje, chatbot, ultimo],
                outputs=[chatbot, ultimo, mensaje],
            ).then(
                lambda estado: (
                    _estado(
                        f"Último turno: emoción {estado['emotion']} | "
                        f"intentos {estado['attempts']} | "
                        f"fallback {estado['fallback']}"
                    )
                    if estado
                    else "Sin turno todavía."
                ),
                inputs=[ultimo],
                outputs=[estado_chat],
            )
            mensaje.submit(
                enviar,
                inputs=[mensaje, chatbot, ultimo],
                outputs=[chatbot, ultimo, mensaje],
            )
            bien.click(
                lambda ultimo: feedback(ultimo, "up", "", ""),
                inputs=[ultimo],
                outputs=[estado_feedback],
            )
            mal.click(
                lambda ultimo: feedback(ultimo, "down", "", ""),
                inputs=[ultimo],
                outputs=[estado_feedback],
            )
            guardar_feedback.click(
                feedback,
                inputs=[ultimo, gr.State("down"), respuesta_corregida, emocion_corregida],
                outputs=[estado_feedback],
            )

        with gr.Tab("Dataset"):
            ruta_dataset = gr.Textbox(
                label="Archivo",
                value=str(servicio.dataset_dir / "seed.jsonl"),
            )
            with gr.Row():
                cargar_btn = gr.Button("Cargar")
                validar_btn = gr.Button("Validar")
                guardar_btn = gr.Button("Guardar", variant="primary")
            editor = gr.Code(language="json", lines=24, label="JSONL")
            estado_dataset = gr.Markdown("")
            cargar_btn.click(
                cargar_dataset, inputs=[ruta_dataset], outputs=[editor, estado_dataset]
            )
            validar_btn.click(validar_dataset, inputs=[editor], outputs=[estado_dataset])
            guardar_btn.click(
                guardar_dataset,
                inputs=[editor, ruta_dataset],
                outputs=[estado_dataset],
            )

        with gr.Tab("Versiones"):
            tabla = gr.Dataframe(
                headers=["Nombre", "Base", "Tipo", "Estado", "Creada"],
                value=refrescar_versiones(),
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                refrescar_btn = gr.Button("Refrescar")
                version_activar = gr.Textbox(label="Versión a activar")
                activar_btn = gr.Button("Activar", variant="primary")
                rollback_btn = gr.Button("Rollback")
            estado_versiones = gr.Markdown("")
            with gr.Accordion("Reconstruir Modelfile", open=False):
                version_nueva = gr.Textbox(label="Nueva versión", value="nitshio-v2")
                crear_check = gr.Checkbox(label="Ejecutar ollama create", value=False)
                activar_check = gr.Checkbox(label="Activar al terminar", value=False)
                dataset_modelo = gr.Textbox(
                    label="Dataset usado (opcional)",
                    value=str(servicio.dataset_dir / "synthetic.jsonl"),
                )
                construir_btn = gr.Button("Construir Modelfile")
                estado_modelo = gr.Markdown("")
            refrescar_btn.click(refrescar_versiones, outputs=[tabla])
            activar_btn.click(
                activar_version, inputs=[version_activar], outputs=[estado_versiones]
            )
            rollback_btn.click(hacer_rollback, outputs=[estado_versiones])
            construir_btn.click(
                construir,
                inputs=[version_nueva, crear_check, activar_check, dataset_modelo],
                outputs=[estado_modelo],
            )

        with gr.Tab("Nube"):
            gr.Markdown(
                "## Fine-tune en Kaggle/Colab\n\n"
                f"**Notebook:** `{plan['notebook']}`\n\n"
                f"**Dataset:** `{plan['dataset']}`\n\n"
                f"{plan['message']}\n\n"
                "```powershell\n"
                f"{plan['command']}\n"
                "```\n\n"
                "Después de descargar el GGUF:\n\n"
                "```powershell\n"
                "ollama create nitshio-vN -f models/Modelfile\n"
                "python training/evaluate.py --compare-base --record-metrics\n"
                "```"
            )
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UI Gradio de Nitshio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        from brain.config import cargar_config

        config = cargar_config(args.config)
        servicio = TrainingService(config)
        demo = build_ui(servicio)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
