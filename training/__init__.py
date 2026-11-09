"""Herramientas de entrenamiento, evaluación y versionado de Nitshio.

El entrenamiento real (QLoRA) se ejecuta solo en Kaggle/Colab. Este paquete
contiene el núcleo local que prepara, valida y evalúa los datos, además de los
scripts que se ejecutan en la nube.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.3.0"
