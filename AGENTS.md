# AGENTS.md — reglas para cualquier sesión de OpenCode

Este proyecto se construye por fases. Antes de tocar nada, lee:

1. `docs/PLAN.md` — estado y orden de las fases.
2. `docs/ARCHITECTURE.md` — cómo encajan los módulos.
3. `docs/CHANGELOG.md` — qué se cambió por última vez.
4. `docs/DECISIONS.md` — decisiones ya tomadas y sus motivos.

## Regla obligatoria de documentación

**Cada vez que se modifique, añada o elimine algo, en la misma tarea hay que:**

- Añadir una entrada en `docs/CHANGELOG.md` con fecha, qué cambió, por qué y
  archivos afectados.
- Actualizar la documentación correspondiente (`README.md`,
  `docs/ARCHITECTURE.md`, `docs/TRAINING.md`, `docs/EXTENDING.md`,
  `docs/TROUBLESHOOTING.md`, `docs/VSEEFACE.md`, etc.).
- Actualizar `docs/PLAN.md` si cambia el alcance o se cierra una fase.
- Ejecutar `python scripts/check_docs.py` antes de dar una tarea por terminada.

Una tarea no está terminada si el changelog o la documentación quedaron atrás.

## Reglas técnicas

- **Nada de nombres de modelo hardcodeados.** Se leen de `config/config.yaml`.
- **Nada de secretos en el código ni en el YAML.** Solo `.env` (ver
  `.env.example`).
- **No borrar ni modificar el borrador `AppData/Roaming/npm/nova`.** Nitshio se
  construye desde cero; si se importa algo, se copia, no se altera el original.
- **No tocar archivos fuera de `allowed_paths`.** Toda herramienta de archivos
  debe validar rutas, symlinks y tamaño.
- **Borrado por defecto con confirmación** (`confirm_delete: true`), papelera y
  log. No cambiar el default sin decisión explícita del usuario.
- **Toda herramienta nueva es un plugin** en `tools/` con esquema y descripción.
- **Toda emoción nueva** se declara en `config/emotions.yaml`; el parser lee la
  lista de ahí.
- **Toda llamada a Ollama** pasa por `brain/llm/ollama_client.py`.
- **Cada módulo debe tener tests mínimos** en `tests/` y docstrings claros.
- **Idioma:** código y comentarios en español; nombres de API en inglés cuando
  sea natural.
- **Antes de decir “terminado”:** ejecutar tests y el script de comprobación que
  corresponda; pegar el resultado real, no suponerlo.

## Flujo de trabajo por fase

1. Implementar.
2. Probar (unitario + integración real cuando aplique).
3. Actualizar changelog, docs y plan.
4. Mostrar al usuario: qué se hizo, cómo probarlo, resultado y qué sigue.
5. No avanzar a la siguiente fase si la actual no pasó sus pruebas.
