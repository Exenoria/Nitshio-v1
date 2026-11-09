# VSeeFace — avatar, VMC y audio

> Estado: implementado y validado con audio real. El lip-sync funciona con el
> **proveedor viejo** de VSeeFace; el proveedor nuevo no movía la boca en las
> pruebas realizadas.

Nitshio envía emociones a VSeeFace por VMC/OSC y manda el audio del TTS a
VB-Cable. VSeeFace lee ese audio desde `CABLE Output` y hace el lip-sync. Los
visemas `A/I/U/E/O` por VMC **no se usan**: en las pruebas no movieron la boca y
el lip-sync por audio los sobrescribía.

## 1. Requisitos

- VSeeFace instalado y abierto.
- Un avatar `.vrm` con grupos de blendshapes VRM, como `Neutral`, `A/I/U/E/O`,
  `Blink` y las emociones que quieras usar.
- VB-Cable instalado.
- Nitshio con la voz Piper descargada.

> Si el avatar no declara grupos VRM, las emociones no se verán aunque el VMC
> llegue. Usa un avatar con grupos.

## 2. Configuración de VSeeFace

1. Abre VSeeFace y carga el avatar.
2. Activa la recepción de **VMC Protocol** (`VMCReceiveEnable=1`). Comprueba
   que escucha en `127.0.0.1:39539`.
3. Entra en **Configuración General** y activa **Lipsync por audio**
   (`AudioLipSync=1`).
4. En el micrófono de VSeeFace selecciona
   **`CABLE Output (VB-Audio Virtual Cable)`**. Ese es el audio que Nitshio
   escribe en `CABLE Input`.
5. En **Configuración General → Opciones avanzadas**, activa
   **“Usar el modelo viejo para lip sync (menor rendimiento)”**
   (`OldLipSyncProvider=1`). Es el ajuste que hizo funcionar la boca en este
   equipo; con el proveedor nuevo el audio llegaba pero la boca no se movía.
6. Cierra y vuelve a abrir VSeeFace si cambiaste el micrófono o el proveedor.
   Los ajustes deben persistir en el archivo `settings.ini` de VSeeFace, bajo
   `%USERPROFILE%\AppData\LocalLow\<proveedor>\VSeeFace\`:

   ```ini
   Microphone=CABLE Output (VB-Audio Virtual Cable)
   AudioLipSync=1
   OldLipSyncProvider=1
   VMCReceiveEnable=1
   ```

> No edites `settings.ini` a mano para cambiar el micrófono: VSeeFace puede
> revertirlo al arrancar. Hazlo desde su interfaz.

### Nombres de VB-Cable

| Dirección | Dispositivo de Windows | Uso |
|---|---|---|
| Reproducción | `CABLE Input (VB-Audio Virtual Cable)` | Nitshio escribe aquí el TTS. |
| Grabación | `CABLE Output (VB-Audio Virtual Cable)` | VSeeFace lo lee como micrófono. |

El nombre puede variar si tienes varias instancias de VB-Cable. Ajusta
`tts.audio.output_device` en `config/config.yaml` con el nombre exacto que
muestre `run.py --mode diagnose`. El reproductor resuelve el dispositivo por
nombre, no por índice, porque los índices cambian.

## 3. Configuración de Nitshio

En `config/config.yaml`:

```yaml
avatar:
  enabled: true
  host: "127.0.0.1"
  port: 39539
  address_prefix: "/VMC/Ext"
  tick_hz: 60

tts:
  audio:
    output_device: "CABLE Input"
```

Las emociones y sus blendshapes están en `config/emotions.yaml`. Los nombres
deben coincidir con los del avatar VRM (`Neutral`, `Fun`, `Angry`, `Joy`,
`Sorrow`, `Surprised`).

## 4. Prueba de emociones

Sin VSeeFace, comprueba primero el mapeo:

```powershell
.\.venv\Scripts\python.exe scripts\test_emotions.py --all --dry-run
```

Con VSeeFace abierto:

```powershell
.\.venv\Scripts\python.exe scripts\test_emotions.py --all
```

El script recorre las seis emociones, imprime los valores máximos y finales y
vuelve a neutral. Si no se ve nada, revisa el puerto VMC, que el avatar tenga
grupos VRM y el firewall.

## 5. Prueba de conversación

```powershell
# Texto con voz y avatar
.\.venv\Scripts\python.exe run.py --mode text --speak

# Voz con avatar (habla por el micrófono real)
.\.venv\Scripts\python.exe run.py --mode voice
```

Para aislar problemas:

```powershell
# Sin avatar
.\.venv\Scripts\python.exe run.py --mode text --once "Hola" --no-avatar

# Sin TTS
.\.venv\Scripts\python.exe run.py --mode text --once "Hola" --no-tts
```

### Prueba integral sin hablar al micrófono

`scripts/test_voice_roundtrip.py` inyecta una frase sintetizada con Piper en un
micrófono falso, usa Moonshine real, el orquestador real (Ollama + memoria) y
el TTS/avatar reales, y graba `CABLE Output` para comprobar que el audio salió:

```powershell
.\.venv\Scripts\python.exe scripts\test_voice_roundtrip.py --no-tools --json
```

Resultado esperado: transcripción no vacía, respuesta JSON válida, emoción del
turno, chunks de TTS y `CABLE Output` con RMS máximo > 0.001. El audio de
evidencia se guarda en `data/cache/audio/voice_roundtrip_cable.wav`.

## 6. Diagnóstico

```powershell
.\.venv\Scripts\python.exe run.py --mode diagnose
```

Revisa:

- que `CABLE Input` aparezca en la lista de salidas;
- que la voz Piper exista;
- que el puerto VMC sea el correcto;
- que Ollama responda.

## 7. Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| El avatar no cambia de emoción | VMC desactivado, puerto distinto o avatar sin grupos VRM | Activa VMC, revisa `avatar.port` y usa un VRM con blendshapes. |
| Se oye pero no mueve la boca | VSeeFace no escucha `CABLE Output` o usa el proveedor nuevo | Selecciona `CABLE Output` como micrófono y activa “Usar el modelo viejo para lip sync”. |
| La boca se mueve a destiempo | Micrófono equivocado o buffer del cable | Revisa que VSeeFace use `CABLE Output` y que el TTS salga por `CABLE Input`. |
| No se oye nada | `output_device` no coincide | Copia el nombre exacto de `diagnose` o usa el índice. |
| Error `No se pudo cargar la voz Piper` | Falta el `.onnx` o el `.onnx.json` | Ejecuta `scripts/download_voice.py`. |
| Emoción desconocida | Nombre fuera de `emotions.yaml` | Añade la emoción o usa una existente. |
| VMC no llega | Firewall o VSeeFace en otra máquina | Usa `127.0.0.1`, revisa firewall y el puerto. |
| El avatar se ve pero la captura sale en blanco | La ventana está minimizada | Restaura VSeeFace antes de capturar. |

## 8. Nota sobre lip-sync

La ruta activa es: TTS → `CABLE Input` → `CABLE Output` → micrófono de
VSeeFace → lip-sync por audio. `avatar/lip_sync.py` existe como utilidad
opcional para enviar visemas por VMC, pero **no se activa automáticamente**:
en las pruebas, los visemas VMC no movieron la boca y el lip-sync por audio del
proveedor viejo sí. No actives ambos a la vez.

## 9. Evidencia de las pruebas

- `data/cache/screens/caras_montage.png` y `lipsync3_montage.png`: la boca se
  abre y cierra siguiendo el TTS.
- `data/cache/audio/voice_roundtrip_cable.wav`: audio real capturado de
  `CABLE Output` durante la prueba integral.
- Ajustes persistidos tras cerrar y reiniciar VSeeFace:
  `AudioLipSync=1`, `OldLipSyncProvider=1`,
  `Microphone=CABLE Output (VB-Audio Virtual Cable)`.
