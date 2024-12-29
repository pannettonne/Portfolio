import gradio as gr
from TTS.api import TTS
import soundfile as sf
import os
print(f"Directorio de trabajo actual: {os.getcwd()}")


# Cargar el modelo XTTS-v2
MODEL_NAME = "coqui/xtts-v2"
tts = TTS(model_name=MODEL_NAME)

def generar_audio(texto, fuente_audio, audio_grabado):
    if not texto.strip():
        return "Por favor, proporciona un texto válido."

    if fuente_audio == "Archivo de audio existente":
        ruta_audio = "audio_usuario.wav"
        print("hola")
    elif fuente_audio == "Grabar con micrófono" and audio_grabado is not None:
        ruta_audio = "audio_grabado.wav"
        sf.write(ruta_audio, audio_grabado[1], audio_grabado[0])
    else:
        return "Por favor, proporciona una fuente de audio válida."

    # Generar síntesis de voz utilizando el audio de referencia
    print(f"ruta audio {ruta_audio}")
    archivo_salida = "voz_generada2.wav"
    tts.tts_to_file(
        text=texto,
        file_path=archivo_salida,
        speaker_wav=ruta_audio,  # Usa la voz de referencia
        language="es"  # Especifica el idioma español
    )

    return archivo_salida

# Definir la interfaz de Gradio
descripcion = """
# Generación de Voz con Coqui XTTS-v2
1. Selecciona la fuente de audio: "Archivo de audio existente" o "Grabar con micrófono".
2. Si seleccionas "Grabar con micrófono", graba tu voz.
3. Escribe el texto que deseas convertir a audio.
4. Haz clic en "Submit" para generar el audio.
5. Escucha el resultado.
"""

with gr.Blocks() as demo:
    fuente_audio = gr.Radio(
        choices=["Archivo de audio existente", "Grabar con micrófono"],
        label="Selecciona la fuente de audio",
        value="Archivo de audio existente"
    )
    audio_input = gr.Audio(sources="microphone", label="Graba tu voz", visible=False)
    texto_input = gr.Textbox(lines=5, placeholder="Escribe aquí el texto que quieres convertir a audio...", label="Texto a convertir")
    boton_submit = gr.Button("Submit")
    audio_output = gr.Audio(label="Audio Generado")

    def actualizar_visibilidad(fuente):
        return gr.update(visible=fuente == "Grabar con micrófono")

    fuente_audio.change(fn=actualizar_visibilidad, inputs=fuente_audio, outputs=audio_input)
    boton_submit.click(fn=generar_audio, inputs=[texto_input, fuente_audio, audio_input], outputs=audio_output)

if __name__ == "__main__":
    demo.launch()