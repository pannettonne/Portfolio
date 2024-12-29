import gradio as gr
from TTS.api import TTS
import soundfile as sf

# Carga del modelo multi-speaker y multilingüe (por ejemplo, YourTTS)
MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
tts = TTS(model_name=MODEL_NAME)

def clonar_voz_y_generar_audio(audio, texto, idioma):
    if audio is None or not texto.strip():
        return "Por favor, proporciona un audio y texto válidos."

    # Desempaqueta la tupla
    sample_rate, audio_data = audio

    # Define la ruta del archivo de audio temporal
    ruta_audio = "audio_usuario.wav"

    # Guarda el audio en un archivo WAV
    sf.write(ruta_audio, audio_data, sample_rate)
    print(f"Audio guardado en: {ruta_audio}")

    # Generar síntesis de voz utilizando el embedding del usuario
    archivo_salida = "voz_clonada.wav"
    tts.tts_to_file(
        text=texto,
        file_path=archivo_salida,
        speaker_wav=ruta_audio,  # Usa la voz del usuario como referencia
        language=idioma  # Especifica el idioma deseado
    )

    return archivo_salida

# Definir la interfaz de Gradio
descripcion = """
# Clonación de Voz con Coqui TTS
1. Graba tu voz utilizando el botón "Record".
2. Escribe el texto que quieres que sea leído con tu voz clonada.
3. Selecciona el idioma deseado.
4. Escucha el resultado.
"""

demo = gr.Interface(
    fn=clonar_voz_y_generar_audio,
    inputs=[
        gr.Audio(sources="microphone", value="file"),
        gr.Textbox(lines=5, placeholder="Escribe aquí el texto que quieres leer..."),
        gr.Dropdown(choices=["es", "en", "fr", "de"], label="Idioma")  # Lista de idiomas soportados
    ],
    outputs="audio",
    title="Clonación de Voz con Coqui",
    description=descripcion,
)

if __name__ == "__main__":
    demo.launch()