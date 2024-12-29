# archivo: crear_audiolibro_acelerado.py

from gtts import gTTS
from pydub import AudioSegment

def crear_audiolibro(archivo_texto, archivo_salida="audiolibro_rapido.mp3", velocidad=1.2):
    """
    Lee texto de 'archivo_texto', lo convierte a audio usando gTTS,
    y luego acelera el audio con pydub al 'velocidad' indicado.
    Guarda el resultado final en 'archivo_salida'.
    """

    # 1. Leer el texto del archivo
    with open(archivo_texto, 'r', encoding='utf-8') as f:
        texto = f.read()

    # 2. Crear el objeto gTTS y guardar temporalmente el audio en MP3
    tts = gTTS(text=texto, lang='es')
    tts.save("temp_audiolibro.mp3")
    print("Audio base generado con gTTS...")

    # 3. Aumentar la velocidad del audio usando pydub
    print(f"Acelerando el audio en un {int((velocidad-1)*100)}%...")
    audio = AudioSegment.from_file("temp_audiolibro.mp3", format="mp3")
    faster_audio = audio.speedup(playback_speed=velocidad)

    # 4. Exportar el audio final
    faster_audio.export(archivo_salida, format="mp3")
    print(f"Audiolibro final creado: {archivo_salida}")

if __name__ == "__main__":
    # Ejemplo de uso
    crear_audiolibro(
        archivo_texto="texto.txt",
        archivo_salida="audiolibro_rapido.mp3",
        velocidad=1.1
    )