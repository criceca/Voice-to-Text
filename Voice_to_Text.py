from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import os
import torchaudio
import soundfile as sf

# Configuración de dispositivos
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el dataset desde Hugging Face
dataset_url = "charris/hubert_process_filter_spotify"
dataset = load_dataset(dataset_url, split="train")


# Cargar modelo y procesador Whisper
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)

model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Directorio de salida para archivos descargados y transcripciones
output_dir = "./audio_files"
os.makedirs(output_dir, exist_ok=True)

# Función para transcribir un archivo de audio
def transcribe_audio(audio_path):
    # Leer archivo de audio
    speech_array, sampling_rate = torchaudio.load(audio_path)

    # Procesar audio
    inputs = processor(speech_array[0], sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Generar transcripción
    inputs = inputs.to(device)
    generated_ids = model.generate(inputs)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Descargar y transcribir archivos de audio
transcriptions = []
for i, audio_data in enumerate(dataset["audio"]):
    print(f"Procesando archivo {i+1}/{len(dataset)}")

    # Extraer datos de audio
    audio_array = audio_data["array"]
    sampling_rate = audio_data["sampling_rate"]

    # Guardar el archivo de audio
    audio_path = os.path.join(output_dir, f"audio_{i}.wav")
    sf.write(audio_path, audio_array, sampling_rate)

    # Transcribir el archivo
    transcription = transcribe_audio(audio_path)
    transcriptions.append({"audio_file": audio_path, "transcription": transcription})

# Guardar las transcripciones en un archivo
with open(os.path.join(output_dir, "transcriptions.txt"), "w") as f:
    for item in transcriptions:
        f.write(f"Archivo: {item['audio_file']}\nTranscripción: {item['transcription']}\n\n")

print("Proceso completado. Las transcripciones están disponibles en transcriptions.txt.")