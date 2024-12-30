# Voice to Text with Whisper

Este repositorio contiene un script en Python que realiza transcripción de audio a texto utilizando el modelo Whisper de OpenAI. El script descarga y procesa archivos de audio de un dataset alojado en Hugging Face, y genera transcripciones automáticas.

## Características
- **Modelo utilizado:** Whisper-small de OpenAI.
- **Dataset:** charris/hubert_process_filter_spotify desde Hugging Face.
- **Procesamiento de audio:** torchaudio y soundfile.


## Requisitos
Antes de ejecutar el script, asegúrese de tener instaladas las siguientes dependencias:

```bash
pip install torch torchaudio transformers datasets soundfile
```

Además, para aprovechar la aceleración por GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Estructura del Proyecto
```
.
|-- Voice_to_Text.py    # Script principal
|-- audio_files/        # Directorio donde se guardan archivos descargados y transcripciones
```

## Uso
1. Clonar el repositorio:
```bash
git clone <URL_DEL_REPOSITORIO>
cd <nombre_del_repositorio>
```

2. Ejecutar el script:
```bash
python Voice_to_Text.py
```

El script cargará un dataset de audio, procesará los archivos y generará transcripciones.

## Configuración
El script permite ajustar el nombre del modelo, el dataset y el directorio de salida modificando las siguientes variables:

```python
model_name = "openai/whisper-small"
dataset_url = "charris/hubert_process_filter_spotify"
output_dir = "./audio_files"
```

## Agradecimientos
Este proyecto utiliza el modelo Whisper de OpenAI y datasets de Hugging Face. Agradecemos a las comunidades de código abierto por sus contribuciones.

## Licencia
[MIT License](LICENSE)

