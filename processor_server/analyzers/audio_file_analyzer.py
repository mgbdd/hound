import io
from faster_whisper import WhisperModel
import subprocess

MODEL_NAME = "tiny"
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


# TODO: Ставить ffmpeg на сервер
# system‑dependency: ffmpeg должен быть в PATH!
def any_to_wav(file_path):
    proc = subprocess.Popen(
        ["ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1",
         "-f", "wav", "pipe:1", "-loglevel", "quiet"],
        stdout=subprocess.PIPE
    )
    wav, _ = proc.communicate()
    return wav


def audio_file_analyzer(file_path):
    wav = any_to_wav(file_path)

    with io.BytesIO(wav) as buf:
        segments, _ = model.transcribe(buf, language="ru")
    text = "".join(seg.text for seg in segments).strip()

    return text
