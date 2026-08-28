"""Транскрипция аудио (faster-whisper tiny). Модель грузится лениво."""

import io
import logging
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)

_MODEL_NAME = "tiny"


@lru_cache(maxsize=1)
def _get_model():
    from faster_whisper import WhisperModel

    logger.info("Загружаю Whisper (%s)...", _MODEL_NAME)
    return WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")


# system-dependency: ffmpeg должен быть в PATH (в Docker-образе ставится apt-ом).
def any_to_wav(file_path) -> bytes:
    proc = subprocess.Popen(
        ["ffmpeg", "-i", str(file_path), "-ar", "16000", "-ac", "1",
         "-f", "wav", "pipe:1", "-loglevel", "quiet"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        wav, _ = proc.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        logger.warning("ffmpeg завис на %s — прерван по таймауту", file_path)
        return b""
    return wav or b""


def audio_file_analyzer(file_path):
    wav = any_to_wav(file_path)
    if not wav:
        return ""
    model = _get_model()
    with io.BytesIO(wav) as buf:
        segments, _ = model.transcribe(buf, language="ru")
    return "".join(seg.text for seg in segments).strip()
