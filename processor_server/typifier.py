import magic, filetype, mimetypes
from pathlib import Path


def detect_extension(path):
    kind = filetype.guess(path)
    if kind:
        return f'.{kind.extension}'

    try:
        mime = magic.from_file(path, mime=True)
        ext = mimetypes.guess_extension(mime)
        if ext:
            return ext
    except Exception:
        pass

    ext = Path(path).suffix
    return ext
