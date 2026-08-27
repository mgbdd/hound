from pathlib import Path
from tempfile import TemporaryDirectory
from processor_server.analyzers.image_file_analyzer import image_file_analyzer
from processor_server.finder_soffice import find_soffice

import subprocess, shlex
import docx2txt


def docx_file_analyzer(file_path, img_limit=None):
    src = Path(file_path)
    with TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        if src.suffix.lower() == ".docx":
            text = docx2txt.process(src, tmpdir)

        elif src.suffix.lower() == ".doc":
            soffice_bin = find_soffice()
            subprocess.check_call([soffice_bin, "--headless",
                                   "--convert-to", "docx", "--outdir", tmpdir, src])
            converted = tmpdir / (src.stem + ".docx")
            text = docx2txt.process(converted, tmpdir)
        # TODO Подумать над сохранением картинок, их как отдельные вектора или все к одному документу
        if img_limit is not None:
            for i, pic in enumerate(tmpdir.iterdir()):
                if pic.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}:
                    if i >= img_limit:
                        break

                    text = text + "\n" + f"![image {i + 1}]({pic.name}): {image_file_analyzer(pic) or 'нет описания'}"
                    # text = text + "\n" + f"<IMAGE {i+1} src=\"{pic.name}\">{image_file_analyzer(pic) or 'нет описания'}</IMAGE>"

        return text
