import shutil

def find_soffice() -> str:
    for name in ("soffice.com", "soffice.exe", "soffice"):
        p = shutil.which(name)
        if p:
            return p
    raise FileNotFoundError(
        "LibreOffice не найден. Установите пакет libreoffice и убедитесь, "
        "что бинарник soffice доступен в PATH."
    )