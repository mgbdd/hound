import os
import tempfile
import requests


def load_temp_files(file_url):
    if not file_url.startswith("http"):
        print(f"Невалидный URL: {file_url}")
        return None

    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, os.path.basename(file_url))

    try:
        response = requests.get(file_url)
        response.raise_for_status()
        with open(temp_file_path, 'wb') as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке файла {file_url}: {e}")
        return None

    return temp_file_path


def delete_temp_files(paths: list[str] | str | None) -> None:
    if paths is None:
        return
    if isinstance(paths, str):
        paths = [paths]

    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
