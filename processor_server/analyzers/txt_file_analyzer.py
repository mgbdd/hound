def txt_file_analyzer(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    return content.strip()
