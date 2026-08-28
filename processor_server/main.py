import logging
import socket

# Не импортировать qdrant/upload_data на уровне модуля: HuggingFace + fastembed съедают RAM,
# после чего падает даже import json_parser (Errno 12).

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def receive_all(conn):
    buffer = b""
    while True:
        try:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buffer += chunk
        except Exception as e:
            log.warning("Ошибка чтения из сокета: %s", e)
            break
    return buffer


if __name__ == "__main__":
    from processor_server.qdrant import QdrantManager
    from processor_server.upload_data import upload_data

    log.info("Подключаюсь к Qdrant...")
    qm = QdrantManager()

    log.info("Слушаю сокет на 0.0.0.0:3030")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 3030))
    s.listen(1)

    while True:
        conn, addr = s.accept()
        with conn:
            data = receive_all(conn)
            if not data:
                log.info("Пустое соединение, пропуск")
                continue
            try:
                upload_data(qm, data)
            except Exception as e:
                # Один битый батч не должен ронять сервер и не должен помечаться обработанным на стороне бота.
                log.exception("upload_data упал: %s", e)
                try:
                    conn.sendall(f"Upload failed: {e}".encode("utf-8"))
                except OSError:
                    pass
                continue
            conn.sendall(b"Upload successful")
