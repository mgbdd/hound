import socket

# Не импортировать qdrant/upload_data на уровне модуля: HuggingFace + fastembed съедают RAM,
# после чего падает даже import json_parser (Errno 12).


def receive_all(conn):
    buffer = b""
    while True:
        try:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buffer += chunk
        except Exception as e:
            print(f"Ошибка чтения: {e}")
            break
    return buffer


if __name__ == "__main__":
    from processor_server.qdrant import QdrantManager
    from processor_server.upload_data import upload_data

    print("Connecting to Qdrant...")
    qm = QdrantManager()

    print("Creating socket server on port 3030...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 3030))
    s.listen(1)

    while True:
        print("Waiting for connection...")
        conn, addr = s.accept()
        with conn:
            data = receive_all(conn)
            if not data:
                print("No data received")
                continue
            try:
                upload_data(qm, data)
            except Exception as e:
                # Один битый батч не должен ронять сервер и не должен помечаться обработанным на стороне бота.
                print(f"upload_data failed: {e}")
                try:
                    conn.sendall(f"Upload failed: {e}".encode("utf-8"))
                except OSError:
                    pass
                continue
            conn.sendall(b"Upload successful")
