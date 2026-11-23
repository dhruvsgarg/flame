"""Dead-simple TCP echo server: whatever it receives, it sends back.
This is the "dummy TCP server" Checkpoint A wants -- not MQTT yet,
just something Toxiproxy can sit in front of so we can measure timing.
Should have no toxicity added to network
"""
import socket

HOST = "127.0.0.1"
PORT = 9999

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"echo server listening on {HOST}:{PORT}")
    while True:
        conn, addr = server.accept()
        with conn:
            data = conn.recv(65536)
            if data:
                conn.sendall(data)

if __name__ == "__main__":
    main()