# src/app/hardware.py
"""Audio input helpers for real-time streaming."""

import socket
import os


class UdpAudioSource:
    def __init__(
        self,
        host="0.0.0.0",
        port=5555,
        chunk_frames=1024,
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
        timeout_s=0.5,
    ):
        self.host = host
        self.port = port
        self.chunk_frames = chunk_frames
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.timeout_s = timeout_s
        self.debug = os.environ.get("DRONE_DEBUG", "0") == "1"

        self._socket = None
        self._running = False
        self._last_client_addr = None
        self._display_host = socket.gethostbyname(socket.gethostname())

    @property
    def chunk_bytes(self):
        return self.chunk_frames * self.channels * self.sample_width_bytes

    def start(self):
        if self._running:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(self.timeout_s)
        self._running = True

    def read_chunk(self):
        if not self._running or self._socket is None:
            return None
        try:
            data, addr = self._socket.recvfrom(self.chunk_bytes)
            if data:
                self._last_client_addr = addr
            return data
        except socket.timeout:
            return None
        except (OSError, socket.error) as e:
            # Socket may be closed by stop() in another thread
            if self.debug:
                print(f"[UDP] Socket error in read_chunk: {e}")
            self._running = False
            return None

    def read_available(self, max_reads=50):
        if not self._running or self._socket is None:
            return []
        chunks = []
        try:
            for _ in range(max_reads):
                try:
                    data, addr = self._socket.recvfrom(self.chunk_bytes)
                except socket.timeout:
                    break
                if not data:
                    break
                self._last_client_addr = addr
                chunks.append(data)
        except (OSError, socket.error) as e:
            # Socket may be closed by stop() in another thread
            if self.debug:
                print(f"[UDP] Socket error in read_available: {e}")
            self._running = False
        return chunks

    def stop(self):
        self._running = False
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception as e:
                if self.debug:
                    print(f"[UDP] Error closing socket: {e}")
            self._socket = None

    def get_device_info(self):
        if self._last_client_addr:
            return f"{self._last_client_addr[0]}:{self._last_client_addr[1]}"
        return f"{self._display_host}:{self.port}"
