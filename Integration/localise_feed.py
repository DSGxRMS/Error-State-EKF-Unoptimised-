# localise_feed.py
import socket, json, threading, time
from collections import deque

class LocaliseClient:
    """
    Subscribes to UDP JSON datagrams from localise.py (predictions only).
    Message schema: {"t": float, "x": float, "y": float, "psi": float, "v": float}
    """
    def __init__(self, host="127.0.0.1", port=5601):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # bind to receive from localise.py; use SO_REUSEADDR so restart is easy
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self._latest = {"t": 0.0, "x": 0.0, "y": 0.0, "psi": 0.0, "v": 0.0}
        self._lock = threading.Lock()
        self._running = False
        self._thr = None
        self._yaw_hist = deque(maxlen=4)

    def start(self):
        if self._running: return
        self._running = True
        self._thr = threading.Thread(target=self._rx_loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._running = False
        try: self.sock.close()
        except: pass

    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self.sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                with self._lock:
                    self._latest = msg
                    self._yaw_hist.append((msg.get("t", 0.0), msg.get("psi", 0.0)))
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.02)

    def get_state(self):
        """
        Returns (t, x, y, psi, v).
        """
        with self._lock:
            m = dict(self._latest)
        return m.get("t", 0.0), m.get("x", 0.0), m.get("y", 0.0), m.get("psi", 0.0), m.get("v", 0.0)
