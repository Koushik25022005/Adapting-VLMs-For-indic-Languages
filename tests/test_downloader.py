"""Downloader behaviour against a real local HTTP server (fast, dead, HTML, trickling)."""
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from indicvlm import prepare_data as pdm

TA = "சென்னையில் இன்று கனமழை பெய்தது."


def _png(size=(120, 90), mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="PNG")
    return buf.getvalue()


PNG = _png()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/ok"):
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.end_headers()
            self.wfile.write(PNG)
        elif self.path.startswith("/html"):
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(b"<html>nope</html>")
        elif self.path.startswith("/trickle"):  # sends a byte per second, forever-ish
            self.send_response(200); self.send_header("Content-Type", "image/png"); self.end_headers()
            try:
                for _ in range(60):
                    self.wfile.write(b"x"); self.wfile.flush(); time.sleep(1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_fetch_image_reasons(server):
    img, r = pdm.fetch_image(f"{server}/ok/1.png")
    assert r is None and img.size == (120, 90) and img.mode == "RGB"
    assert pdm.fetch_image(f"{server}/missing.png")[1] == "http_error"
    assert pdm.fetch_image(f"{server}/html")[1] == "not_image"
    assert pdm.fetch_image("http://127.0.0.1:9/x.png", timeout=1)[1] == "connection"
    assert pdm.fetch_image(f"{server}/ok/1.png", min_side=200)[1] == "too_small"


def test_trickling_server_is_cut_off_at_deadline(server):
    t = time.monotonic()
    img, reason = pdm.fetch_image(f"{server}/trickle", timeout=3, deadline=2)
    took = time.monotonic() - t
    assert img is None and reason == "slow" and took < 5  # requests' timeout alone would wait ~60 s


def test_downscale_and_transparency():
    big = Image.new("RGBA", (3000, 1500), (0, 0, 0, 0))
    img, _ = pdm.fetch_image(big, max_side=1024)
    assert img.size == (1024, 512) and img.mode == "RGB"
    assert img.getpixel((0, 0)) == (255, 255, 255)  # transparent -> white, not black
    pal = Image.new("P", (100, 100)); pal.info["transparency"] = 0
    assert pdm.to_rgb(pal).mode == "RGB"


def test_slow_sites_do_not_stall_the_run(server, tmp_path, monkeypatch):
    """10 trickling URLs + 40 good ones with 8 workers: the good ones finish while the slow
    ones time out in parallel, so the run takes ~one deadline, not ten."""
    import datasets

    rows = []
    for i in range(50):
        path = f"/trickle/{i}" if i % 5 == 0 else f"/ok/{i}.png"
        rows.append({"image": f"{server}{path}", "text": f"{TA} {i}"})
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(datasets, "get_dataset_config_names", lambda n: ["Chitrakshara-Cap"])
    out = tmp_path / "o"
    t = time.monotonic()
    pdm.main(["--out", str(out), "--per-language", "40", "--languages", "Tamil", "--workers", "8",
              "--timeout", "2", "--deadline", "2", "--val-fraction", "0.05"])
    took = time.monotonic() - t
    stats = json.loads((out / "stats.json").read_text())
    assert stats["per_language"] == {"Tamil": 40}
    # slow downloads still running when the quota fills are abandoned, not waited for
    assert 1 <= stats["counters"]["failed_slow"] <= 10
    assert took < 10, took
