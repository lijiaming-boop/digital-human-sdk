#!/usr/bin/env python3
"""Local real TTS service compatible with HttpTTSClient.

Uses eSpeak NG for synthesis and FFmpeg for conversion to raw PCM. It is
intended for local development and deterministic end-to-end validation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class TTSHandler(BaseHTTPRequestHandler):
    server_version = "DigitalHumanEspeakTTS/1.0"

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        self._json(
            200,
            {
                "status": "ok",
                "engine": "espeak-ng",
                "voice": self.server.voice,
                "sample_rate": 16000,
                "channels": 1,
                "format": "pcm_s16le",
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/tts":
            self._json(404, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 64 * 1024:
                self._json(400, {"error": "invalid request body size"})
                return
            request = json.loads(self.rfile.read(content_length).decode("utf-8"))
            text = str(request.get("text", "")).strip()
            sample_rate = int(request.get("sample_rate", 16000))
            channels = int(request.get("channels", 1))
            audio_format = str(request.get("format", "pcm_s16le"))
            if not text:
                self._json(400, {"error": "text is required"})
                return
            if len(text) > 2000:
                self._json(400, {"error": "text is too long"})
                return
            if sample_rate != 16000 or channels != 1 or audio_format != "pcm_s16le":
                self._json(
                    400,
                    {"error": "only 16000 Hz mono pcm_s16le is supported"},
                )
                return

            wave = subprocess.run(
                [
                    "espeak-ng",
                    "-v",
                    self.server.voice,
                    "-s",
                    str(self.server.speed),
                    "--stdout",
                    text,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            ).stdout
            pcm = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "pipe:1",
                ],
                input=wave,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            ).stdout
            if not pcm:
                raise RuntimeError("TTS generated empty PCM")

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(pcm)))
            self.send_header("X-Audio-Sample-Rate", "16000")
            self.send_header("X-Audio-Channels", "1")
            self.send_header("X-Audio-Format", "pcm_s16le")
            self.end_headers()
            self.wfile.write(pcm)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "TTS synthesis timed out"})
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.decode("utf-8", errors="replace").strip()
            self._json(500, {"error": message or "TTS process failed"})
        except Exception as exc:  # local validation service boundary
            self._json(500, {"error": str(exc)})

    def log_message(self, message: str, *args: object) -> None:
        print(f"{self.address_string()} - {message % args}", flush=True)


class TTSServer(ThreadingHTTPServer):
    voice: str
    speed: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--voice", default="cmn")
    parser.add_argument("--speed", type=int, default=150)
    args = parser.parse_args()

    for dependency in ("espeak-ng", "ffmpeg"):
        if shutil.which(dependency) is None:
            raise SystemExit(f"missing dependency: {dependency}")

    server = TTSServer((args.host, args.port), TTSHandler)
    server.voice = args.voice
    server.speed = args.speed
    print(
        f"eSpeak NG TTS listening on http://{args.host}:{args.port} "
        f"voice={args.voice} speed={args.speed}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()