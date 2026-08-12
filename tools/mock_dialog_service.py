#!/usr/bin/env python3
"""Local deterministic service for HTTP, llama.cpp adapter, and TTS tests."""

import argparse
import json
import math
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            request = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400, "invalid JSON")
            return

        if self.path == "/v1/chat/completions":
            messages = request.get("messages", [])
            valid = (
                messages
                and messages[-1].get("role") == "user"
                and request.get("chat_template_kwargs", {}).get(
                    "enable_thinking"
                )
                is False
            )
            if not valid:
                self.send_error(400, "invalid llama.cpp request")
                return
            if request.get("stream") is not True:
                payload = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "LLAMA_CPP_OK",
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            payload = (
                'data: {"choices":[{"delta":{"role":"assistant",'
                '"content":null}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"LLAMA_"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"CPP_OK"}}]}\n\n'
                'data: {"choices":[{"finish_reason":"stop",'
                '"delta":{}}]}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/text":
            payload = json.dumps(
                {"reply": "网络服务正常。"}, ensure_ascii=False
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/text-sse":
            payload = (
                'data: {"delta":"网络服务"}\n\n'
                'data: {"delta":"正常。"}\n\n'
                'data: {"done":true}\n\n'
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/tts":
            sample_rate = 16000
            samples = bytearray()
            for index in range(3200):
                value = int(
                    0.08
                    * 32767
                    * math.sin(2.0 * math.pi * 220.0 * index / sample_rate)
                )
                samples.extend(struct.pack("<h", value))
            payload = bytes(samples)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_error(404)

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
