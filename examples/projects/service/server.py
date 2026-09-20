"""Upload this file in Service mode. No third-party dependencies or GPU required."""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    healthy = True
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200 if Handler.healthy else 503)
            self.end_headers()
            self.wfile.write(b'ready')
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for index in range(3):
                self.wfile.write(f'data: {index}\n\n'.encode())
                self.wfile.flush()
                time.sleep(0.15)
        elif self.path == '/headers':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Set-Cookie', 'injected=bad')
            self.send_header('Link', '</a>; rel=first')
            self.send_header('Link', '</b>; rel=next')
            self.end_headers()
            self.wfile.write(json.dumps(dict(self.headers)).encode())
        elif self.path == '/delayed':
            time.sleep(5)
            self.send_response(200)
            self.end_headers()
        elif self.path == '/slow':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            try:
                for index in range(200):
                    self.wfile.write(f'data: {index}\n\n'.encode())
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == '/pid':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(str(os.getpid()).encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'message': 'Service is running', 'path': self.path}).encode())

    def do_POST(self):
        if self.path == '/unhealthy':
            Handler.healthy = False
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(('127.0.0.1', int(os.environ['DISPATCH_SERVICE_PORT'])), Handler).serve_forever()
