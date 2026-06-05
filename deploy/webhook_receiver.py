import json
import logging
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

PORT = 9000
# Optional: Verify secret token in headers (X-Gitea-Token)
SECRET_TOKEN = os.getenv("GITEA_WEBHOOK_SECRET", "")

class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if SECRET_TOKEN:
            token = self.headers.get("X-Gitea-Token")
            if token != SECRET_TOKEN:
                logging.warning("Unauthorized webhook request (token mismatch)")
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"Unauthorized")
                return

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode('utf-8'))
            ref = payload.get("ref", "")
            repo = payload.get("repository", {}).get("name", "")
            logging.info(f"Received webhook for repository '{repo}', ref '{ref}'")

            # Deploy on main branch pushes
            if "refs/heads/main" in ref:
                logging.info("Triggering deployment script...")
                deploy_script = os.path.join(
                    os.getenv("SERVO_SKULL_HOME", "/opt/servo-skull"), "deploy", "deploy.sh"
                )
                subprocess.Popen(["/bin/bash", deploy_script])
                self.send_response(202)
                self.end_headers()
                self.wfile.write(b"Deployment triggered")
            else:
                logging.info(f"Skipping deployment for ref: {ref}")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"No action taken")
        except Exception as e:
            logging.error(f"Error parsing webhook payload: {e}")
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Bad request")

def run():
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, WebhookHandler)
    logging.info(f"Webhook receiver listening on port {PORT}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Webhook receiver stopped.")

if __name__ == "__main__":
    run()
