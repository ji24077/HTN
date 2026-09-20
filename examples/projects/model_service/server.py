"""Template: install vLLM and place the model at MODEL_PATH on eligible workers."""
import os
import subprocess
import sys
from pathlib import Path

MODEL_PATH = '/models/my-finetune'  # Change to the pinned model directory on your workers.

subprocess.run([
    str(Path(sys.executable).with_name('vllm')), 'serve', MODEL_PATH,
    '--host', '127.0.0.1', '--port', os.environ['DISPATCH_SERVICE_PORT'],
], check=True)
