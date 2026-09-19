"""Execute each current notebook in a fresh kernel and save its outputs."""
import os
import sys
import asyncio
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
os.environ['IPYTHONDIR'] = str(ROOT / '.ipython-verify')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.jupyter-verify')
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

for notebook_path in sorted((ROOT / 'notebooks').glob('[0-9][0-9]_*.ipynb')):
    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(notebook, timeout=240, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.write(notebook, notebook_path)
    print(f'PASS {notebook_path.name}', flush=True)
