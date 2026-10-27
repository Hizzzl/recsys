"""Execute every cell and preserve the notebook even if a cell fails."""

import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
path = root / "experiments.ipynb"
notebook = nbformat.read(path, as_version=4)


def report(cell_index, **kwargs):
    print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)


client = NotebookClient(notebook, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(root)}}, on_cell_execute=report)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print(f"Executed {path}", flush=True)
