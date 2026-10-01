"""Ejecuta el notebook celda por celda con nbclient, guardando un checkpoint del .ipynb
despues de cada celda (para poder inspeccionar progreso/resultados parciales sin esperar a que
termine la ejecucion completa) e imprimiendo progreso a stdout en tiempo real.
"""
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

PATH = "nootebooks/laboratorio_6_transfer_learning.ipynb"

nb = nbformat.read(PATH, as_version=4)
nb.metadata["kernelspec"] = {
    "name": "python3", "display_name": "Python 3", "language": "python",
}

client = NotebookClient(nb, timeout=-1, kernel_name="python3", allow_errors=False)

t_start = time.time()
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        is_code = cell.cell_type == "code"
        if is_code:
            print(f"[{time.strftime('%H:%M:%S')}] >>> celda {i}/{len(nb.cells)-1} ...", flush=True)
        t0 = time.time()
        try:
            client.execute_cell(cell, i)
        except CellExecutionError as e:
            print(f"[{time.strftime('%H:%M:%S')}] !!! ERROR en celda {i}: {e}", flush=True)
            nbformat.write(nb, PATH)
            sys.exit(1)
        if is_code:
            print(f"[{time.strftime('%H:%M:%S')}] <<< celda {i} completada en {time.time()-t0:.1f}s "
                  f"(total transcurrido {time.time()-t_start:.1f}s)", flush=True)
        nbformat.write(nb, PATH)

print(f"Notebook ejecutado completo en {time.time()-t_start:.1f}s", flush=True)
