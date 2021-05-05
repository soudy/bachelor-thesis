import subprocess
import psutil
import time
import numpy as np

from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile, assemble

from quantuminspire.qiskit import QI
from quantuminspire.qiskit.backend_qx import QuantumInspireBackend

from qaoa_maxcut import set_qi_authentication, DEFAULT_QI_API_URL


qubits = list(range(1, 36))
shots = 5
depth = 2


def create_cqasm_file(backend, n):
    circ = random_circuit(n, depth, measure=True, max_operands=2, seed=55)

    (experiment,) = assemble(transpile(circ, backend=backend),
                                backend=backend).experiments

    cqasm = QuantumInspireBackend._generate_cqasm(
        experiment,
        full_state_projection=False
    )

    print(cqasm)

    with open(f"/scratch/memory_cqasm_{n}.qc", "w+") as f:
        f.write(cqasm)


if __name__ == "__main__":
    set_qi_authentication("c15a7bdafcb71f7a3462c8515b23d64b7c3392d3",
                          DEFAULT_QI_API_URL)
    backend = QI.get_backend("QX single-node simulator")

    data = {}
    for n in qubits:
        mems = []
        create_cqasm_file(backend, n)

        for _ in range(shots):
            peak_memory = 0
            proc = subprocess.Popen(f"qx-simulator /scratch/memory_cqasm_{n}.qc 1 40", shell=True)
            INTERVAL = 0.0001

            while proc.poll() is None:
                p = psutil.Process(proc.pid)
                rss_usage = p.memory_info().rss
                if rss_usage > peak_memory:
                    peak_memory = rss_usage

            mems.append(peak_memory)

        print(f"{n} qubit mems: {mems}")

        data[n] = mems
        print(data)
        np.savez("/home/soud/mem_usage_qx.npz", data=data)
