import subprocess
import psutil
import time
import numpy as np


qubits = list(range(1, 36))
shots = 10


def create_cqasm_file(n):
    cqasm = f"""version 1.0

qubits {n}

measure_all
"""
    with open(f"/scratch/memory_cqasm_{n}.qc", "w+") as f:
        f.write(cqasm)


if __name__ == "__main__":
    data = {}
    for n in qubits:
        mems = []
        create_cqasm_file(n)

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
