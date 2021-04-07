#! /usr/bin/env python3

import numpy as np
import os
from pathlib import Path

from qaoa_maxcut import QAOA, init_random_graph, init_params, set_qi_authentication, \
                        DEFAULT_QI_API_URL

OUT_PATH = Path(os.path.dirname(__file__)) / ".." / "data" / "qx_benchmark_cqasm"

if __name__ == "__main__":
    p = 1
    d = 2
    seed = 0xc0ffee

    np.random.seed(seed)

    params = init_params(p)

    set_qi_authentication("c15a7bdafcb71f7a3462c8515b23d64b7c3392d3",
                          DEFAULT_QI_API_URL)

    ns = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

    for n in ns:
        G = init_random_graph(d, n, seed)

        qaoa = QAOA(G, n, p=p, use_qi=True, qi_backend_type="QX single-node simulator")
        cqasm = qaoa.get_circuit_cqasm(params)

        with open(OUT_PATH / f"qaoa_n{n}_p{p}_d{d}.cqasm", "w") as f:
            f.write(cqasm)
