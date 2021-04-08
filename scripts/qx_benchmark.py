#! /usr/bin/env python3

import subprocess
import argparse
import resource
import os
import time

import numpy as np

def main(args):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args["omp_threads"])
    env["KMP_AFFINITY"] = "granularity=fine,scatter"
    # env["OMP_PLACES"] = "cores"
    # env["OMP_PROC_BIND"] = "spread"

    result = {
        "cpu_times": [],
        "shared_mems": [],
        "unshared_mems": [],
    }

    for _ in range(args["reps"]):
        call_args = ["qx-simulator", args["cqasm"], "1", str(args["omp_threads"])]
        print(f"Running {call_args}")

        usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        start_time = time.time()
        subprocess.call(call_args, env=env)
        end_time = time.time()
        usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)

        result["cpu_times"].append(end_time - start_time)
        result["shared_mems"].append(usage_end.ru_ixrss - usage_start.ru_ixrss)
        result["unshared_mems"].append(usage_end.ru_idrss - usage_start.ru_idrss)

    print(result)

    np.savez(args["output"], **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QX simulator benchmark script",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("output", type=str, help="output file to write results to")
    parser.add_argument("cqasm", type=str, help="path to the cQASM file to benchmark")
    parser.add_argument("omp_threads", type=int, help="value to set OMP_NUM_THREADS to")
    parser.add_argument("--reps", type=int, default=5,
                        help="number of repititions to measure and average")

    args = parser.parse_args()

    main(vars(args))
