#! /usr/bin/env python3

import subprocess
import argparse
import resource
import os
import time

import numpy as np


fat_soil_affinities = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36,  # node 0
    1, 5, 9, 13, 17, 21, 25, 29, 33, 37,  # node 1
    2, 6, 10, 14, 18, 22, 26, 30, 34, 38, # node 2
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39  # node 3
]


def get_core_affinities(n):
    return ",".join(map(str, fat_soil_affinities[:n]))


def main(args):
    affinities = get_core_affinities(args["omp_threads"])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args["omp_threads"])
    env["KMP_AFFINITY"] = f"granularity=fine,explicit,proclist=[{affinities}]"
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
        print(f"\twith env: {env}")

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
