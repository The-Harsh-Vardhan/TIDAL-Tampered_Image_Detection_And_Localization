"""
W&B Sweep Dispatcher for Image Tampering Detection Ablation Study.

Called by `wandb agent`. Reads experiment_id from command-line args
(passed by sweep via ${args}), maps it to the corresponding pre-converted
Python module, and runs it in the current process so that the module's
wandb.init() inherits the sweep context from environment variables.

Usage (called automatically by wandb agent):
    python sweep_runner.py --experiment_id=P.3
"""

import argparse
import os
import sys
import runpy

EXPERIMENT_MODULES = {
    "P.0":    "vrp0.py",
    "P.1":    "vrp1.py",
    "P.1.5":  "vrp1_5.py",
    "P.2":    "vrp2.py",
    "P.3":    "vrp3.py",
    "P.4":    "vrp4.py",
    "P.5":    "vrp5.py",
    "P.6":    "vrp6.py",
    "P.7":    "vrp7.py",
    "P.8":    "vrp8.py",
    "P.9":    "vrp9.py",
    "P.10":   "vrp10.py",
    "P.11":   "vrp11.py",
    "P.12":   "vrp12.py",
    "P.13":   "vrp13.py",
    "P.14":   "vrp14.py",
    "P.15":   "vrp15.py",
    "P.16":   "vrp16.py",
    "P.17":   "vrp17.py",
    "P.18":   "vrp18.py",
    "P.19":   "vrp19.py",
    "P.20":   "vrp20.py",
    "P.21":   "vrp21.py",
    "P.22":   "vrp22.py",
    "P.23":   "vrp23.py",
    "P.24":   "vrp24.py",
    "P.25":   "vrp25.py",
    "P.26":   "vrp26.py",
    "P.27":   "vrp27.py",
    "P.28":   "vrp28.py",
}


def main():
    parser = argparse.ArgumentParser(description="W&B Sweep Runner")
    parser.add_argument("--experiment_id", type=str, required=True,
                        help="Experiment ID (e.g., P.3, P.10)")
    args = parser.parse_args()

    experiment_id = args.experiment_id
    if experiment_id not in EXPERIMENT_MODULES:
        print(f"ERROR: Unknown experiment_id '{experiment_id}'")
        print(f"Valid IDs: {sorted(EXPERIMENT_MODULES.keys())}")
        sys.exit(1)

    module_file = EXPERIMENT_MODULES[experiment_id]
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "modules", module_file)

    if not os.path.exists(module_path):
        print(f"ERROR: Module not found: {module_path}")
        print("Run convert_notebooks.py first to generate modules.")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  SWEEP RUNNER: Launching experiment {experiment_id}")
    print(f"  Module: {module_file}")
    print(f"  Sweep ID: {os.environ.get('WANDB_SWEEP_ID', 'N/A')}")
    print(f"{'='*60}")

    # Run the module in the current process.
    # wandb.init() inside the module detects WANDB_SWEEP_ID env var
    # and automatically joins the sweep.
    runpy.run_path(module_path, run_name="__main__")


if __name__ == "__main__":
    main()
