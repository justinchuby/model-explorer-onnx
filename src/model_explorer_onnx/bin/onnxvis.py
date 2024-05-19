#!/usr/bin/env python
"""A shortcut to run model explorer with ONNX extension."""

import subprocess
import sys


def main():
    # Run model explorer
    subprocess.run(
        [
            "model-explorer",
            "--extensions",
            "model_explorer_onnx",
            *sys.argv[1:],
        ]
    )


if __name__ == "__main__":
    main()
