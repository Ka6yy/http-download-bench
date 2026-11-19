#!/usr/bin/env python3
"""Backward-compatible entry point: `python main.py URL ...`.

The implementation lives in the `httpbench` package. Installing the project
(`pip install .`) additionally provides the `httpbench` console script; see
pyproject.toml.
"""

import sys

from httpbench.cli import main

if __name__ == "__main__":
    sys.exit(main())
