#!/usr/bin/env python3
"""Entry point for HoVer claim/evidence graph generation."""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(130)
    except Exception as error:
        print(f"\nFatal error: {error}")
        sys.exit(1)
