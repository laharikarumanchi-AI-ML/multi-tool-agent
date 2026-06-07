"""Allow `python -m multitool`."""
import sys

from multitool.cli import main

if __name__ == "__main__":
    sys.exit(main())
