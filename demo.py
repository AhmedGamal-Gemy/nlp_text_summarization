"""
CLI demo entry point.

Delegates to src.ui.demo module.
"""

import sys
from pathlib import Path

# Add project root to path so we can import src as a package
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run from src package
from src.ui.demo import run_demo

if __name__ == "__main__":
    run_demo()