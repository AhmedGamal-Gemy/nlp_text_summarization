"""
Evaluation entry point.

Delegates to src.services.evaluation module.
"""

import sys
from pathlib import Path

# Add project root to path so we can import src as a package
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the main function from src package
from src.services.evaluation import main

if __name__ == "__main__":
    main()