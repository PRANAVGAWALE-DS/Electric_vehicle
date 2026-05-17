# conftest.py  (project root)
# Ensures the project root is on sys.path so `import api.main` resolves
# regardless of how pytest is invoked.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
