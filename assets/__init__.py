import sys
from pathlib import Path

# Add the assets folder to the Python path
assets_path = Path(__file__).parent / 'assets'
sys.path.append(str(assets_path))

from .putting_probabilities import probabilities

