import sys
import os
from pathlib import Path

# Add project root to Python path so 'app' module is importable by Vercel
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app.main import app
