import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

SCRP_CITIES = PROJECT_ROOT / "data" / "scraped_cities.txt"
LINKS_FOLDER = PROJECT_ROOT / "data" / "raw"
