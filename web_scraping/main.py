import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import SCRP_CITIES, LINKS_FOLDER
from core.cardekho_scraper import collect_car_links, scrape_car_links
from utils.helpers import is_city_processed, mark_city_processed, clean_car_links

def main():
    city = "Varanasi"
    target_cars = 100
    total_pages = 10

    if city is None or target_cars is None or total_pages is None:
        print("Please set the 'city', 'target_cars', and 'total_pages' variables before running the script.")
        return
    
    if is_city_processed(city, SCRP_CITIES):
        print(f"{city} has already been processed. Skipping.")
        return

    else:
        collect_car_links(city, total_pages, target_cars, f"{LINKS_FOLDER}/{city}_car_links.txt")
        clean_car_links(f"{LINKS_FOLDER}/{city}_car_links.txt")
        scrape_car_links(f"{LINKS_FOLDER}/{city}_car_links.txt", f"{LINKS_FOLDER}/{city}_car_data.json")
        mark_city_processed(city, SCRP_CITIES)
        return 

if __name__ == "__main__":
    main()
    