import os
from pathlib import Path

def is_city_processed(city, processed_file):
    if not os.path.exists(processed_file):
        return False
    with open(processed_file, 'r') as f:
        return city in f.read().splitlines()

def mark_city_processed(city, processed_file):
    with open(processed_file, 'a') as f:
        f.write(city + '\n') 

def clean_car_links(file_path):
    """Reads a .txt file containing car URLs, removes duplicates, empty lines,

    and invalid spaces, then updates the file and returns a list of clean URLs.

    :param file_path: Path object or string path to the links .txt file
    :return: list of cleaned unique URLs
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Warning: Link file not found at {file_path}")
        return []
 
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
 
    cleaned_urls = []
    seen = set()

    for line in raw_lines:
        url = line.strip()

        # Check for valid non-empty URLs starting with http
        if url and url.startswith("http") and url not in seen:
            seen.add(url)
            cleaned_urls.append(url)

    # Overwrite the file with cleaned, deduplicated URLs
    file_path.write_text("\n".join(cleaned_urls) + "\n", encoding="utf-8")

    return cleaned_urls
