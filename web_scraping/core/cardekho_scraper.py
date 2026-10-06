import json
import os
import random
import time
from bs4 import BeautifulSoup as soup
from splinter import Browser
from tqdm import tqdm


def collect_car_links(city, total_pages, target_cars, output_file):
    cars_collected = 0

    existing_links = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_links = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(existing_links)} existing links.")

    browser = Browser("chrome")

    # Initialize tqdm for page iteration
    pbar = tqdm(range(1, total_pages + 1), desc=f"Collecting Links ({city})")

    with open(output_file, "a") as f:
        for page_num in pbar:
            if page_num == 1:
                url = f"https://www.cardekho.com/used-cars+in+{city}"
            else:
                url = (
                    f"https://www.cardekho.com/used-cars+in+{city}/page-{page_num}"
                )

            browser.visit(url)
            time.sleep(2)
            browser.execute_script("window.scrollTo(0, 1000);")
            time.sleep(1)

            current_soup = soup(browser.html, "html.parser")
            page_links_found = 0

            for link in current_soup.find_all("a", href=True):
                href = link["href"]

                if "used-car-details" in href:
                    full_url = (
                        f"https://www.cardekho.com{href}"
                        if href.startswith("/")
                        else href
                    )

                    if full_url not in existing_links:
                        f.write(full_url + "\n")
                        existing_links.add(
                            full_url
                        )  # track in set for O(1) lookup
                        cars_collected += 1
                        page_links_found += 1

            # Update progress bar metadata on every page
            pbar.set_postfix(
                {
                    "New Links": page_links_found,
                    "Total Collected": cars_collected,
                }
            )

            if cars_collected >= target_cars:
                tqdm.write(f"\nTarget of {target_cars} links reached!")
                break

    browser.quit()
    print(
        f"All links saved in {output_file}. Total unique: {len(existing_links)}"
    )


def scrape_car_links(links_file, output_file):
    def get_browser():
        return Browser("chrome")

    if not os.path.exists(links_file):
        print(f"File not found: {links_file}")
        return

    with open(links_file, "r") as f:
        all_links = [line.strip() for line in f.readlines() if line.strip()]

    browser = get_browser()

    # Progress bar wrapping link iteration
    pbar = tqdm(enumerate(all_links), total=len(all_links), desc="Scraping Cars")

    for index, link in pbar:
        try:
            # Visit page
            browser.visit(link)

            # Human delay + scroll
            time.sleep(random.uniform(1, 2))
            browser.execute_script("window.scrollTo(0, 600);")
            time.sleep(0.5)

            # Expand specifications
            try:
                view_all_spec_btn = browser.find_by_text(
                    "View all Specifications"
                )
                if view_all_spec_btn:
                    browser.execute_script(
                        "arguments[0].click();",
                        view_all_spec_btn.first._element,
                    )
                    time.sleep(0.6)
            except Exception:
                pass

            # Parse page
            page_soup = soup(browser.html, "html.parser")
            car_data = {"url": link}

            # -------------------------
            # CAR NAME EXTRACTION
            # -------------------------
            name_tag = page_soup.find("div", class_="vehicleName")
            h1 = (
                name_tag.find("h1")
                if (name_tag and name_tag.find("h1"))
                else page_soup.find("h1")
            )

            if h1:
                parts = h1.get_text(separator="|", strip=True).split("|")
                car_data["car_name"] = (
                    parts[1].strip() if len(parts) >= 2 else parts[0].strip()
                )

            # -------------------------
            # PRICE EXTRACTION
            # -------------------------
            price_div = page_soup.find("div", class_="vehiclePrice")
            if price_div:
                price_span = price_div.find("span")
                if price_span:
                    car_data["Price"] = price_span.get_text(strip=True)

            # -------------------------
            # SPECIFICATIONS EXTRACTION
            # -------------------------
            spec_items = page_soup.find_all("li", class_="gsc_col-xs-12")

            for item in spec_items:
                label_tag = item.find("div", class_="label")
                value_tag = item.find("span", class_="value-text")

                if label_tag and value_tag:
                    label = label_tag.get_text(strip=True)
                    value = value_tag.get_text(strip=True)
                    car_data[label] = value

            # Save data
            if len(car_data) > 1:
                with open(output_file, "a") as out:
                    out.write(json.dumps(car_data) + "\n")

                # Update live stats on the progress bar display
                pbar.set_postfix(
                    {
                        "Price": car_data.get("Price", "N/A"),
                        "Specs": len(car_data) - 2,
                    }
                )
            else:
                tqdm.write(f"No data found for: {link}")

        except Exception as e:
            tqdm.write(f"Error at index {index+1} ({link}): {e}")

            browser.quit()
            browser = get_browser()
            time.sleep(1)
            continue

    browser.quit()
    