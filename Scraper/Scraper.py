import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

def start_browser():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scrape_injuries_safely():
    driver = start_browser()

    base_url = "https://www.prosportstransactions.com/basketball/Search/SearchResults.php"
    base_params = "?Player=&Team=&BeginDate=2020-04-01&EndDate=&ILChkBx=yes&Submit=Search&start="

    injuries = []
    total_pages = 465  # Planned pages
    increment = 25     # Each page shows 25 rows

    for idx, start in tqdm(enumerate(range(0, increment * total_pages, increment)), total=total_pages, desc="Scraping Pages"):
        url = base_url + base_params + str(start)
        driver.get(url)

        try:
            WebDriverWait(driver, 6).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.datatable.center"))
            )
        except:
            print(f"⚠️ Table not found at start={start}. Skipping.")
            continue

        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table", {"class": "datatable center"})

        if table:
            rows = table.find_all('tr')[1:]  # Skip header
            for tr in rows:
                cols = tr.find_all('td')
                if len(cols) == 5:
                    injuries.append({
                        "Date": cols[0].get_text(strip=True),
                        "Team": cols[1].get_text(strip=True),
                        "Acquired": cols[2].get_text(strip=True),
                        "Relinquished": cols[3].get_text(strip=True),
                        "Notes": cols[4].get_text(strip=True),
                    })
        else:
            print(f"⚠️ No table content found at start={start}.")

        # Every 100 pages: chill a little
        if idx > 0 and idx % 100 == 0:
            print(f"⏳ Cooling down at page {idx}...")
            time.sleep(5)

        # Every 150 pages: restart the browser to avoid memory leaks
        if idx > 0 and idx % 150 == 0:
            driver.quit()
            driver = start_browser()
            print(f"♻️ Restarted browser at page {idx}...")

    driver.quit()

    # ── Save to CSV ──
    injuries_df = pd.DataFrame(injuries)
    if not injuries_df.empty:
        injuries_df['Date'] = pd.to_datetime(injuries_df['Date'], errors='coerce')

    injuries_df.to_csv("injuries_scraped_fast_safe.csv", index=False)
    print(f"\n✅ Successfully scraped and saved {len(injuries_df)} injury records to injuries_scraped_fast_safe.csv!")

if __name__ == "__main__":
    scrape_injuries_safely()
