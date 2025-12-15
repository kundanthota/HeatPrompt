import json, time, urllib.parse, pathlib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------- 1.  CONFIG -------------------------------------------------
HTML_FILE = pathlib.Path("esri.html").absolute().as_uri()  # Your local HTML
DOWNLOAD_DIR = pathlib.Path("data/images").absolute()
DOWNLOAD_DIR.mkdir(exist_ok=True)

# ---------- 2.  LOAD INPUT FILE ---------------------------------------
with open("data/atlas_data/geometry_by_id.json", "r") as f:
    features = json.load(f)  # format: list of {"id": ..., "geom": [[...]]}

# ---------- 3.  START CHROME WITH AUTO-DOWNLOAD -----------------------
chrome_opts = webdriver.ChromeOptions()
prefs = {
    "download.default_directory": str(DOWNLOAD_DIR),
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_opts.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(options=chrome_opts)
driver.maximize_window()

# ---------- 4.  LOOP THROUGH FEATURES ----------------------------------
for key in features.keys():
    try:
        # Encode polygon and ID into URL
        poly_param = urllib.parse.quote(json.dumps(features[key][0], separators=(',', ':')))
        id_param = urllib.parse.quote(str(key))
        url = f"{HTML_FILE}?poly={poly_param}&id={id_param}"

        print(f"➡️  Processing ID {key}")

        driver.get(url)
        time.sleep(1.5)  # allow tiles to load

        # Save Left Map
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "saveLeft"))
        ).click()
        time.sleep(1)

        # Save Right Mask
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "saveRight"))
        ).click()
        time.sleep(1)

        print(f"✅  Saved image + mask for ID {key}")

    except Exception as e:
        print(f"❌  Failed on ID {key}: {e}")
        continue

# ---------- 5.  CLEANUP ------------------------------------------------
driver.quit()
print("🎉 All done.")
