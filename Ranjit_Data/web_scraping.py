url = "https://m.dailyhunt.in/news/india/assamese/news?mode=pwa&action=click"

"""
Assamese paragraph scraper with resume & checkpointing.

Behavior:
- Reads last saved paragraph from assamese_paragraphs.txt (if any).
- Loads the target URL using Firefox profile.
- Scrolls the page until the last paragraph is visible (or gives up after MAX_FIND_ATTEMPTS).
- Scrolls further until content stabilizes.
- Appends new paragraphs every 10 scroll attempts (checkpointing).
- Finally appends any remaining new paragraphs.
"""

import os
import re
import time
import shutil
import tempfile
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from bs4 import BeautifulSoup

# ------------- User settings -------------
profile_path = "/home/ranjit/.mozilla/firefox/luddqo6a.default-release"
# url = "https://m.dailyhunt.in/news/india/assamese/tv9assamese-epaper-tvassame/baksat+163+dhava+javi+pvshasanvh+kvib+novave+pvtibad+dib+novave+shlogan+bibhinn+dishat+nishedhagya+-newsid-n685230825?listname=topicsList&index=1&topicIndex=0&mode=pwa&action=click"
output_file = r"Ranjit_Data/assamese_paragraphs.txt"

HEADLESS = False
WAIT_BETWEEN_SCROLLS = 5     # Just makes each scroll slower (waits 5 seconds before the next scroll). No effect on the logic.
MAX_FIND_ATTEMPTS = 800000000     # Only affects how long it keeps searching for your last saved paragraph before giving up. Doesn’t affect checkpointing.
MAX_SCROLL_ATTEMPTS = 300     # Sets the total possible scrolls before stopping. Checkpoints will still happen at every 10th scroll (so roughly 30 checkpoints max).
STABLE_REQUIRED = 10     # Makes it more patient before deciding the page is “fully loaded.” Still unrelated to checkpointing.
CHECKPOINT_EVERY = 10     # Works exactly as intended.

# -----------------------------------------

assamese_pattern = re.compile(r"[\u0980-\u09FF]")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def safe_copy_profile(src_path: str) -> str:
    if not os.path.isdir(src_path):
        raise FileNotFoundError(f"profile path not found: {src_path}")
    tmpdir = tempfile.mkdtemp(prefix="fh_profile_")
    dest = os.path.join(tmpdir, "profile_copy")

    def ignore_lock_files(directory, contents):
        return [f for f in contents if f in ('lock', 'parent.lock')]

    shutil.copytree(src_path, dest, ignore=ignore_lock_files)
    return dest

def start_driver_with_profile(copied_profile_path: str):
    options = Options()
    options.headless = HEADLESS
    profile_obj = FirefoxProfile(copied_profile_path)
    options.profile = profile_obj
    driver = webdriver.Firefox(options=options)
    return driver

def find_last_saved_paragraph(file_path: str):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return None
    blocks = [normalize_text(b) for b in re.split(r'\n\s*\n', content) if normalize_text(b)]
    if not blocks:
        return None
    return blocks[-1]

def extract_paragraphs_in_order(soup):
    tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    paragraphs = []
    for tag in tags:
        txt = tag.get_text(separator=" ", strip=True)
        txt = normalize_text(txt)
        if txt and assamese_pattern.search(txt):
            paragraphs.append(txt)
    return paragraphs

def scroll_once(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    driver.execute_script("""
        const sel = document.querySelector('.infinite-scroll-component') ||
                    document.querySelector('.infinite-scroll-component__outerdiv') ||
                    document.querySelector('[data-infinite-scroll]') ||
                    document.querySelector('div[style*="overflow:auto"]');
        if (sel) { sel.scrollTop = sel.scrollHeight; }
    """)

def scroll_until_paragraph_found(driver, target_norm_text):
    print("Searching for last saved paragraph on the loaded page...")
    if not target_norm_text:
        print("No previous paragraph found; starting fresh.")
        return True
    attempt = 0
    while attempt < MAX_FIND_ATTEMPTS:
        attempt += 1
        scroll_once(driver)
        time.sleep(WAIT_BETWEEN_SCROLLS)
        html = driver.page_source
        if target_norm_text in normalize_text(BeautifulSoup(html, "html.parser").get_text(separator=" ")):
            print(f"Found last-saved paragraph on attempt #{attempt}.")
            return True
        if attempt % 10 == 0:
            print(f"...still searching (attempt {attempt}/{MAX_FIND_ATTEMPTS})")
    print("Warning: last saved paragraph not found after max attempts.")
    return False

def append_new_paragraphs_to_file(file_path: str, existing_last_norm: str, all_paragraphs):
    start_index = 0
    if existing_last_norm:
        try:
            start_index = all_paragraphs.index(existing_last_norm) + 1
        except ValueError:
            start_index = 0

    new_paras = all_paragraphs[start_index:]
    if not new_paras:
        return 0

    with open(file_path, "a", encoding="utf-8") as fout:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            fout.write("\n")
        for p in new_paras:
            fout.write(p + "\n\n")
    return len(new_paras)

def scroll_and_checkpoint(driver, output_file, last_saved_norm):
    last_height = driver.execute_script("return document.body.scrollHeight")
    stable_count = 0
    scroll_attempts = 0
    appended_total = 0

    while scroll_attempts < MAX_SCROLL_ATTEMPTS:
        scroll_attempts += 1
        scroll_once(driver)
        time.sleep(WAIT_BETWEEN_SCROLLS)

        # checkpoint
        if scroll_attempts % CHECKPOINT_EVERY == 0:
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = extract_paragraphs_in_order(soup)
            appended = append_new_paragraphs_to_file(output_file, last_saved_norm, paragraphs)
            if appended:
                last_saved_norm = paragraphs[-1]
                appended_total += appended
                print(f"[Checkpoint] Scroll {scroll_attempts}: appended {appended} paragraph(s).")

        # check for stable scroll
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            stable_count += 1
            if stable_count >= STABLE_REQUIRED:
                print(f"Reached stable bottom after {scroll_attempts} scrolls.")
                break
        else:
            stable_count = 0
            last_height = new_height

    # final append
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = extract_paragraphs_in_order(soup)
    appended = append_new_paragraphs_to_file(output_file, last_saved_norm, paragraphs)
    appended_total += appended
    print(f"Total new paragraphs appended in this run: {appended_total}")

def main():
    last_saved = find_last_saved_paragraph(output_file)
    if last_saved:
        print("Last saved paragraph (preview):")
        print(last_saved[:200] + ("..." if len(last_saved) > 200 else ""))
    else:
        print("No existing output file found or file empty — starting fresh.")

    print("Copying Firefox profile (to avoid lock issues)...")
    copied_profile = safe_copy_profile(profile_path)

    try:
        driver = start_driver_with_profile(copied_profile)
    except Exception as e:
        raise RuntimeError("Failed to start Firefox webdriver. Check geckodriver/selenium and profile path.") from e

    try:
        print("Loading URL:", url)
        driver.get(url)
        time.sleep(WAIT_BETWEEN_SCROLLS + 1.0)

        scroll_until_paragraph_found(driver, last_saved)
        scroll_and_checkpoint(driver, output_file, last_saved)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print("Scraping complete.")

if __name__ == "__main__":
    main()
