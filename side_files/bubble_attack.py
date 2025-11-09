"""
# tabs open all at once:

python bubble_tester.py \
  --url "https://cute.codeshop.in/" \
  --profile "/home/ranjit/.mozilla/firefox/luddqo6a.default-release" \
  --tabs 8 \
  --duration 60


"""

import argparse, os, random, shutil, tempfile, time, traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# =========================
# Driver setup
# =========================
def start_driver_with_profile(copied_profile_path: str, headless: bool = False):
    options = Options()
    options.headless = headless
    profile_obj = FirefoxProfile(copied_profile_path)
    options.profile = profile_obj
    # Slightly faster loads
    options.set_preference("webdriver.load.strategy", "eager")
    return webdriver.Firefox(options=options)

# =========================
# Selectors from your DOM
# =========================
NAME_INPUT_SEL = "#nameInput"
START_BTN_SEL  = "#makeItCuteButton"
SCORE_POPUP_ID = "scorePopup"

# =========================
# Basic actions
# =========================
def set_username(driver, username):
    try:
        el = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, NAME_INPUT_SEL)))
        el.clear()
        el.send_keys(username)
        return True
    except Exception:
        return False

def click_start(driver):
    # Exact button first
    try:
        btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, START_BTN_SEL)))
        btn.click()
        return True
    except Exception:
        pass
    # Fallback: text contains "start"/"play"
    try:
        buttons = driver.find_elements(By.XPATH, "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'start') or contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'play')]")
        for b in buttons:
            if b.is_displayed():
                try:
                    b.click()
                    return True
                except Exception:
                    driver.execute_script("arguments[0].click();", b)
                    return True
    except Exception:
        pass
    return False

def find_html_bubbles(driver):
    sels = [
        "[class*='bubble']",
        ".bubble,.bubble-item",
        ".ball,.dot,.circle,.blob"
    ]
    found = []
    for sel in sels:
        try:
            for e in driver.find_elements(By.CSS_SELECTOR, sel):
                if e.is_displayed() and e.size.get("width",0) > 5 and e.size.get("height",0) > 5:
                    found.append(e)
        except Exception:
            pass
    uniq, seen = [], set()
    for e in found:
        try:
            k = (e.tag_name, e.location.get('x',0), e.location.get('y',0), e.size.get('width',0), e.size.get('height',0))
            if k not in seen:
                seen.add(k); uniq.append(e)
        except Exception:
            continue
    return uniq

def find_svg_bubbles(driver):
    candidates = []
    try:
        circles = driver.find_elements(By.CSS_SELECTOR, "svg circle, svg g[class*='bubble']")
        for e in circles:
            if e.is_displayed():
                candidates.append(e)
    except Exception:
        pass
    return candidates

def click_elem(driver, e):
    try:
        e.click(); return True
    except Exception:
        try:
            driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('click', {bubbles:true}))", e); return True
        except Exception:
            try:
                ActionChains(driver).move_to_element(e).click().perform(); return True
            except Exception:
                return False

def canvas_click_fallback(driver, clicks_per_second):
    try:
        canvases = driver.find_elements(By.TAG_NAME, "canvas")
        if not canvases:
            return False
        c = canvases[0]
        rect = driver.execute_script("return arguments[0].getBoundingClientRect();", c)
        if not rect or rect["width"] < 10 or rect["height"] < 10:
            return False
        x0, y0 = rect["left"]+3, rect["top"]+3
        w, h = rect["width"]-6, rect["height"]-6
        rx = x0 + random.randint(0, int(max(1,w)))
        ry = y0 + random.randint(0, int(max(1,h)))
        driver.execute_script(
            "var ev = new MouseEvent('click',{clientX:arguments[0],clientY:arguments[1],bubbles:true});"
            "document.elementFromPoint(arguments[0],arguments[1]).dispatchEvent(ev);",
            rx, ry
        )
        time.sleep(max(0.001, 1.0/clicks_per_second))
        return True
    except Exception:
        return False

def one_burst_clicking(driver, cps, burst_seconds=0.4):
    """
    Perform a short burst of clicking in the current tab (HTML/SVG/canvas),
    for ~burst_seconds. Keeps each tab 'active' without long blocking.
    """
    end = time.time() + burst_seconds
    while time.time() < end:
        did = False
        for e in find_html_bubbles(driver):
            if click_elem(driver, e):
                did = True
                time.sleep(max(0.0005, 1.0/cps))
                if time.time() >= end: break
        if not did:
            for e in find_svg_bubbles(driver):
                if click_elem(driver, e):
                    did = True
                    time.sleep(max(0.0005, 1.0/cps))
                    if time.time() >= end: break
        if not did:
            canvas_click_fallback(driver, cps)
        time.sleep(0.008)

def read_score_if_any(driver):
    score = None
    try:
        cand = driver.find_elements(By.XPATH, "//*[contains(translate(@id,'SCORE','score'),'score') or contains(translate(@class,'SCORE','score'),'score') or contains(translate(normalize-space(text()),'SCORE','score'),'score')]")
        for c in cand:
            if c.is_displayed():
                score = c.text.strip()
                break
    except Exception:
        pass
    if not score:
        try:
            pop = driver.find_element(By.ID, SCORE_POPUP_ID)
            if pop.is_displayed() and pop.text.strip():
                score = pop.text.strip()
        except Exception:
            pass
    return score

# =========================
# New: open all tabs at once
# =========================
def open_tabs_all_at_once(driver, url, count):
    """
    Opens 'count' tabs immediately. The first tab uses the current window,
    others are opened via JS window.open(...).
    Returns the list of window handles (length == count).
    """
    # ensure first tab navigated
    driver.get(url)
    # blast open remaining tabs quickly
    to_open = max(0, count - 1)
    if to_open > 0:
        driver.execute_script(
            "for (let i=0;i<arguments[1];i++){window.open(arguments[0],'_blank');}", url, to_open
        )
    # wait for handles
    deadline = time.time() + 5
    while time.time() < deadline and len(driver.window_handles) < count:
        time.sleep(0.05)
    # return exactly count handles (best-effort)
    handles = driver.window_handles[:count]
    return handles

def prep_each_tab(driver, handles, username):
    """
    For each tab: switch, ensure page ready, set username, click Start.
    This is quick per tab to get every tab into 'playing' state.
    """
    for h in handles:
        driver.switch_to.window(h)
        # small wait for DOM
        time.sleep(0.25)
        set_username(driver, username)
        if not click_start(driver):
            time.sleep(0.5)
            click_start(driver)

def run_scheduler_over_tabs(driver, handles, duration, cps, burst_seconds=0.4):
    """
    Time-sliced scheduler: cycles across all tabs and gives each a short burst of clicking.
    Avoids threading while keeping all tabs active.
    """
    end = time.time() + duration
    i = 0
    while time.time() < end:
        h = handles[i % len(handles)]
        try:
            driver.switch_to.window(h)
        except Exception:
            # handle might have closed; rebuild list
            handles = [hh for hh in driver.window_handles if hh in handles]
            if not handles:
                break
            continue
        one_burst_clicking(driver, cps, burst_seconds=burst_seconds)
        i += 1

    # collect scores (optional)
    scores = []
    for h in handles:
        try:
            driver.switch_to.window(h)
            scores.append((h, read_score_if_any(driver)))
        except Exception:
            scores.append((h, None))
    return scores

# =========================
# Profile copy (for separate drivers)
# =========================
def copy_profile(src):
    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    tmp = tempfile.mkdtemp(prefix="ffprof_")
    dst = os.path.join(tmp, "profile")
    shutil.copytree(src, dst)
    return dst

# =========================
# Orchestration
# =========================
def orchestrate(url, profile, username="Bot Attack", tabs=1, separate_drivers=False, drivers=1, duration=30, cps=6.0, headless=False):
    """
    - If separate_drivers=False:
        Open ALL tabs at once, prep them, then run a tab scheduler.
    - If separate_drivers=True:
        Launch N drivers (with copied profiles), each runs solo.
    """
    if not separate_drivers:
        d = start_driver_with_profile(profile, headless=headless)
        try:
            handles = open_tabs_all_at_once(d, url, tabs)
            if not handles:
                raise RuntimeError("No tabs opened.")
            prep_each_tab(d, handles, username)
            scores = run_scheduler_over_tabs(d, handles, duration, cps, burst_seconds=0.45)
            print("[single-driver] Tab scores:", scores)
        finally:
            try: d.quit()
            except: pass
    else:
        # True concurrency with multiple Firefox instances
        drvs = []
        try:
            for i in range(drivers):
                try:
                    p = copy_profile(profile)
                except Exception:
                    p = profile
                d = start_driver_with_profile(p, headless=headless); drvs.append(d)
                d.get(url)
                time.sleep(0.3)
                set_username(d, username)
                click_start(d)
            # scheduler across multiple drivers (round-robin bursts)
            end = time.time() + duration
            idx = 0
            while time.time() < end and drvs:
                drv = drvs[idx % len(drvs)]
                try:
                    one_burst_clicking(drv, cps, burst_seconds=0.5)
                except Exception:
                    pass
                idx += 1
            # gather scores
            scores = []
            for d in drvs:
                try:
                    scores.append(read_score_if_any(d))
                except Exception:
                    scores.append(None)
            print("[multi-driver] Scores:", scores)
        finally:
            for d in drvs:
                try: d.quit()
                except: pass

# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bubble game tester (tabs opened all at once).")
    parser.add_argument("--url", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--username", default="ranjit")
    parser.add_argument("--tabs", type=int, default=3, help="Tabs to open at once in a single driver.")
    parser.add_argument("--separate-drivers", action="store_true", help="Use multiple Firefox instances instead of tabs.")
    parser.add_argument("--drivers", type=int, default=1)
    parser.add_argument("--duration", type=int, default=45)
    parser.add_argument("--cps", type=float, default=6.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    orchestrate(
        url=args.url,
        profile=args.profile,
        username=args.username,
        tabs=args.tabs,
        separate_drivers=args.separate_drivers,
        drivers=args.drivers,
        duration=args.duration,
        cps=args.cps,
        headless=args.headless
    )
