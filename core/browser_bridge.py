"""
Browser Bridge for Colab Backend Bootstrapping.

Uses Playwright to launch a non-headless Chromium instance with an isolated 
local profile (`./colab_profile`). Navigates to Colab, waits for user login,
detects captchas, triggers 'Run All', and scrapes the Cloudflare tunnel URL.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from utils.logger import get_logger

logger = get_logger(__name__)


class ColabBridgeWorker(QThread):
    """
    Background worker that uses Playwright to interact with Colab and
    extract the Cloudflare tunnel URL.
    """
    status_changed = pyqtSignal(str)
    url_extracted = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, notebook_url: str, parent=None):
        super().__init__(parent)
        self.notebook_url = notebook_url
        self._running = True

    def run(self):
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import Error as PlaywrightError
            # Note: TargetClosedError is a type of PlaywrightError
        except ImportError:
            self.error_occurred.emit("Playwright not installed. Please run: pip install playwright")
            return

        with sync_playwright() as p:
            profile_dir = Path("colab_profile").absolute()
            
            self.status_changed.emit("Launching Playwright Chromium...")
            try:
                context = p.chromium.launch_persistent_context(
                    user_data_dir=str(profile_dir),
                    headless=False,
                    ignore_https_errors=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-site-isolation-trials",
                        "--disable-features=IsolateOrigins,site-per-process",
                    ],
                    viewport={"width": 1280, "height": 900}
                )
                page = context.pages[0] if context.pages else context.new_page()
            except Exception as e:
                self.error_occurred.emit(f"Failed to launch browser: {e}")
                return

            try:
                self.status_changed.emit("Navigating to Colab...")
                
                # Auto-accept any alerts or prompts
                page.on("dialog", lambda dialog: dialog.accept())
                
                page.goto(self.notebook_url, wait_until="domcontentloaded", timeout=60000)

                # --- 1. Auth Check ---
                try:
                    # Give it 10 seconds to see if the "Sign in" button exists
                    page.wait_for_selector("text='Sign in'", timeout=10000, state="attached")
                    self.status_changed.emit("⚠ Action Required: Log into Google in the browser window")
                    
                    try:
                        # Now block until the main Colab toolbar appears (meaning login is done)
                        # 5 minute timeout for the user to log in
                        page.wait_for_selector("#top-toolbar", timeout=300000, state="attached")
                        self.status_changed.emit("Login detected, continuing...")
                    except PlaywrightTimeoutError:
                        self.error_occurred.emit("Login timed out after 5 minutes.")
                        return
                except PlaywrightTimeoutError:
                    # No "Sign in" button found within 10s, assume already logged in
                    self.status_changed.emit("Session active, continuing...")

                # --- 2. Captcha Detection ---
                # Colab sometimes throws an iframe captcha "Are you a robot?"
                while self._running:
                    if page.locator("iframe[title*='recaptcha']").count() > 0 or page.locator("text='Are you a robot?'").count() > 0:
                        self.status_changed.emit("⚠ Action Required: Solve Captcha in browser...")
                        time.sleep(3)
                    else:
                        break

                if not self._running:
                    return

                # --- 3. Run All ---
                self.status_changed.emit("Executing 'Run All' (Ctrl+F9)...")
                # Ensure page has focus
                page.bring_to_front()
                # A small delay to ensure the UI is fully responsive
                time.sleep(2) 
                page.keyboard.press("Control+F9")

                # --- 4. URL Extraction Loop ---
                self.status_changed.emit("Running Colab cells...")
                
                start_time = time.time()
                url_found = None
                
                # Number of consecutive content() failures before giving up
                content_fail_count = 0

                while self._running and (time.time() - start_time) < 180: # 3 min timeout
                    try:
                        content = page.content()
                        content_fail_count = 0 # reset on success
                    except PlaywrightError as e:
                        if "Target closed" in str(e) or "Browser closed" in str(e):
                            self.error_occurred.emit("Browser window was closed. Boot aborted.")
                            return
                        else:
                            content_fail_count += 1
                            logger.debug(f"page.content() error: {e}")
                            if content_fail_count > 3:
                                self.error_occurred.emit(f"Failed to read page content repeatedly: {e}")
                                return
                            time.sleep(5)
                            continue
                    
                    # GPU Quota Check
                    if "Cannot connect to GPU backend" in content or "usage limit" in content.lower():
                        self.error_occurred.emit("Colab Error: GPU quota exceeded or unavailable.")
                        return

                    # Reconnect Check
                    reconnect_btn = page.locator("text='Reconnect'")
                    if reconnect_btn.count() > 0 and reconnect_btn.is_visible():
                        self.status_changed.emit("Clicking Reconnect...")
                        try:
                            # Use force=True to bypass any overlapping error dialogs (like <mwc-dialog id="error-dialog">)
                            reconnect_btn.click(force=True, timeout=5000)
                        except Exception as e:
                            logger.debug(f"Failed to click Reconnect: {e}")
                        time.sleep(2)
                    elif "Runtime disconnected" in content:
                         self.status_changed.emit("Runtime disconnected. Attempting to recover...")
                         # We can try to see if a reconnect button appears, or just wait
                         time.sleep(2)
                         
                    # JavaScript/Third-party cookies error check
                    if "Could not load the JavaScript files" in content:
                        self.status_changed.emit("Colab UI failed to load. Forcing reload...")
                        try:
                            page.reload(wait_until="domcontentloaded")
                            time.sleep(5)
                            continue
                        except Exception as e:
                            logger.error(f"Reload failed: {e}")

                    # Regex match
                    match = re.search(r"(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)", content)
                    if match:
                        url_found = match.group(1)
                        break
                        
                    time.sleep(5)

                if url_found:
                    self.status_changed.emit("✓ Tunnel URL extracted!")
                    self.url_extracted.emit(url_found)
                    
                    # Keep-alive loop to prevent browser from closing and Colab from terminating
                    while self._running:
                        try:
                            content = page.content()
                            if "Could not load the JavaScript files" in content:
                                self.status_changed.emit("Colab UI failed to load. Forcing reload...")
                                page.reload(wait_until="domcontentloaded")
                        except Exception:
                            pass
                        time.sleep(1)
                        
                elif self._running:
                    self.error_occurred.emit("Timed out waiting for tunnel URL.")

            except PlaywrightTimeoutError as e:
                self.error_occurred.emit(f"Navigation timed out: {e}")
            except Exception as e:
                self.error_occurred.emit(f"Bridge error: {e}")
                
            finally:
                # Clean up: close the context and browser
                try:
                    context.close()
                except:
                    pass

    def stop(self):
        self._running = False
