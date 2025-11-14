#!/usr/bin/env python3
"""
Browser automation script to click the upvote container on YC launch page.
Requires: pip install playwright && playwright install
"""

from playwright.sync_api import sync_playwright
import time


def click_upvote_container():
    """Navigate to YC launch page and click the upvote container."""

    url = "https://www.ycombinator.com/launches/OqQ-tensr-building-autonomous-robotic-factories"

    with sync_playwright() as p:
        # Launch browser (headless=False to see what's happening)
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        try:
            print(f"Navigating to: {url}")
            page.goto(url, wait_until="networkidle")

            # Wait a bit for dynamic content to load
            time.sleep(2)

            # Find the upvote container by class
            selector = '.upvote-container.col.justify-end.launches-show'
            print(f"Looking for element with selector: {selector}")

            # Wait for the element to be visible
            page.wait_for_selector(selector, timeout=10000)

            # Get the element
            upvote_element = page.locator(selector)

            # Check if element exists
            count = upvote_element.count()
            print(f"Found {count} matching element(s)")

            if count > 0:
                # Scroll element into view
                upvote_element.scroll_into_view_if_needed()
                time.sleep(0.5)

                # Click the element
                print("Clicking the upvote container...")
                upvote_element.click()
                print("Click successful!")

                # Wait to see the result
                time.sleep(3)
            else:
                print("Element not found!")

                # Debug: print page content
                print("\nAvailable elements with 'upvote' in class:")
                upvote_elements = page.locator('[class*="upvote"]')
                for i in range(upvote_elements.count()):
                    elem = upvote_elements.nth(i)
                    print(f"  - {elem.get_attribute('class')}")

        except Exception as e:
            print(f"Error occurred: {e}")

            # Take a screenshot for debugging
            page.screenshot(path="error_screenshot.png")
            print("Screenshot saved to error_screenshot.png")

        finally:
            # Keep browser open for a moment to see result
            time.sleep(2)
            browser.close()


if __name__ == "__main__":
    print("Starting YC upvote automation...")
    print("Note: This will open a visible browser window")
    click_upvote_container()
    print("Done!")
