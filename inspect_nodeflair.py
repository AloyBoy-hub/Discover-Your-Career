import asyncio
from playwright.async_api import async_playwright
import json

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Monitor network requests
        async def handle_response(response):
            # Look for JSON responses that might contain job data
            if "api" in response.url or "jobs" in response.url or "search" in response.url:
                try:
                    if response.request.method == "GET" and response.status == 200:
                        content_type = response.headers.get("content-type", "")
                        if "json" in content_type:
                            print(f"API Candidate: {response.url}")
                            # Inspect generic body keys
                            body = await response.json()
                            if isinstance(body, dict):
                                keys = list(body.keys())
                                print(f"Keys: {keys}")
                                if "jobs" in keys or "job_listings" in keys or "data" in keys:
                                    print("POTENTIAL HIT!")
                except Exception:
                    pass

        page.on("response", handle_response)

        print("Navigating to NodeFlair...")
        await page.goto("https://www.nodeflair.com/jobs?query=software+engineer", wait_until="networkidle")
        
        await asyncio.sleep(8)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
