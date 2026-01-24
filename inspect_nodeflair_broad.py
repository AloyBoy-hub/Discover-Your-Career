import asyncio
from playwright.async_api import async_playwright
import json

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Monitor ALL network responses
        async def handle_response(response):
            try:
                ct = response.headers.get("content-type", "")
                if "json" in ct:
                    print(f"JSON Response: {response.url}")
                    # Peek at body to see if it looks like job data
                    if "nodeflair" in response.url:
                        body = await response.json()
                        keys = list(body.keys())
                        print(f"  Keys: {keys}")
            except Exception:
                pass

        page.on("response", handle_response)

        print("Navigating to Glints...")
        await page.goto("https://glints.com/sg/opportunities/jobs/explore?keyword=software+engineer", wait_until="networkidle")
        
        await asyncio.sleep(5)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
