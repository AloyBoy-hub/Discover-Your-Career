import asyncio
from playwright.async_api import async_playwright
import json

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Monitor network requests
        async def handle_response(response):
            if "api.mycareersfuture.gov.sg" in response.url:
                print(f"API Response: {response.url} [{response.status}]")
                try:
                    # distinct checking for search-like endpoints
                    if "search" in response.url or "jobs" in response.url:
                        if response.request.method == "GET" or response.request.method == "POST":
                            print(f"Checking content for: {response.url}")
                            # Inspect a bit of the body to see if it has job listings
                            body = await response.json()
                            if "results" in body or "jobs" in body:
                                print(f"FOUND JOB LISTING API: {response.url}")
                                print(f"Keys in response: {list(body.keys())}")
                except Exception as e:
                    pass # ignore non-json or errors

        page.on("response", handle_response)

        print("Navigating to MyCareersFuture...")
        await page.goto("https://www.mycareersfuture.gov.sg/search?search=data&sortBy=new_posting_date&page=0", wait_until="networkidle")
        
        await asyncio.sleep(5)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
