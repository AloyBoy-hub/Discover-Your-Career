import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating to NodeFlair...")
        await page.goto("https://www.nodeflair.com/jobs?query=software+engineer", wait_until="networkidle")
        
        # Get content
        content = await page.content()
        with open("nodeflair.html", "w") as f:
            f.write(content)
        print("Dumped HTML to nodeflair.html")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
