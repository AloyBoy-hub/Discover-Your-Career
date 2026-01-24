import requests
import json
import time

class MCFClient:
    BASE_URL = "https://api.mycareersfuture.gov.sg/v2"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/json"
        })

    def search_jobs(self, query: str, limit: int = 20, page: int = 0):
        url = f"{self.BASE_URL}/search"
        params = {
            "limit": limit,
            "page": page
        }
        payload = {
            "search": query,
            # We can add more filters here if needed
            "sortBy": ["new_posting_date"]
        }
        
        try:
            response = self.session.post(url, params=params, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching jobs: {e}")
            return None

    def get_job_details(self, job_uuid: str):
        url = f"{self.BASE_URL}/jobs/{job_uuid}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching job details for {job_uuid}: {e}")
            return None
