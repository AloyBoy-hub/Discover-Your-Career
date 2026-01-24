import argparse
import sys
from database import init_db, SessionLocal, engine
from models import Job, Skill
from client import MCFClient
from filters import is_buzzword
from sqlalchemy.dialects.postgresql import insert
import datetime

def save_job(session, job_data):
    # Process Skills
    skills_list = job_data.get('skills', [])
    skill_objs = []
    
    for s in skills_list:
        skill_uuid = s.get('uuid')
        skill_name = s.get('skill')
        
        if is_buzzword(skill_name):
            continue
        
        # Upsert Skill
        # Try to get existing skill
        existing = session.query(Skill).filter(Skill.uuid == skill_uuid).first()
        if not existing:
            existing = Skill(uuid=skill_uuid, name=skill_name)
            session.add(existing)
            # Flush to ensure it's in the session map
            session.flush()
        skill_objs.append(existing)

    # Process Job
    job_uuid = job_data.get('uuid')
    meta = job_data.get('metadata', {})
    address_info = job_data.get('address', {})
    salary_info = job_data.get('salary', {})
    posted_company = job_data.get('postedCompany', {})
    
    # Check existing job
    existing_job = session.query(Job).filter(Job.uuid == job_uuid).first()
    if existing_job:
        print(f"Job {job_uuid} already exists. Updating.")
        # If updating, we can update fields here
        job_obj = existing_job
    else:
        job_obj = Job(uuid=job_uuid)
    
    job_obj.title = job_data.get('title')
    job_obj.company_name = posted_company.get('name')
    job_obj.description = job_data.get('description', '') # Might be empty in list view
    job_obj.salary_min = salary_info.get('minimum')
    job_obj.salary_max = salary_info.get('maximum')
    job_obj.apply_url = meta.get('jobDetailsUrl')
    job_obj.original_url = meta.get('jobDetailsUrl')
    job_obj.posted_date = datetime.datetime.fromisoformat(meta.get('newPostingDate')) if meta.get('newPostingDate') else None
    
    # Update skills
    job_obj.skills = skill_objs
    
    session.add(job_obj)
    session.commit()
    return job_obj

def run_scraper(query, limit_pages=1):
    print(f"Initializing database...")
    try:
        init_db()
    except Exception as e:
        print(f"Database error: {e}")
        print("Please ensure your DATABASE_URL in .env is correct and Postgres is running.")
        return

    client = MCFClient()
    session = SessionLocal()
    
    total_new = 0
    total_updated = 0
    
    for page in range(limit_pages):
        print(f"Fetching page {page} for query '{query}'...")
        result = client.search_jobs(query, page=page)
        
        if not result:
            break
            
        jobs = result.get('results', [])
        print(f"Found {len(jobs)} jobs.")
        
        if not jobs:
            print("No more jobs found.")
            break
            
        for j in jobs:
            # We might need to fetch full details if description is missing or incomplete
            # The search result doesn't usually have the full raw HTML description
            # Let's check a job detail
            job_uuid = j.get('uuid')
            
            # Fetch full details
            # Optimisation: check if we already have it with description
            # FOR NOW, let's just fetch it to get the description
            full_job = client.get_job_details(job_uuid)
            if full_job:
                save_job(session, full_job)
                sys.stdout.write(".")
                sys.stdout.flush()
            else:
                # Fallback to search result data
                save_job(session, j)
                sys.stdout.write("x")
        
        print("\nPage finished.")
    
    session.close()
    print("Scraping completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyCareersFuture Scraper")
    parser.add_argument("--query", type=str, required=True, help="Job search query (e.g. 'Data Analyst')")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to scrape")
    
    args = parser.parse_args()
    
    run_scraper(args.query, args.pages)
