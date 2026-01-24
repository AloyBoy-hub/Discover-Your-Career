import csv
import argparse
import uuid
import datetime
from database import SessionLocal, init_db
from models import Job, Skill
from filters import is_buzzword
from skill_extractor import extract_skills_from_text

def import_csv(file_path):
    session = SessionLocal()
    count = 0
    
    # Initialize DB if needed (unlikely if running after main, but good practice)
    init_db()
    
    print(f"Reading {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read first line to normalize headers
        header_line = f.readline()
        fieldnames = [h.strip().lower() for h in header_line.split(',')]
        
        # Reset file pointer? DictReader needs clean start or we pass fieldnames
        # Better: use pandas if available? No, stick to stdlib.
        # Re-open or seek 0
        f.seek(0)
        reader = csv.DictReader(f, fieldnames=fieldnames)
        next(reader) # Skip original header
        
        # Validate headers
        required_headers = {'title', 'company'}
        if not required_headers.issubset(fieldnames):
            print(f"Error: CSV missing required headers. Found: {fieldnames}")
            print(f"Required: {required_headers}")
            return

        for row in reader:
            job_title = row.get('title')
            company = row.get('company')
            desc = row.get('description', '')
            skills_str = row.get('skills', '')
            
            # Generate a deterministic UUID based on title+company to avoid duplicates if re-run
            # Or just random if we don't care. Let's use random for external files as they might lack unique IDs.
            # actually, using uuid5 with a namespace is better for idempotency, but let's stick to random for simplicity OR unique per row content
            job_uuid = str(uuid.uuid4())
            
            # Create Job
            job = Job(
                uuid=job_uuid,
                title=job_title,
                company_name=company,
                description=desc,
                salary_min=int(row['salary_min']) if row.get('salary_min') else None,
                salary_max=int(row['salary_max']) if row.get('salary_max') else None,
                created_at=datetime.datetime.utcnow(),
                posted_date=datetime.datetime.utcnow(),
                apply_url="csv_import"
            )
            
            # Process Skills
            skill_objs = []
            final_skill_names = set()
            
            # 1. From CSV Column
            if skills_str:
                # Assume comma separated
                skill_names = [s.strip() for s in skills_str.split(',')]
                for s_name in skill_names:
                    if s_name:
                        final_skill_names.add(s_name)
            
            # 2. Extract from Description
            extracted = extract_skills_from_text(desc)
            for s_name in extracted:
                final_skill_names.add(s_name)
            
            # 3. Save to DB
            for s_name in final_skill_names:
                if is_buzzword(s_name):
                    continue
                    
                # Check if skill exists
                # We need a stable UUID for skills. 
                # Options: Search by name, or generate UUID from name.
                # MCF used random UUIDs from API. Here we only have names.
                # We will verify if a skill with ILIKE name exists, else create new.
                
                existing_skill = session.query(Skill).filter(Skill.name.ilike(s_name)).first()
                if existing_skill:
                    skill_objs.append(existing_skill)
                else:
                    new_skill_uuid = str(uuid.uuid4())
                    new_skill = Skill(uuid=new_skill_uuid, name=s_name)
                    session.add(new_skill)
                    session.flush() # get it ready
                    skill_objs.append(new_skill)
            
            job.skills = skill_objs
            session.add(job)
            count += 1
            
    session.commit()
    session.close()
    print(f"Successfully imported {count} jobs from {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Jobs from CSV")
    parser.add_argument("file", help="Path to CSV file")
    args = parser.parse_args()
    
    import_csv(args.file)
