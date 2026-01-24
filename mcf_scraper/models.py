from sqlalchemy import Column, String, Integer, Text, ARRAY, JSON, DateTime, Table, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from database import Base
import datetime

# Association table for Job <-> Skill
job_skills = Table(
    'job_skills', Base.metadata,
    Column('job_uuid', String, ForeignKey('jobs.uuid'), primary_key=True),
    Column('skill_uuid', String, ForeignKey('skills.uuid'), primary_key=True)
)

class Job(Base):
    __tablename__ = 'jobs'

    uuid = Column(String, primary_key=True, index=True)
    title = Column(String)
    company_name = Column(String)
    description = Column(Text)
    salary_min = Column(Integer, nullable=True)
    salary_max = Column(Integer, nullable=True)
    currency = Column(String, default="SGD")
    apply_url = Column(String)
    original_url = Column(String)
    metadata_json = Column(JSON) # Store raw extra data
    
    posted_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    skills = relationship("Skill", secondary=job_skills, back_populates="jobs")

class Skill(Base):
    __tablename__ = 'skills'
    
    uuid = Column(String, primary_key=True)
    name = Column(String)
    
    jobs = relationship("Job", secondary=job_skills, back_populates="skills")
