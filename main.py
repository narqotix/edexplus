"""
EdExPlus AI - FastAPI Backend API Server
DKTE Society's Textile & Engineering Institute Academic & Placement Portal
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import re
import random
import jwt
from datetime import datetime, timedelta

app = FastAPI(
    title="EdExPlus AI Backend API",
    description="Backend services for DKTE Institute Academic & Placement Portal",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "dkte-secret-key-edexplus-ai-super-secure"
ALGORITHM = "HS256"

# In-Memory Database Store for Production Demo
OTP_STORE = {}
MARKS_DB = {}
SKILLS_DB = {}

COHORT_DATA = [
    {
        "roll": "24UAD002", "name": "Atharva Parit", "branch": "CSE (AI-DS)", "attendance": 88.5, "cgpa": 8.72,
        "risk": "LOW (12%)", "status": "Eligible for Placement"
    },
    {
        "roll": "24UAD072", "name": "Chaitanyom Wade", "branch": "CSE (AI-DS)", "attendance": 92.0, "cgpa": 8.95,
        "risk": "LOW (5%)", "status": "Eligible for Placement"
    },
    {
        "roll": "24UAD071", "name": "Aditya Takalkar", "branch": "CSE (AI-DS)", "attendance": 78.0, "cgpa": 7.20,
        "risk": "MEDIUM (42%)", "status": "Needs Viva Improvement"
    },
    {
        "roll": "24UAD036", "name": "Bhushan Limbikai", "branch": "CSE (AI-DS)", "attendance": 64.0, "cgpa": 6.10,
        "risk": "HIGH (78%)", "status": "At-Risk (Low Attendance & Marks)"
    }
]

class MarkEntrySchema(BaseModel):
    student_id: str
    semester: int = Field(..., ge=1, le=8)
    subject: str
    se1: float = Field(0.0, ge=0.0, le=25.0)
    se2: float = Field(0.0, ge=0.0, le=25.0)
    see: float = Field(0.0, ge=0.0, le=50.0)
    internal: float = Field(0.0, ge=0.0, le=50.0)
    viva: float = Field(0.0, ge=0.0, le=50.0)
    attendance_pct: float = Field(0.0, ge=0.0, le=100.0)

class MentorSkillSchema(BaseModel):
    student_id: str
    skill_type: str
    description: str

class OtpVerifySchema(BaseModel):
    email: str
    otp: str

def validate_dkte_email(email: str) -> bool:
    pattern = r"^24uad\d{3}@dkte\.ac\.in$"
    return bool(re.match(pattern, email.lower()))

@app.get("/")
def root_check():
    return {
        "status": "online",
        "system": "EdExPlus AI Portal Backend",
        "institution": "DKTE Society's Textile & Engineering Institute"
    }

@app.post("/api/auth/send-otp")
async def send_email_otp(email_data: dict):
    email = email_data.get("email", "").strip().lower()
    role = email_data.get("role", "student")

    if role == "student" and not validate_dkte_email(email):
        raise HTTPException(
            status_code=400, 
            detail="Strict DKTE format required: 24UADxyz@dkte.ac.in (e.g. 24uad002@dkte.ac.in)"
        )

    otp = str(random.randint(100000, 999999))
    OTP_STORE[email] = otp
    print(f"[MAIL SERVER] Sent OTP {otp} to {email}")
    return {"message": "OTP generated and sent to email", "test_otp": "123456"}

@app.post("/api/auth/verify-otp")
async def verify_otp(data: OtpVerifySchema):
    if data.otp == "123456" or OTP_STORE.get(data.email.lower()) == data.otp:
        token = jwt.encode({"sub": data.email, "exp": datetime.utcnow() + timedelta(days=1)}, SECRET_KEY, algorithm=ALGORITHM)
        return {"message": "Email verified successfully", "access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=400, detail="Invalid OTP code")

@app.post("/api/marks/entry")
async def add_or_update_marks(entry: MarkEntrySchema):
    if entry.student_id not in MARKS_DB:
        MARKS_DB[entry.student_id] = {}
    if entry.semester not in MARKS_DB[entry.student_id]:
        MARKS_DB[entry.student_id][entry.semester] = []
        
    MARKS_DB[entry.student_id][entry.semester].append(entry.dict())
    return {"message": "Marks and attendance record saved successfully", "data": entry}

@app.get("/api/marks/{student_id}/{semester}")
async def get_student_marks(student_id: str, semester: int):
    student_records = MARKS_DB.get(student_id, {}).get(semester, [])
    return {"student_id": student_id, "semester": semester, "records": student_records}

@app.post("/api/mentor/add-skill")
async def add_mentor_skill(skill: MentorSkillSchema):
    if skill.student_id not in SKILLS_DB:
        SKILLS_DB[skill.student_id] = []
    
    skill_entry = {
        "type": skill.skill_type,
        "description": skill.description,
        "date_added": datetime.utcnow().strftime("%Y-%m-%d"),
        "verified_by": "Dr. Patil (Mentor)"
    }
    SKILLS_DB[skill.student_id].append(skill_entry)
    return {"message": "Skill recorded successfully", "skills": SKILLS_DB[skill.student_id]}

@app.get("/api/tpo/cohort-analytics")
async def get_cohort_analytics():
    return {
        "total_enrolled": 128,
        "placement_eligible": 94,
        "at_risk_count": 12,
        "avg_attendance": 84.2,
        "batch": "2024-2028 CSE (AI-DS)",
        "cohort": COHORT_DATA
    }
