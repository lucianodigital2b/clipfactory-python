import threading
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def create_job(self, job_data: Dict[str, Any]) -> str:
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "status": JobStatus.PENDING.value,
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "progress": 0,
                "progress_message": "Job created",
                "input_data": job_data,
                "result": None,
                "error": None
            }
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details by ID"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job_status(self, job_id: str, status: JobStatus, **kwargs):
        """Update job status and other fields"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["status"] = status.value
                
                if status == JobStatus.PROCESSING and "started_at" not in self.jobs[job_id]:
                    self.jobs[job_id]["started_at"] = datetime.now().isoformat()
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
                
                # Update any additional fields
                for key, value in kwargs.items():
                    self.jobs[job_id][key] = value
    
    def update_progress(self, job_id: str, progress: int, message: str = ""):
        """Update job progress"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["progress"] = progress
                self.jobs[job_id]["progress_message"] = message
    
    def set_job_result(self, job_id: str, result: Any):
        """Set job result and mark as completed"""
        self.update_job_status(job_id, JobStatus.COMPLETED, result=result)
    
    def set_job_error(self, job_id: str, error: str):
        """Set job error and mark as failed"""
        self.update_job_status(job_id, JobStatus.FAILED, error=error)
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove jobs older than specified hours"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self.lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                created_time = datetime.fromisoformat(job["created_at"]).timestamp()
                if created_time < cutoff_time:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
    
    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all jobs (for debugging)"""
        with self.lock:
            return self.jobs.copy()

# Global job manager instance
job_manager = JobManager()