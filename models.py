from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

# ==========================================
# 1. ACTION SPACE (What the agent can do)
# ==========================================

class Action(BaseModel):
    """
    The strictly typed action space for the Auto-Applicant environment.
    The agent must choose one action_type and provide the relevant optional fields.
    """
    action_type: Literal[
        "search_jobs", 
        "view_job", 
        "fill_field", 
        "upload_file", 
        "query_salary_db",
        "submit_application"
    ] = Field(..., description="The type of action to execute in the environment.")
    
    # Payload fields (Optional depending on the action taken)
    search_query: Optional[str] = Field(None, description="Keywords to search. Use with 'search_jobs'.")
    job_id: Optional[str] = Field(None, description="The ID of the job. Use with 'view_job' or 'submit_application'.")
    field_name: Optional[str] = Field(None, description="The exact name of the form field. Use with 'fill_field'.")
    field_value: Optional[str] = Field(None, description="The text/value to input. Use with 'fill_field'.")
    file_name: Optional[str] = Field(None, description="The name of the file from available_files. Use with 'upload_file'.")
    location: Optional[str] = Field(None, description="City to query the salary DB for (e.g., 'San Francisco'). Use with 'query_salary_db'.")


# ==========================================
# 2. OBSERVATION SPACE (What the agent sees)
# ==========================================

# Helper Models for the Observation
class JobSummary(BaseModel):
    job_id: str
    title: str
    company: str

class FormField(BaseModel):
    name: str
    field_type: Literal["text", "number", "dropdown", "file_upload", "textarea"]
    required: bool
    options: Optional[List[str]] = None  # Populated if field_type is 'dropdown'

class Observation(BaseModel):
    """
    The state of the environment returned to the agent after every step.
    """
    current_page: Literal["home", "search_results", "job_details", "application_form", "success", "error"] = Field(..., description="Where the agent currently is.")
    system_message: str = Field(..., description="Critical feedback from the last action (e.g., 'Success', 'Error: Invalid ID').")
    
    # The agent's local context (always visible)
    available_files: List[str] = Field(..., description="Files the agent has access to on their local machine.")
    student_profile: Optional[Dict[str, Any]] = Field(None, description="The student's background data (portfolio, dates, etc.) used to fill forms.")
    
    # Dynamic context (visible depending on the page)
    visible_jobs: Optional[List[JobSummary]] = Field(None, description="List of jobs when on 'search_results'.")
    job_description: Optional[str] = Field(None, description="Full text description when on 'job_details'.")
    form_schema: Optional[List[FormField]] = Field(None, description="The fields required for the current application form.")
    current_form_state: Optional[Dict[str, Any]] = Field(None, description="What the agent has successfully filled out so far.")
    salary_db_result: Optional[str] = Field(None, description="The response from the market salary database tool.")


# ==========================================
# 3. REWARD SPACE
# ==========================================

class Reward(BaseModel):
    """
    OpenEnv requires typed rewards. We use this to return both the numerical 
    score and a string explanation for transparency during evaluation.
    """
    value: float = Field(..., ge=-1.0, le=1.0, description="The numerical reward for the step.")
    reason: str = Field(..., description="Why this reward was given (helps debug the agent).")