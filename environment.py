import json
import os
from typing import Tuple, Dict, Any
from models import Action, Observation, Reward, JobSummary, FormField

class AutoApplicantEnv:
    def __init__(self):
        self.jobs_data = self._load_json("data/mock_jobs.json")
        self.profiles_data = self._load_json("data/mock_profiles.json")
        
        # Internal State tracking
        self.current_level = "level_1"
        self.profile = {}
        self.current_page = "home"
        self.system_message = "Environment initialized."
        self.visible_jobs = []
        self.job_description = None
        self.form_schema = None
        self.current_form_state = {}
        self.salary_db_result = None
        
        self.step_count = 0
        self.max_steps = 15
        self.final_score = 0.0
        self.final_reason = "Application not submitted."

    def _load_json(self, filepath: str) -> Any:
        # Failsafe loading
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as f:
            return json.load(f)

    def reset(self, level: str = "level_1") -> Observation:
        """Resets the environment for a specific difficulty level."""
        self.current_level = level
        profile_key = f"{level}_profile"
        self.profile = self.profiles_data.get(profile_key, self.profiles_data["level_1_profile"])
        
        self.current_page = "home"
        self.system_message = f"Welcome to Auto-Applicant. You are acting as {self.profile['name']}. Target Role: {self.profile['target_role']}."
        self.visible_jobs = None
        self.job_description = None
        self.form_schema = None
        self.final_score = 0.0
        self.final_reason = "Application not submitted."
        self.current_form_state = {}
        self.salary_db_result = None
        self.step_count = 0
        
        return self._get_observation()

    def state(self):
        """OpenEnv required interface method to return the current state."""
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """Processes the agent's action and updates the state."""
        self.step_count += 1
        reward_value = 0.0
        reason = "Step recorded."
        done = False
        
        # Guardrail: Prevent infinite loops
        if self.step_count >= self.max_steps:
            self.system_message = "Max steps reached. You timed out."
            return self._get_observation(), Reward(value=0.0, reason="Timeout"), True, {}

        # --- ACTION ROUTER ---
        if action.action_type == "search_jobs":
            # Return all jobs for simplicity, but format them as JobSummary models
            self.visible_jobs = [JobSummary(**job) for job in self.jobs_data]
            self.current_page = "search_results"
            self.system_message = "Search complete. Found listings."
            reward_value = 0.1 # Small partial reward for exploring
            reason = "Successfully searched jobs."

        elif action.action_type == "view_job":
            job = next((j for j in self.jobs_data if j["job_id"] == action.job_id), None)
            if job:
                self.current_page = "job_details"
                self.job_description = job["description"]
                self.form_schema = [FormField(**field) for field in job["form_schema"]]
                self.system_message = f"Viewing job: {job['title']}."
            else:
                self.system_message = "Error: Job ID not found."
                reward_value = -0.1

        elif action.action_type == "fill_field":
            if self.current_page != "job_details":
                self.system_message = "Error: You must view a job before filling its form."
                reward_value = -0.1
            elif action.field_name and action.field_value:
                self.current_form_state[action.field_name] = action.field_value
                self.system_message = f"Field '{action.field_name}' updated."
            else:
                self.system_message = "Error: Missing field_name or field_value."

        elif action.action_type == "upload_file":
            if action.file_name not in self.profile.get("available_files", []):
                self.system_message = f"Error: File '{action.file_name}' is not in available_files."
                reward_value = -0.1
            else:
                self.current_form_state["resume"] = action.file_name
                self.system_message = f"File '{action.file_name}' attached to form."

        elif action.action_type == "query_salary_db":
            # The Level 3 Obstacle
            self.salary_db_result = "Median salary for Machine Learning Engineer in San Francisco is 145000."
            self.system_message = "Salary database queried successfully."
            reward_value = 0.2
            reason = "Used external tool to overcome data gap."

        elif action.action_type == "submit_application":
            done = True
            reward_value, reason = self._grade(action.job_id)
            self.current_page = "success" if reward_value > 0.5 else "error"
            self.system_message = reason
            self.final_score = reward_value
            self.final_reason = reason

        return self._get_observation(), Reward(value=reward_value, reason=reason), done, {}

    def _grade(self, submitted_job_id: str) -> Tuple[float, str]:
        """The Deterministic Grader. Strictly evaluates the final form state."""
        # 1. Did they apply to the right job?
        target_jobs = {"level_1": "job_001", "level_2": "job_002", "level_3": "job_003"}
        correct_job_id = target_jobs.get(self.current_level)
        
        if submitted_job_id != correct_job_id:
            return 0.0, f"Failed. Applied to {submitted_job_id} instead of {correct_job_id}."

        # 2. Check basic requirements
        form = self.current_form_state
        if "full_name" not in form or "email" not in form or "resume" not in form:
            return 0.2, "Applied to correct job, but missing basic required fields."

        # 3. Level-Specific Logic Checks
        if self.current_level == "level_2":
            if form.get("resume") != "backend_resume.pdf":
                return 0.5, "Failed Level 2: Uploaded wrong resume variant."
            if form.get("requires_visa") == "Yes" and "visa_type" not in form:
                return 0.5, "Failed Level 2: Missed conditional visa field."
            return 1.0, "Success! Level 2 passed perfectly."

        if self.current_level == "level_3":
            # Math check: dates in profile were 2025-01-01 to 2026-03-01 (14 months)
            if str(form.get("months_python_experience")) != "14":
                return 0.4, "Failed Level 3: Incorrectly calculated months of experience."
            
            # Tool usage check: did they find the salary?
            if str(form.get("expected_salary")) != "145000":
                return 0.6, "Failed Level 3: Incorrect or missing expected salary."
                
            # Synthesis check: did they write a project summary?
            if len(form.get("project_summary", "")) < 10:
                return 0.8, "Failed Level 3: Project summary too short or missing."

            return 1.0, "Success! Level 3 frontier challenge completed perfectly."

        # Level 1 success
        return 1.0, "Success! Level 1 basic application completed."

    def state(self) -> Observation:
        """Returns the current state without state mutation."""
        return self._get_observation()

    def _get_observation(self) -> Observation:
        """Packages the internal state into the Pydantic Observation model."""
        # Hide the profile details so the agent has to figure it out
        safe_profile = self.profile.copy()
        safe_profile.pop("name", None) # Let the agent use the environment intro

        return Observation(
            current_page=self.current_page,
            system_message=self.system_message,
            available_files=self.profile.get("available_files", []),
            student_profile=safe_profile,
            visible_jobs=self.visible_jobs,
            job_description=self.job_description,
            form_schema=self.form_schema,
            current_form_state=self.current_form_state,
            salary_db_result=self.salary_db_result
        )