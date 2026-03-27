# Auto-Applicant: OpenEnv Job Board Simulation

## Description
Auto-Applicant is an OpenEnv-compliant simulation of a real-world job board and application portal. It tests an AI agent's ability to navigate complex, multi-page web forms, synthesize portfolio data into cover letters, utilize external tools (like a salary database), and adhere to strict formatting rules.

## Observation & Action Spaces
* **Action Space:** `search_jobs`, `view_job`, `fill_field`, `upload_file`, `query_salary_db`, `submit_application`.
* **Observation Space:** Current page state, visible job listings, dynamic form schemas, system feedback, and the agent's available local files/profile context.

## Tasks (Difficulty Progression)
1. **Level 1 (Easy):** Basic search and apply. Agent must find the correct job and fill out static text fields.
2. **Level 2 (Medium):** Conditional logic. Agent must select the correct resume variant based on the job description and handle conditional form fields (e.g., Visa requirements).
3. **Level 3 (Hard):** Synthesis & Tool Use. The agent must calculate exact dates of experience from a raw JSON portfolio, query an external salary database tool to bypass a missing information obstacle, and generate a synthesized cover letter paragraph.

## Baseline Scores
Tested using `Qwen2.5-72B-Instruct` via Hugging Face Router:
* Level 1: 1.0
* Level 2: 1.0
* Level 3: 1.0