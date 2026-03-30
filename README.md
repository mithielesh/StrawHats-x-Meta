---
title: StrawHat-X-Meta AutoApplicant
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags: [openenv]
---

# AutoApplicantEnv (OpenEnv Simulation)

## Environment Description
An autonomous job-application agent environment. The environment simulates a real-world job portal where an agent must search for jobs, navigate conditional form logic (such as visa requirements), query an external tool (Salary Database), and successfully submit applications.

## Action and Observation Spaces
* **Observation Space:** A structured JSON object containing the current `screen`, available `jobs`, active `form_fields`, `tool_results`, and `system_message`.
* **Action Space:** A JSON object defining the `action_type` (e.g., `search_jobs`, `view_job`, `fill_field`, `upload_file`, `query_salary_db`, `submit_application`) and its required parameters based on the dynamically provided schema.

## Tasks
* **level_1 (Easy):** Basic job searching, standard form filling, and resume upload.
* **level_2 (Medium):** Introduces conditional logic handling (checking Visa requirements based on role).
* **level_3 (Hard):** Requires external tool usage (querying a Salary DB by location) and mathematical reasoning (converting years to months of experience).

