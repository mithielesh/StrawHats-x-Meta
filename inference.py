import os
import json
import requests
from openai import OpenAI

# ==========================================
# HACKATHON REQUIRED ENVIRONMENT VARIABLES
# ==========================================
# The grader will inject these. We use fallback values just in case you test locally.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:11434/v1") 
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen2.5-coder:1.5b")
HF_TOKEN = os.environ.get("HF_TOKEN", "ollama")

# 1. Initialize OpenAI Client strictly using the required variables
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN 
)

# 2. Your Live OpenEnv Space URL
SPACE_URL = "https://liquozous-strawhat-x-meta.hf.space"

def run_agent(level: str):
    print(f"\n{'='*50}\nStarting Agent for {level.upper()}\n{'='*50}")

    # Reset Environment for the specific level
    res = requests.post(f"{SPACE_URL}/reset", params={"level": level})
    obs = res.json()

    messages = [
        {
            "role": "system", 
            "content": "You are an autonomous AI agent whose goal is to successfully apply for a job. You will be provided with the current environment observation. You must respond ONLY with a valid JSON object representing your next action. Do NOT wrap your response in markdown code blocks. Start directly with {."
        }
    ]

    for step in range(15): # Max 15 steps per episode
        print(f"\n--- Step {step + 1} ---")

        schema_res = requests.get(f"{SPACE_URL}/tasks").json()
        action_schema = schema_res["action_schema"]

        prompt = f"""
        Current Observation:
        {json.dumps(obs, indent=2)}

        Available Action Schema:
        {json.dumps(action_schema, indent=2)}

        Analyze the observation. Choose the best next action to progress the job application. 
        Respond strictly with a JSON object that matches the Action Schema.
        """

        messages.append({"role": "user", "content": prompt})

        try:
            # Strictly using the required MODEL_NAME variable
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages
            )
        except Exception as e:
            print(f"API Error: {e}")
            break

        action_json_str = response.choices[0].message.content
        print(f"Raw Agent decision: {action_json_str}")

        cleaned_json_str = action_json_str.strip()
        if cleaned_json_str.startswith("```json"):
            cleaned_json_str = cleaned_json_str[7:]
        elif cleaned_json_str.startswith("```"):
            cleaned_json_str = cleaned_json_str[3:]
        
        if cleaned_json_str.endswith("```"):
            cleaned_json_str = cleaned_json_str[:-3]
            
        cleaned_json_str = cleaned_json_str.strip()

        try:
            action_dict = json.loads(cleaned_json_str)
        except json.JSONDecodeError:
            print("Agent returned invalid JSON. Aborting.")
            break

        # Send the LLM's action to your FastAPI server
        step_res = requests.post(f"{SPACE_URL}/step", json=action_dict)
        step_data = step_res.json()

        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]

        messages.append({"role": "assistant", "content": cleaned_json_str})
        messages.append({"role": "user", "content": f"Action result - Reward: {reward['value']}, System Message: {obs['system_message']}"})

        if done:
            print("\nAgent finished the episode!")
            break

    grader_res = requests.get(f"{SPACE_URL}/grader")
    grader_data = grader_res.json()
    print(f"\n>>> FINAL SCORE for {level}: {grader_data['score']} ({grader_data['reason']}) <<<")

if __name__ == "__main__":
    tasks = ["level_1", "level_2", "level_3"]
    for task in tasks:
        run_agent(task)