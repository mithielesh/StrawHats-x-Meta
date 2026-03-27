import os
import os
import json
import requests
from openai import OpenAI
from dotenv import dotenv_values # <-- We use dotenv_values instead

# 1. Read the file directly, ignoring Windows environment variables completely
config = dotenv_values(".env") 
api_key = config.get("OPENAI_API_KEY")

# 2. Safety check: Did it actually find the file?
if not api_key:
    print("ERROR: Python cannot find your key. Your file might be named '.env.txt' instead of '.env'.")
    exit(1)

print(f"DEBUG: Successfully pulled key from file: {api_key[:6]}...")

# 3. Initialize the client using the direct file string
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=api_key 
)

BASE_URL = "https://liquozous-strawhat-x-meta.hf.space"

def run_agent(level: str):
    print(f"\n{'='*50}\nStarting Agent for {level.upper()}\n{'='*50}")

    # 1. Reset Environment for the specific level
    res = requests.post(f"{BASE_URL}/reset", params={"level": level})
    obs = res.json()

    # The System Prompt explaining the rules to the LLM
    messages = [
        {
            "role": "system", 
            "content": "You are an autonomous AI agent whose goal is to successfully apply for a job. You will be provided with the current environment observation. You must respond ONLY with a valid JSON object representing your next action. Do NOT wrap your response in markdown code blocks. Start directly with {."
        }
    ]

    for step in range(15): # Max 15 steps per episode
        print(f"\n--- Step {step + 1} ---")

        # Get action schema so the LLM knows what it is allowed to do
        schema_res = requests.get(f"{BASE_URL}/tasks").json()
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

        # Ask the Meta Llama model via Hugging Face for the next move
        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct:fastest", 
                messages=messages
            )
        except Exception as e:
            print(f"API Error: {e}")
            break

        action_json_str = response.choices[0].message.content
        print(f"Raw Agent decision: {action_json_str}")

        # Robustness: Clean up markdown if the LLM ignores instructions
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
        step_res = requests.post(f"{BASE_URL}/step", json=action_dict)
        step_data = step_res.json()

        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]

        # Add the result to the history so the LLM remembers what just happened
        messages.append({"role": "assistant", "content": cleaned_json_str})
        messages.append({"role": "user", "content": f"Action result - Reward: {reward['value']}, System Message: {obs['system_message']}"})

        if done:
            print("\nAgent finished the episode!")
            break

    # Get final Grader Score from your engine
    grader_res = requests.get(f"{BASE_URL}/grader")
    grader_data = grader_res.json()
    print(f"\n>>> FINAL SCORE for {level}: {grader_data['score']} ({grader_data['reason']}) <<<")

if __name__ == "__main__":
    # The hackathon requires testing at least 3 tasks
    tasks = ["level_1", "level_2", "level_3"]
    for task in tasks:
        run_agent(task)