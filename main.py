from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from models import Action, Observation, Reward
from environment import AutoApplicantEnv

app = FastAPI(title="Auto-Applicant OpenEnv", version="1.0.0")

# Instantiate our environment engine globally
env = AutoApplicantEnv()

@app.get("/")
def home():
    return {"status": "OpenEnv Server is Live", "author": "Liquozous"}

@app.post("/reset", response_model=Observation)
async def reset_environment(level: str = "level_1"):
    """Resets the environment to the beginning of a task."""
    return env.reset(level=level)

@app.post("/step")
async def take_step(action: Action):
    """Takes an action from the agent and returns the result."""
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=Observation)
async def get_state():
    """Returns the current state without taking an action."""
    return env.state()

@app.get("/tasks")
async def get_tasks():
    """Returns the available tasks and the action schema."""
    return {
        "tasks": ["level_1", "level_2", "level_3"],
        "action_schema": Action.model_json_schema()
    }

@app.get("/grader")
async def get_grader():
    """Returns the perfectly tracked grader score."""
    return {
        "score": env.final_score,
        "reason": env.final_reason
    }

@app.post("/baseline")
async def run_baseline():
    return {"message": "Baseline script execution pending!"}