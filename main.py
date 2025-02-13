from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import openai
import json
import subprocess
from pathlib import Path
from typing import Optional
import requests
import httpx
import shutil
import markdown
import csv
import duckdb
from PIL import Image

# Load OpenAI API Key
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
LLM_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
print(os.getenv("AIPROXY_TOKEN"))
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)
DATA_DIR = Path("/data")


TASK_TOOLS = [
    {
        # A1: Install and run datagen.py
        "type": "function",
        "function": {
            "name": "install_and_run_datagen",
            "description": "Install uv (if required) and run datagen.py with the provided email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The email of the user"}
                },
                "required": ["email"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A2: Format Markdown file
        "type": "function",
        "function": {
            "name": "format_markdown",
            "description": "Format a Markdown file using Prettier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Markdown file"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A3: Count specific weekdays in a file
        "type": "function",
        "function": {
            "name": "count_weekday",
            "description": "Count occurrences of a specific weekday in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "weekday": {"type": "string", "description": "Name of the weekday"}
                },
                "required": ["file_path", "weekday"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]
    # # A4: Sort contacts JSON file
    # {
    #     "name": "sort_contacts",
    #     "description": "Sort contacts.json by last and first name.",
    #     "parameters": {"input_file": {"type": "string"}, "output_file": {"type": "string"}}
    # },
    # # A5: Extract recent log entries
    # {
    #     "name": "extract_recent_logs",
    #     "description": "Extract first lines from the 10 most recent log files.",
    #     "parameters": {"log_dir": {"type": "string"}, "output_file": {"type": "string"}}
    # },
    # # A6: Generate Markdown index
    # {
    #     "name": "create_markdown_index",
    #     "description": "Generate an index of H1 titles from Markdown files.",
    #     "parameters": {"docs_dir": {"type": "string"}, "output_file": {"type": "string"}}
    # },
    # # A7: Extract sender email from email text
    # {
    #     "name": "extract_email_sender",
    #     "description": "Extract sender email from an email text file.",
    #     "parameters": {"email_file": {"type": "string"}, "output_file": {"type": "string"}}
    # },
    # # A8: Extract credit card number from an image
    # {
    #     "name": "extract_credit_card",
    #     "description": "Extract a credit card number from an image.",
    #     "parameters": {"image_path": {"type": "string"}, "output_file": {"type": "string"}}
    # },
    # # A9: Find most similar comments using embeddings
    # {
    #     "name": "find_similar_comments",
    #     "description": "Find the most similar comments using embeddings.",
    #     "parameters": {"input_file": {"type": "string"}, "output_file": {"type": "string"}}
    # },
    # # A10: Compute total ticket sales for Gold tickets
    # {
    #     "name": "compute_ticket_sales",
    #     "description": "Compute total sales for a specific ticket type in SQLite.",
    #     "parameters": {"db_path": {"type": "string"}, "ticket_type": {"type": "string"}, "output_file": {"type": "string"}},
    #     "required":["db_path","ticket_type","output_file"]
    # }



def call_llm(prompt: str) -> dict:
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    data = {"model": "gpt-4o-mini", 
           "messages": [{"role": "system", "content": "You are an automation agent. Return structured JSON for execution."}, {"role": "user", "content": prompt}],
             "tools": TASK_TOOLS,
            "tool_choice": "required",}
    print(data)
    response = httpx.post(LLM_API_URL, json=data, headers=headers,verify=False,timeout=100) 
    print("RAW LLM RESPONSE:", response.json())  # Debugging
    print(AIPROXY_TOKEN)
    func_name= response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    params_val= response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    func=response.json()["choices"][0]["message"]["tool_calls"][0]["function"]
    return func
    # print("params_val",type(params_val))
    # params=json.loads(params_val)
    # print("params",type(params))
    # print("Output: ",json.loads(params_val).get("email"))
    chosen_function=eval(func_name)
    chosen_function(**params)

    

@app.post("/run")
def run_task(task: str = Query(...)):
    """ Processes a task using OpenAI function calling. """
    response1 = call_llm("Install uv and run datagen.py with test@example.com")
    print(response1)
    function_call = call_llm(task)
    
    if not function_call:
        raise HTTPException(status_code=400, detail="Unable to process task.")
    
    function_name = function_call["name"]
    print("function_name",function_name)
    arguments = json.loads(function_call.get("arguments", "{}"))
    print("arguments",arguments)
    
    try:
        result = globals()[function_name](**arguments)
        return {"status": "success", "output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
def read_file(path: str = Query(...)):
    """ Reads and returns the content of a file. """
    file_path = DATA_DIR / path.lstrip("/")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return file_path.read_text()

# A1 Implementation
def install_and_run_datagen(email: str):

    print("Running A1:")
    print("email:",email)
    subprocess.run("pip install uv virtualenv", shell=True, check=True)
    subprocess.run(" virtualenv venv", shell=True, check=True)
    if platform.system() == "Windows":
        subprocess.run(r"D:\\LLM Agent 1\\venv\Scripts\\activate.bat", shell=True, check=True)
    else: 
        subprocess.run("source venv/bin/activate", shell=True, check=True)
    subprocess.run("uv pip install -r requirements.txt", shell=True, check=True)
    subprocess.run(f"uv run datagen.py {email}", shell=True, check=True)
    return "Data generation complete."

# A2 Implementation
def format_markdown(file_path: str):
    subprocess.run(f"npx prettier@3.4.2 --write {DATA_DIR / file_path}", shell=True, check=True)
    return "File formatted."

# A3 Implementation
def count_weekday(file_path: str, weekday: str):
    with open(DATA_DIR / file_path) as f:
        count = sum(1 for line in f if weekday.lower() in line.lower())
    output_file = DATA_DIR / f"{file_path}-count.txt"
    output_file.write_text(str(count))
    return f"{count} {weekday}s counted."

# A4 Implementation
def sort_contacts(input_file: str, output_file: str):
    with open(DATA_DIR / input_file) as f:
        contacts = json.load(f)
    contacts.sort(key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))
    with open(DATA_DIR / output_file, "w") as f:
        json.dump(contacts, f, indent=2)
    return "Contacts sorted."

# Additional implementations for A5-A10 and B1-B10 will follow...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
