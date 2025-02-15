from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
import os
from typing import List
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
import platform
import re
import base64
#import easyocr
import pytesseract

# Load OpenAI API Key
AIPROXY_TOKEN = os.getenv("")
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
if platform.system() == "Windows":
    DATA_DIR = Path("D:/data")  # Use Windows-compatible path
else:
    DATA_DIR = Path("/data")
TASK_TOOLS = [
    {
        # A1: Install and run datagen.py
        "type": "function",
        "function": {
            "name": "install_and_run_datagen",
            "description": "Install a package and run the script with the provided arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "array",
                    "items":{"type":"string"
                    },
                    "description": "List of argument from user user"},
                    "script":{"type":"string","description":"The script that is to be run"}
                    
                },
                "required": ["args","script"],
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
                    "input_file_path": {"type": "string", "description": "Path to the input file"},
                    "weekday": {"type": "string", "description": "Name of the weekday"},
                    "output_file_path": {"type": "string", "description": "Path to the output file"},

                },
                "required": ["input_file_path", "weekday","output_file_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A4: Sort contacts JSON file
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "description": "Sort contacts.json by last and first name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the input JSON file containing contacts"},
                    "output_file": {"type": "string", "description": "Path to the output JSON file where sorted contacts will be saved"}
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A5: Extract recent log entries
        "type": "function",
        "function": {
            "name": "extract_recent_logs",
            "description": "Extract first lines from the 10 most recent log files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_dir": {"type": "string", "description": "Path to the directory containing log files"},
                    "output_file": {"type": "string", "description": "Path to the output file where extracted log entries will be saved"}
                },
                "required": ["log_dir", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A6: Generate Markdown index
        "type": "function",
        "function": {
            "name": "create_markdown_index",
            "description": "Generate an index of H1 titles from Markdown files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs_dir": {"type": "string", "description": "Path to the directory containing Markdown files"},
                    "output_file": {"type": "string", "description": "Path to the output file where the index will be saved"}
                },
                "required": ["docs_dir", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A7: Extract sender email from email text
        "type": "function",
        "function": {
            "name": "extract_email_sender",
            "description": "Extract sender email from an email text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_file": {"type": "string", "description": "Path to the email text file"},
                    "output_file": {"type": "string", "description": "Path to the output file where extracted email will be saved"}
                },
                "required": ["email_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A8: Extract credit card number from an image
        "type": "function",
        "function": {
            "name": "extract_credit_card",
            "description": "Extract a credit card number from an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image containing a credit card"},
                    "output_file": {"type": "string", "description": "Path to the output file where extracted credit card number will be saved"}
                },
                "required": ["image_path", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A9: Find most similar comments using embeddings
        "type": "function",
        "function": {
            "name": "find_similar_comments",
            "description": "Find the most similar comments using embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string", "description": "Path to the input file containing comments"},
                    "output_file": {"type": "string", "description": "Path to the output file where similar comments will be saved"}
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        # A10: Compute total ticket sales for Gold tickets
        "type": "function",
        "function": {
            "name": "compute_ticket_sales",
            "description": "Compute total sales for a specific ticket type in SQLite.",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {"type": "string", "description": "Path to the SQLite database"},
                    "ticket_type": {"type": "string", "description": "Type of ticket to compute sales for"},
                    "output_file": {"type": "string", "description": "Path to the output file where sales data will be saved"}
                },
                "required": ["db_path", "ticket_type", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


def call_llm_for_sender_email(email_text: str) -> str:
    """Calls the LLM API to extract the sender's email from the given email text."""

    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the sender's email from the provided email text. Return only the email address."},
            {"role": "user", "content": email_text}
        ]
    }
    response=httpx.post(LLM_API_URL, json=data, headers=headers,verify=False,timeout=20)
    print("RAW MAIL RESPONSE:", response.json()) 
    from_email=response.json()["choices"][0]["message"]["content"].strip()
    print(from_email)
    return from_email

def call_llm_for_credit_card_extraction(base64_image: str) -> str:
    """Calls ChatGPT API to extract text from an image (Base64 format)."""

    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini-2024-07-18",  # Use GPT-4 with vision capabilities
        "messages": [
            {"role": "system", "content": "Extract the credit card number from the image provided."},
            {"role": "user", "content": f"Here is an image encoded in Base64: {base64_image}"}
        ]
    }

    response=httpx.post(LLM_API_URL, json=data, headers=headers,verify=False,timeout=20)
    print("RAW Image RESPONSE:", response.json()) 
    
    #extracted_text = response["choices"][0]["message"]["content"].strip()
    #return extracted_text

def call_llm(prompt: str) -> dict:
    headers = {"Content-type":"application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",}
    data = {"model": "gpt-4o-mini", 
           "messages": [{"role": "system", "content": """
                         You are a multilingual automation agent.
                         If your task involve running a script then use run_task
                         Return structured JSON for execution.If the """}, {"role": "user", "content": prompt}],
             "tools": TASK_TOOLS,
            "tool_choice": "required",}
    
    response = httpx.post(LLM_API_URL, json=data, headers=headers,verify=False,timeout=20) 
    print("RAW LLM RESPONSE:", response.json())  # Debugging
    
    
    func=response.json()["choices"][0]["message"]["tool_calls"][0]["function"]
    return func
    # print("params_val",type(params_val))
    # params=json.loads(params_val)
    # print("params",type(params))
    # print("Output: ",json.loads(params_val).get("email"))
    

    

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
    chosen_function=eval(function_name)
    res=chosen_function(**arguments)
    try:
        # result = globals()[function_name](**arguments)
        # print(res)
        return {"status": "success", "output": res,"status code":Response.status_code}
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))
        return {"status": "Failure", "output": res,"status code":status_code}

@app.get("/read")
def read_file(path: str = Query(...)):
    """ Reads and returns the content of a file. """
    file_path = DATA_DIR / path.lstrip("/")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return file_path.read_text()

# A1 Implementation
def install_and_run_datagen(args: List[str],script: str,):

    print("Running A1:")
    
    email=args[0]
    print("email:",email)
    print("script", script)
    command = ["uv", "run", script,email] 
    subprocess.run(command, check=True)
    Response.status_code=200
    return "Data generation complete."

# A2 Implementation
def format_markdown(file_path: str):
    subprocess.run(f"npx prettier@3.4.2 --write {DATA_DIR / file_path}", shell=True, check=True)
    Response.status_code=200
    return "File formatted."

# A3 Implementation
def count_weekday(input_file_path: str, weekday: str, output_file_path: str):
    """Counts occurrences of a specific weekday in an input file and writes the count to an output file."""
    
    input_file = DATA_DIR / input_file_path.lstrip("/")
    output_file = DATA_DIR / output_file_path.lstrip("/")

    # Check if the input file exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file '{input_file_path}' not found!")

    # Read file and count occurrences
    with open(input_file, "r", encoding="utf-8") as f:
        count = sum(1 for line in f if weekday.lower() in line.lower())

    # Write result to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(count))
    Response.status_code=200
    return f"{count} occurrences of {weekday} counted and written to {output_file_path}"


# A4 Implementation
def sort_contacts(input_file: str, output_file: str):
    with open(DATA_DIR / input_file) as f:
        contacts = json.load(f)
    contacts.sort(key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))
    with open(DATA_DIR / output_file, "w") as f:
        json.dump(contacts, f, indent=2)
    Response.status_code=200
    return "Contacts sorted."

 # A5 Implementation   
def extract_recent_logs(log_dir: str, output_file: str):
    """Extracts the first line from the 10 most recent .log files in a directory and writes them to an output file."""
    
    log_dir_path = Path(DATA_DIR) / log_dir.lstrip("\\/")  # Remove leading slashes
    output_file_path = Path(DATA_DIR) / output_file.lstrip("\\/")  # Remove leading slashes

    # Ensure the parent directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_dir_path.exists() or not log_dir_path.is_dir():
        raise FileNotFoundError(f"Log directory '{log_dir_path}' not found!")

    # Get all .log files sorted by last modified time (most recent first)
    log_files = sorted(
        log_dir_path.glob("*.log"),
        key=lambda f: f.stat().st_mtime,
        reverse=True  # Most recent first
    )[:10]  # Get the 10 most recent log files

    extracted_lines = []
    for log_file in log_files:
        try:
            with log_file.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                extracted_lines.append(f"{log_file.name}: {first_line}")
        except Exception as e:
            extracted_lines.append(f"{log_file.name}: [Error reading file: {str(e)}]")

    # Write extracted lines to the output file
    output_file_path.write_text("\n".join(extracted_lines), encoding="utf-8")
    Response.status_code=200
    return f"Extracted first lines from {len(log_files)} log files to {output_file_path}"


# A6 Implementation 



def create_markdown_index(docs_dir: str, output_file: str):
    """Generates an index of H1 titles (# headings) from Markdown files in a directory."""
    
    # Standardize paths for Windows & Linux
    docs_dir_path = Path(DATA_DIR) / Path(os.path.normpath(docs_dir).lstrip("\\/"))
    output_file_path = Path(DATA_DIR) / Path(os.path.normpath(output_file).lstrip("\\/"))

    # Ensure the directory exists (Create if missing)
    if not docs_dir_path.exists():
        docs_dir_path.mkdir(parents=True, exist_ok=True)

    if not docs_dir_path.is_dir():
        raise FileNotFoundError(f"Markdown directory '{docs_dir_path}' not found!")

    index = {}

    md_files = list(docs_dir_path.rglob("*.md"))
    print(md_files)

    if not md_files:
        raise FileNotFoundError(f"No Markdown files found in '{docs_dir_path}'!")

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):  # Extract first H1 title
                    relative_filename = os.path.relpath(md_file, docs_dir_path)  # Get relative path
                    index[relative_filename] = line.strip("# ").strip()  # Remove `#` and trim spaces
                    print(index)
                    break  # Stop after first H1 title

    if not index:
        print("No H1 titles found! Creating an empty JSON file.")
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=4)
        return "No H1 titles found in any Markdown files! Empty index created."

    # Write extracted titles to the output JSON file
    output_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)
        print(json.dump(index, f, indent=4))

    Response.status_code = 200
    return f"Markdown index created with {len(index)} entries in {output_file_path}"


# A7 Implementation 
def extract_email_sender(email_file: str, output_file: str):
    """Extracts the sender's email address from an email text file."""
    
    email_file_path = Path(DATA_DIR) / Path(os.path.normpath(email_file).lstrip("\\/"))
    output_file_path = Path(DATA_DIR) / Path(os.path.normpath(output_file).lstrip("\\/"))
    
    if not email_file_path.exists():
        Response.status_code = 400
        raise FileNotFoundError(f"Email file '{email_file}' not found!")

    with open(email_file_path, "r", encoding="utf-8") as f:
        email_content = f.read()

    # Call the LLM to extract the sender's email
    sender_email = call_llm_for_sender_email(email_content)

    if sender_email:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(sender_email, encoding="utf-8")
        Response.status_code = 200
        return f"Extracted sender email: {sender_email}"
    Response.status_code = 500
    return "No sender email found in the file."

# A8 Implementation 
def extract_credit_card(image_path: str, output_file: str):
    """Extracts numeric sequences (credit card numbers) from an image using Tesseract OCR."""

    image_path_obj = Path(DATA_DIR) / Path(os.path.normpath(image_path).lstrip("\\/"))
    output_file_path = Path(DATA_DIR) / Path(os.path.normpath(output_file).lstrip("\\/"))

    # Ensure the image file exists
    if not image_path_obj.exists():
        Response.status_code = 400
        raise FileNotFoundError(f"Image file '{image_path_obj}' not found!")

    # Perform OCR with Tesseract
    extracted_text = pytesseract.image_to_string(Image.open(image_path_obj))

    # Extract numeric sequences (credit card numbers)
    cc_regex = re.compile(r"\b(\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b")
    matches = cc_regex.findall(extracted_text)

    # Save extracted numbers
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text("\n".join(matches), encoding="utf-8")

    # Return result
    if matches:
        Response.status_code = 200
        return f"Extracted {len(matches)} credit card numbers: {matches}"
    
    Response.status_code = 500
    return "No credit card numbers detected in the image."


#A9 Implementation
def find_similar_comments(input_file: str, output_file: str):
    """Finds the most similar comments in a given file using embeddings."""
    
    input_file_path = Path(DATA_DIR) /  Path(os.path.normpath(input_file).lstrip("\\/"))
    output_file_path = Path(DATA_DIR) / Path(os.path.normpath(output_file).lstrip("\\/"))

    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file '{input_file}' not found!")

    with open(input_file_path, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f.readlines()]

    if len(comments) < 2:
        return "Not enough comments for similarity comparison."

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(comments, convert_to_tensor=True)

    # Compute pairwise similarity scores
    similarity_scores = util.pytorch_cos_sim(embeddings, embeddings).tolist()

    # Find the most similar comment pair
    max_score = -1
    best_pair = ("", "")
    
    for i in range(len(comments)):
        for j in range(i + 1, len(comments)):
            if similarity_scores[i][j] > max_score:
                max_score = similarity_scores[i][j]
                best_pair = (comments[i], comments[j])

    result = {"Most Similar Comments": best_pair, "Similarity Score": max_score}
    output_file_path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    Response.status_code=200
    return f"Most similar comments saved to {output_file}."

# A10 Implementation
def compute_ticket_sales(db_path: str, ticket_type: str, output_file: str):
    """Computes total sales for a specific ticket type from an SQLite database."""
    
    db_path_obj = Path(DATA_DIR) / db_path.lstrip("/")
    output_file_path = Path(DATA_DIR) / output_file.lstrip("/")

    if not db_path_obj.exists():
        raise FileNotFoundError(f"Database file '{db_path}' not found!")

    conn = sqlite3.connect(db_path_obj)
    cursor = conn.cursor()

    cursor.execute("SELECT SUM(price) FROM tickets WHERE type = ?", (ticket_type,))
    total_sales = cursor.fetchone()[0]

    conn.close()

    if total_sales is None:
        total_sales = 0

    output_file_path.write_text(str(total_sales), encoding="utf-8")
    Response.status_code=200
    return f"Total sales for {ticket_type} tickets: {total_sales}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
