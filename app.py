# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "google-genai",
#   "python-dotenv",
# ]
# ///

from fastapi import FastAPI,File, UploadFile    
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import os
import re
import subprocess

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, we'll just continue without it
    pass

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"]) # Allow GET requests from all origins
# Or, provide more granular control:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow a specific domain
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow specific methods
    allow_headers=["*"],  # Allow all headers
)

def task_breakdown(task:str):
    """Send the question to the LLM and return its response."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set! "
            "Please set it with your Google AI API key. "
            "You can get one from https://aistudio.google.com/apikey"
        )
    client = genai.Client(api_key=api_key)
    task_breakdown_file = os.path.join('prompts', "task_breakdown.txt")
    with open(task_breakdown_file, 'r') as f:
        task_breakdown_prompt = f.read()
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[task,task_breakdown_prompt],
    )
    return response.text

@app.get("/")
async def root():
    return {"message": "Hello!"}


import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import pandas as pd

# Directly include scraping and extraction functions from main.py/tools
def get_relevant_data(file_name: str, js_selector: str = None):
    with open(file_name, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    if js_selector:
        elements = soup.select(js_selector)
        return {"data": [el.get_text(strip=True) for el in elements]}
    return {"data": soup.get_text(strip=True)}

async def scrape_website(url: str, output_file: str = "scraped_content.html"):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            content = await page.content()
            with open(output_file, "w", encoding="utf-8") as file:
                await file.write(content)
        except Exception as e:
            print(f"Failed to load page: {e}")
            await browser.close()
            return
        await browser.close()

@app.post("/api/")
async def process_input(
    questions_txt: UploadFile = File(None, alias="questions.txt"),
    image_png: UploadFile = File(None, alias="image.png"),
    data_csv: UploadFile = File(None, alias="data.csv")
):
    try:
        # Only process questions.txt for LLM and code execution
        if questions_txt:
            q_content = await questions_txt.read()
            q_text = q_content.decode("utf-8")
            llm_response = task_breakdown(q_text)
            # Try to extract code block from LLM response
            code_match = re.search(r"```python(.*?)```", llm_response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                # Write code to temp file and inject helper functions
                with open("temp_script.py", "w", encoding="utf-8") as f:
                    f.write('import asyncio\n')
                    f.write('from bs4 import BeautifulSoup\n')
                    f.write('from playwright.async_api import async_playwright\n')
                    f.write('import pandas as pd\n')
                    f.write('import sys\n')
                    f.write('import json\n')
                    f.write('def get_relevant_data(file_name, js_selector=None):\n    with open(file_name, encoding="utf-8") as f:\n        html = f.read()\n    soup = BeautifulSoup(html, "html.parser")\n    if js_selector:\n        elements = soup.select(js_selector)\n        return {"data": [el.get_text(strip=True) for el in elements]}\n    return {"data": soup.get_text(strip=True)}\n')
                    f.write('async def scrape_website(url, output_file="scraped_content.html"):\n    async with async_playwright() as p:\n        browser = await p.chromium.launch(headless=True)\n        page = await browser.new_page()\n        try:\n            await page.goto(url, wait_until="domcontentloaded", timeout=60000)\n            content = await page.content()\n            with open(output_file, "w", encoding="utf-8") as file:\n                await file.write(content)\n        except Exception as e:\n            print(f"Failed to load page: {e}")\n            await browser.close()\n            return\n        await browser.close()\n')
                    f.write(code)
                # Execute the code
                result = subprocess.run(["python", "temp_script.py"], capture_output=True, text=True)
                # Try to parse output as JSON array
                import json
                try:
                    answers = json.loads(result.stdout)
                except Exception:
                    array_match = re.search(r"\[.*?\]", result.stdout, re.DOTALL)
                    if array_match:
                        try:
                            answers = json.loads(array_match.group(0))
                        except Exception:
                            answers = [array_match.group(0)]
                    else:
                        answers = [result.stdout.strip()]
                return answers
            else:
                # If no code, try to extract only the answer array from LLM response
                array_match = re.search(r"\[.*?\]", llm_response, re.DOTALL)
                if array_match:
                    import json
                    try:
                        answers = json.loads(array_match.group(0))
                    except Exception:
                        answers = [array_match.group(0)]
                    return answers
                return []
        return JSONResponse(status_code=400, content={"error": "No valid question provided."})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)