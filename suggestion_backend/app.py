import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from bs4 import BeautifulSoup
import time
import json

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = 'app.log'

file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
logger.info("Model loaded: multi-qa-mpnet-base-dot-v1")

# Global variables to hold templates and their embeddings
templates = []  # Initially empty
template_embeddings = None  # Will be computed after templates are received

# Function to remove HTML tags from text
def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Function to load default templates from all_elements.json
def load_default_templates():
    try:
        with open('all_elements.json', 'r') as f:
            default_templates = json.load(f)
        logger.info(f"Loaded {len(default_templates)} default templates from all_elements.json")
        return default_templates
    except FileNotFoundError:
        logger.error("Default template file all_elements.json not found")
        return []
    except json.JSONDecodeError:
        logger.error("Error decoding all_elements.json")
        return []

# Function to update templates and embeddings
def update_templates(new_templates):
    global templates, template_embeddings
    if not new_templates:  # Check if the provided list is empty
        logger.warning("Received empty templates list, loading default templates")
        new_templates = load_default_templates()
    if new_templates:  # Proceed only if there are templates
        templates = new_templates
        template_texts = [strip_html(t['name'] + ' ' + t['description']) for t in templates]
        template_embeddings = model.encode(template_texts, convert_to_tensor=True)
        logger.info(f"Templates updated: {len(templates)} templates loaded")
    else:
        templates = []
        template_embeddings = None
        logger.warning("No templates available after update")
    
    # Optionally save template_texts to a file for debugging
    with open('template_texts.txt', 'w', encoding='utf-8') as f:
        for idx, text in enumerate(template_texts if 'template_texts' in locals() else []):
            f.write(f"Template {idx}: {text}\n\n")

# Preload default templates at startup
default_templates = load_default_templates()
update_templates(default_templates)

# Define the input schema for updating templates
class TemplatesInput(BaseModel):
    templates: list

# Endpoint to update templates
@app.post('/update-templates')
async def update_templates_endpoint(templates_input: TemplatesInput):
    try:
        update_templates(templates_input.templates)
        logger.info("Templates updated successfully")
        return {"message": "Templates updated successfully"}
    except Exception as e:
        logger.error(f"Error updating templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Define the input schema for suggestions
class ElementInput(BaseModel):
    name: str
    type: str

# Define the suggestion endpoint
@app.post('/suggest')
async def suggest(element: ElementInput):
    start_time = time.time()
    logger.debug(f"Received input: {element.dict()}")
    if not templates or template_embeddings is None:
        logger.warning("Suggest endpoint called without initialized templates")
        raise HTTPException(status_code=400, detail="Templates not initialized")

    element_type = element.type
    element_name = strip_html(element.name)
    logger.debug(f"Processing suggestion for element: {element_name} of type {element_type}")

    # Get indices of templates that apply to the element type
    indices = [i for i, t in enumerate(templates) if element_type in t['appliesTo']]
    if not indices:
        logger.info(f"No templates found for type {element_type}")
        return []

    # Compute similarity scores
    query_embedding = model.encode(element_name, convert_to_tensor=True)
    filtered_embeddings = template_embeddings[indices]
    similarities = torch.matmul(filtered_embeddings, query_embedding)
    sim_np = similarities.cpu().numpy()

    # Filter templates with similarity > 14
    above_threshold = np.where(sim_np > 14)[0]
    logger.debug(f"Found {len(above_threshold)} templates above the threshold")
    if len(above_threshold) == 0:
        logger.info("No suggestions above the threshold")
        return []

    # Sort filtered indices by similarity in descending order
    sorted_indices = above_threshold[np.argsort(sim_np[above_threshold])[::-1]]
    # Take up to top 5
    top_indices = sorted_indices[:5]

    # Construct suggestions list
    suggestions = []
    for idx in top_indices:
        template_idx = indices[idx]
        template = templates[template_idx]
        similarity = similarities[idx].item()
        suggestions.append({
            'id': template['id'],
            'name': template['name'],
            'similarity': similarity
        })

    end_time = time.time()
    logger.debug(f"Suggestions returned: {suggestions}")
    logger.info(f"Suggestions computed in {end_time - start_time:.2f} seconds")
    return suggestions

# Middleware to log requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response: Response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    return response

# Exception handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return Response(content=exc.detail, status_code=exc.status_code)

# Run the FastAPI app with uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)