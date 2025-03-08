from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
import torch
import numpy as np
from bs4 import BeautifulSoup

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained SentenceTransformer model
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2') 
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Load element templates from JSON file
with open('all_elements.json', 'r') as f:
    templates = json.load(f)

# Function to remove HTML tags from text
def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Precompute embeddings for all templates at startup
template_texts = [strip_html(t['name'] + ' ' + t['description']) for t in templates]
template_embeddings = model.encode(template_texts, convert_to_tensor=True)

# Save template_texts to a text file
with open('template_texts.txt', 'w', encoding='utf-8') as f:
    for idx, text in enumerate(template_texts):
        f.write(f"Template {idx}: {text}\n\n")

# Define the input schema using Pydantic
class ElementInput(BaseModel):
    name: str
    type: str

@app.post('/suggest')
async def suggest(element: ElementInput):
    # Extract and clean the input
    element_type = element.type  # e.g., "bpmn:Task"
    element_name = strip_html(element.name)  # e.g., "Send email"

    # Filter templates by element type
    indices = [i for i, t in enumerate(templates) if element_type in t['appliesTo']]
    if not indices:
        return []  # Return empty list if no templates match the type

    # Encode the element's name into an embedding
    query_embedding = model.encode(element_name, convert_to_tensor=True)

    # Compute dot product similarities for filtered templates
    filtered_embeddings = template_embeddings[indices]
    similarities = torch.matmul(filtered_embeddings, query_embedding)

    # Get the top 5 matches
    top_indices = np.argsort(similarities.cpu().numpy())[::-1][:5]

    # Prepare the suggestions list
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

    return suggestions

# Run the FastAPI app with uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)