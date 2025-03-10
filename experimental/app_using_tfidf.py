from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from bs4 import BeautifulSoup

# Initialize FastAPI app
app = FastAPI()

# Load element templates from JSON file
with open('all_elements.json', 'r') as f:
    templates = json.load(f)

# Function to remove HTML tags from text
def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

# Precompute TF-IDF vectors for all templates at startup
template_texts = [strip_html(t['name'] + ' ' + t['description']) for t in templates]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(template_texts)

# Save template_texts to a text file (optional, as in your original code)
with open('template_texts.txt', 'w', encoding='utf-8') as f:
    for idx, text in enumerate(template_texts):
        f.write(f"Template {idx}: {text}\n\n")

# Define the input schema using Pydantic
class ElementInput(BaseModel):
    name: str
    type: str

# Define the suggestion endpoint
@app.post('/suggest')
async def suggest(element: ElementInput):
    # Extract and clean the input
    element_type = element.type  # e.g., "bpmn:Task"
    element_name = strip_html(element.name)  # e.g., "Send email"

    # Filter templates by element type
    indices = [i for i, t in enumerate(templates) if element_type in t['appliesTo']]
    if not indices:
        return []  # Return empty list if no templates match the type

    # Transform the input name into a TF-IDF vector
    query_tfidf = vectorizer.transform([element_name])

    # Get the TF-IDF vectors for the filtered templates
    filtered_tfidf = tfidf_matrix[indices]

    # Compute cosine similarities
    similarities = cosine_similarity(query_tfidf, filtered_tfidf)[0]

    # Get the top 5 matches
    top_indices = np.argsort(similarities)[::-1][:5]

    # Prepare the suggestions list
    suggestions = []
    for idx in top_indices:
        template_idx = indices[idx]
        template = templates[template_idx]
        similarity = similarities[idx]  # Already a float from NumPy
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