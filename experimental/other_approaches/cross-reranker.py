from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import uvicorn
import json
import numpy as np
from bs4 import BeautifulSoup

app = FastAPI()

# 1) Load models
bi_encoder = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2) Load your element templates
with open('all_elements.json', 'r', encoding='utf-8') as f:
    templates = json.load(f)

def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()

# 3) Precompute embeddings for all templates with the bi-encoder
template_texts = []
for t in templates:
    raw_text = t['name'] + ' ' + t['description']
    clean_text = strip_html(raw_text)
    template_texts.append(clean_text)

template_embeddings = bi_encoder.encode(template_texts, convert_to_tensor=True)

# 4) Pydantic model
class ElementInput(BaseModel):
    name: str
    type: str

@app.post('/suggest')
async def suggest(element: ElementInput):
    element_type = element.type
    element_name = strip_html(element.name)

    # Filter templates by type
    indices = [i for i, t in enumerate(templates) if element_type in t['appliesTo']]
    if not indices:
        return []

    # A) First-stage retrieval with bi-encoder
    query_text = f"{element_type}: {element_name}"
    query_embedding = bi_encoder.encode(query_text, convert_to_tensor=True)

    filtered_embeddings = template_embeddings[indices]
    similarities = util.pytorch_cos_sim(query_embedding, filtered_embeddings)[0]

    top_k = 20
    # sort descending by similarity
    top_indices = np.argsort(similarities.cpu().numpy())[::-1][:top_k]
    candidates = []
    for idx_ in top_indices:
        template_idx = indices[idx_]
        # cast similarity to float
        sim_score = float(similarities[idx_].item())
        candidates.append((template_idx, sim_score))

    # B) Second-stage: CrossEncoder reranking
    cross_input = []
    for template_idx, _ in candidates:
        cand_text = template_texts[template_idx]
        cross_input.append((query_text, cand_text))

    # cross_encoder returns a list of floats or np.float32
    cross_scores = cross_encoder.predict(cross_input)

    # Sort by cross-encoder score desc
    reranked = sorted(
        zip(candidates, cross_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # C) Return the top 5
    suggestions = []
    for (template_idx, _sim), cross_score in reranked[:5]:
        template = templates[template_idx]
        suggestions.append({
            'id': template['id'],
            'name': template['name'],
            # cast cross_score to Python float so it's JSON serializable
            'similarity': float(cross_score),
        })

    return suggestions

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
