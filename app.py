"""
PolicyMind API - Multi-Backend LLM Support
Supports:
1. OpenAI API
2. Azure OpenAI Service
3. GitHub Models API

Switch backend via .env
"""

import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI, AzureOpenAI
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load .env
load_dotenv()

BACKEND = os.getenv("LLM_BACKEND", "github").lower()
HACKRX_AUTH_TOKEN = "Bearer 18d7f8e1476570072d32707d6b6e0a57c4397055eeaba964ff5736aa67d81f5c"

# Initialize LLM client
if BACKEND == "github":
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        raise RuntimeError("❌ Missing GITHUB_TOKEN in .env for GitHub backend")
    client = OpenAI(base_url="https://models.github.ai/inference", api_key=GITHUB_TOKEN)
    MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
    EMBEDDING_MODEL = "text-embedding-3-small"  # GitHub supports OpenAI's embeddings

elif BACKEND == "azure":
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    MODEL_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")

elif BACKEND == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
    EMBEDDING_MODEL = "text-embedding-3-small"

else:
    raise RuntimeError(f"❌ Unknown backend: {BACKEND}")


def download_and_extract_pdf(url):
    """Download PDF and extract all text."""
    pdf_path = "temp.pdf"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {e}")

    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF text: {e}")

    return text.strip()


def chunk_text(text, max_chars=1500):
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i + max_chars])
    return chunks


def embed_texts(texts):
    """Get embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embeddings.append(resp.data[0].embedding)
    return np.array(embeddings)


app = FastAPI(title="PolicyMind API", version="4.0.0")


@app.post("/hackrx/run")
async def hackrx_run(request: Request):
    # Auth
    if request.headers.get("Authorization") != HACKRX_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.json()
        document_url = data.get("documents")
        questions = data.get("questions", [])

        if not document_url or not questions:
            raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions'")

        # Step 1: Download & parse
        full_text = download_and_extract_pdf(document_url)
        chunks = chunk_text(full_text)

        # Step 2: Embed all chunks
        chunk_embeddings = embed_texts(chunks)

        answers = []
        for q in questions:
            # Step 3: Embed question
            q_embedding = embed_texts([q])[0].reshape(1, -1)

            # Step 4: Find top 3 most relevant chunks
            similarities = cosine_similarity(q_embedding, chunk_embeddings)[0]
            top_idx = np.argsort(similarities)[::-1][:3]
            relevant_chunks = "\n\n".join([chunks[i] for i in top_idx])

            # Step 5: Ask LLM using only relevant chunks
            prompt = f"""
You are an insurance policy assistant.
Answer the question based ONLY on the document content below:

--- DOCUMENT ---
{relevant_chunks}
--- END DOCUMENT ---

Question: {q}
"""
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts exact answers from insurance policies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            answers.append(response.choices[0].message.content.strip())

        return {"backend": BACKEND, "answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
