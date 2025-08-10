# PolicyMind - HackRx API (FastAPI)

## Overview
PolicyMind ingests policy PDFs, performs semantic retrieval of clauses, and returns structured JSON answers (decision, amount, justification) for each question. Built for HackRx submission.

## Setup (local)
1. Create and activate virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `HACKRX_TOKEN`.
4. Run locally:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

## Test
Use curl to test:
```bash
curl -X POST "http://127.0.0.1:8000/hackrx/run"  -H "Authorization: Bearer <HACKRX_TOKEN>"  -H "Content-Type: application/json"  -d '{"documents":"https://hackrx.blob.core.windows.net/assets/policy.pdf?...", "questions":["Does this policy cover knee surgery?"]}'
```

## Deploy
Push to GitHub and deploy to Render/Railway. Set environment variables in the hosting provider.
