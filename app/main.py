import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.schemas.request import ChatRequest

from app.services.rag_service import run_rag
from app.services.ingestion_service import ingest_folder

app = FastAPI()

@app.post("/chat")
def chat(req: ChatRequest):
    return {"answer": run_rag(req.query, req.role, req.session_id)}


@app.post("/ingest")
def ingest():
    base_path = "resources/data"

    total = 0

    for dept in os.listdir(base_path):
        path = os.path.join(base_path, dept)

        if os.path.isdir(path):
            total += ingest_folder(path, dept)

    return {"chunks_ingested": total}