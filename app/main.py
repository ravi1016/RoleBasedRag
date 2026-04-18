import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.schemas.request import ChatRequest

from app.services.rag_service import run_rag
from app.services.ingestion_service import ingest_folder
from app.utils.vectorstore import init_collection
from app.utils.logger import get_logger

# -----------------------------
# Logger
# -----------------------------
logger = get_logger(__name__)

app = FastAPI()


# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
def startup_event():
    logger.info("🚀 Application startup initiated")

    try:
        init_collection()
        logger.info("✅ Qdrant collection initialized successfully")

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise


# -----------------------------
# Chat endpoint
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):

    logger.info(
        f"💬 /chat request received | role={req.role} | session_id={req.session_id}"
    )

    try:
        answer = run_rag(req.query, req.role, req.session_id)

        logger.info("✅ Chat request processed successfully")

        return {"answer": answer}

    except Exception as e:
        logger.error(f"❌ /chat failed: {str(e)}", exc_info=True)
        raise


# -----------------------------
# Ingestion endpoint
# -----------------------------
@app.post("/ingest")
def ingest():

    base_path = "resources/data"

    logger.info(f"📥 Ingestion triggered | base_path={base_path}")

    total = 0

    try:
        for dept in os.listdir(base_path):
            path = os.path.join(base_path, dept)

            if os.path.isdir(path):
                logger.info(f"📂 Ingesting department: {dept}")

                count = ingest_folder(path, dept)

                logger.info(f"📊 Ingested {count} chunks | department={dept}")

                total += count

        logger.info(f"🎉 Ingestion completed | total_chunks={total}")

        return {"chunks_ingested": total}

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {str(e)}", exc_info=True)
        raise