import os
import uuid
import pandas as pd
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.vectorstore import client, collection_name
from app.utils.embeddings import embeddings
from app.utils.sparse_encoder import encode_sparse
from app.utils.logger import get_logger

from qdrant_client.http import models

# -----------------------------
# Logger
# -----------------------------
logger = get_logger(__name__)


# -----------------------------
# Load files
# -----------------------------
def load_files(folder_path: str, department: str):

    logger.info(f"Loading files | folder={folder_path} | department={department}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = []
    file_count = 0
    chunk_count = 0

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)
        text = ""

        logger.debug(f"Processing file: {file}")

        # ---------------- PDF ----------------
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)

                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                logger.debug(f"PDF loaded: {file}")

            except Exception as e:
                logger.error(f"PDF read failed: {file} | {str(e)}", exc_info=True)
                continue

        # ---------------- MD ----------------
        elif file.endswith(".md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                logger.debug(f"MD loaded: {file}")

            except Exception as e:
                logger.error(f"MD read failed: {file} | {str(e)}", exc_info=True)
                continue

        # ---------------- CSV ----------------
        elif file.endswith(".csv"):
            try:
                df = pd.read_csv(file_path)

                for idx, row in df.iterrows():
                    row_text = " | ".join(
                        [f"{col}: {row[col]}" for col in df.columns]
                    )

                    docs.append(
                        Document(
                            page_content=row_text,
                            metadata={
                                "id": str(uuid.uuid4()),
                                "department": department,
                                "source": file,
                                "row": int(idx),
                                "type": "csv_row"
                            }
                        )
                    )

                logger.debug(f"CSV processed: {file} | rows={len(df)}")
                file_count += 1

            except Exception as e:
                logger.error(f"CSV read failed: {file} | {str(e)}", exc_info=True)
                continue

            continue

        else:
            logger.debug(f"Skipping unsupported file: {file}")
            continue

        # ---------------- Chunk text ----------------
        if not text.strip():
            logger.debug(f"Empty file skipped: {file}")
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "department": department,
                        "source": file,
                        "type": "text_chunk"
                    }
                )
            )
            chunk_count += 1

        file_count += 1

    logger.info(f"File loading completed | files={file_count} | chunks={chunk_count}")

    return docs


# -----------------------------
# Ingest folder
# -----------------------------
def ingest_folder(folder_path: str, department: str):

    logger.info(f"Starting ingestion | folder={folder_path} | department={department}")

    docs = load_files(folder_path, department)

    if not docs:
        logger.warning(f"No documents found | folder={folder_path}")
        return 0

    texts = [d.page_content for d in docs]

    logger.info(f"Embedding started | docs={len(texts)}")

    # -----------------------------
    # Embeddings
    # -----------------------------
    try:
        dense_vectors = embeddings.embed_documents(texts)
        sparse_vectors = encode_sparse(texts)

        logger.info("Embeddings generated successfully")

    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}", exc_info=True)
        raise

    # -----------------------------
    # Build Qdrant points
    # -----------------------------
    points = []

    for i, doc in enumerate(docs):

        points.append(
            models.PointStruct(
                id=str(doc.metadata["id"]),

                vector={
                    "dense": dense_vectors[i],
                    "bm25": models.SparseVector(
                        indices=sparse_vectors[i]["indices"],
                        values=sparse_vectors[i]["values"]
                    )
                },

                payload={
                    "text": doc.page_content,
                    **doc.metadata
                }
            )
        )

    logger.info(f"Upserting to Qdrant | points={len(points)}")

    # -----------------------------
    # Upsert to Qdrant
    # -----------------------------
    try:
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info(f"Ingestion completed | department={department} | points={len(points)}")

    except Exception as e:
        logger.error(f"Qdrant upsert failed: {str(e)}", exc_info=True)
        raise

    return len(points)