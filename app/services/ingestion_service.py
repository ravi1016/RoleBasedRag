import os
import uuid
import pandas as pd
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.vectorstore import client, collection_name
from app.utils.embeddings import embeddings
from app.utils.sparse_encoder import encode_sparse

from qdrant_client.http import models


def load_files(folder_path: str, department: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        text = ""

        # ---------------- PDF ----------------
        if file.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # ---------------- MD ----------------
        elif file.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # ---------------- CSV ----------------
        elif file.endswith(".csv"):
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
            continue

        else:
            continue

        if not text.strip():
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

    return docs


def ingest_folder(folder_path: str, department: str):

    docs = load_files(folder_path, department)

    if not docs:
        return 0

    texts = [d.page_content for d in docs]

    dense_vectors = embeddings.embed_documents(texts)
    sparse_vectors = encode_sparse(texts)

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

    client.upsert(
        collection_name=collection_name,
        points=points
    )

    return len(points)