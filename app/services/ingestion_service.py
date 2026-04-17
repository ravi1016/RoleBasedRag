import os
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.utils.vectorstore import vectorstore


def ingest_folder(folder_path: str, department: str):

    print(f"\n📂 Processing folder: {folder_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_docs = []

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        text = ""

        # -------------------------
        # PDF FILES
        # -------------------------
        if file.endswith(".pdf"):
            reader = PdfReader(file_path)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # -------------------------
        # MARKDOWN FILES
        # -------------------------
        elif file.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # -------------------------
        # CSV FILES (NEW)
        # -------------------------
        elif file.endswith(".csv"):

            print(f"📊 Loading CSV: {file}")

            df = pd.read_csv(file_path)

            for idx, row in df.iterrows():

                row_text = " | ".join(
                    [f"{col}: {row[col]}" for col in df.columns]
                )

                all_docs.append(
                    Document(
                        page_content=row_text,
                        metadata={
                            "department": department,
                            "source": file,
                            "row": int(idx)
                        }
                    )
                )

            print(f"✂️ CSV Rows Added: {len(df)}")
            continue

        else:
            print(f"⏭ Skipping unsupported file: {file}")
            continue

        # -------------------------
        # TEXT CHUNKING (PDF / MD)
        # -------------------------
        print(f"📄 File: {file}")
        print(f"🧠 Text length: {len(text)}")

        if not text.strip():
            print("❌ Empty file skipped")
            continue

        chunks = splitter.split_text(text)

        print(f"✂️ Chunks: {len(chunks)}")

        for chunk in chunks:
            all_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "department": department,
                        "source": file
                    }
                )
            )

    print(f"\n📦 TOTAL DOCS READY: {len(all_docs)}")

    if not all_docs:
        return 0

    vectorstore.add_documents(all_docs)

    print("✅ INGESTION SUCCESS")

    return len(all_docs)