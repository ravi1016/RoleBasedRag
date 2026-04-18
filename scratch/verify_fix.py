import os
from dotenv import load_dotenv
load_dotenv()

from app.services.retrieval_service import retrieve

# Mock role and query
query = "Software Subscriptions"
role = "Finance" # Assuming Finance role has access to some docs

try:
    print(f"Testing retrieval for query: '{query}' with role: '{role}'")
    docs = retrieve(query, role)
    print("\n✅ Retrieval successful!")
    print(f"Number of documents returned: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc[:100]}...")
except Exception as e:
    print(f"\n❌ Retrieval failed with error: {e}")
    import traceback
    traceback.print_exc()
