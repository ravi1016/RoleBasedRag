from qdrant_client import models
from app.utils.embeddings import embeddings
from app.utils.sparse_encoder import encode_sparse_single
from app.utils.rbac import get_allowed_depts
from app.utils.vectorstore import client, collection_name
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_filter(role: str):
    allowed = get_allowed_depts(role)

    logger.debug(f"RBAC filter | role={role} | allowed_depts={allowed}")

    return models.Filter(
        must=[
            models.FieldCondition(
                key="department",
                match=models.MatchAny(any=allowed)
            )
        ]
    )


def retrieve(query: str, role: str, k: int = 5):

    logger.info(f"Retrieval start | query='{query}' | role='{role}' | k={k}")

    # 1. Dense embedding
    try:
        dense_vector = embeddings.embed_query(query)
        logger.debug("Dense embedding generated")
    except Exception as e:
        logger.error(f"Dense embedding failed: {str(e)}", exc_info=True)
        raise

    # 2. Sparse embedding
    try:
        sparse_vector = encode_sparse_single(query)
        logger.debug(f"Sparse embedding generated | non-zero terms={len(sparse_vector['indices'])}")
    except Exception as e:
        logger.error(f"Sparse encoding failed: {str(e)}", exc_info=True)
        raise

    # 3. Query execution
    try:
        results = client.query_points(
            collection_name=collection_name,

            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),

            prefetch=[
                models.Prefetch(
                    query=models.NearestQuery(nearest=dense_vector),
                    using="dense",
                    limit=20
                ),
                models.Prefetch(
                    query=models.NearestQuery(
                        nearest=models.SparseVector(**sparse_vector)
                    ),
                    using="bm25",
                    limit=20
                )
            ],

            limit=k,
            query_filter=build_filter(role)
        ).points

        logger.info(f"Retrieval completed | results={len(results)}")

    except Exception as e:
        logger.error(f"Qdrant query failed: {str(e)}", exc_info=True)
        raise

    # 4. Debug top results (safe preview only)
    for i, r in enumerate(results[:3]):
        text_preview = r.payload.get("text", "")[:120]
        logger.debug(f"Top[{i}] id={r.id} preview={text_preview}")

    return results