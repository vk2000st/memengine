import asyncio
from typing import Any


async def graph_dedup(
    relation_label: str,
    object_value: str,
    user_id: str,
    agent_id: str,
    company_id: str,
    qdrant_client: Any,
    falkordb_graph: Any,
) -> dict:
    """
    Check if a structured memory conflicts with existing graph edges.
    Returns dict with keys: action, supersedes_edge_id, existing_object
    action: "new" | "update" | "duplicate"
    """
    try:
        # Search for existing edges with similar relation_label for this user
        # Use FalkorDB to find all RELATES_TO edges for this user
        query = """
        MATCH (u:User {user_id: $user_id, agent_id: $agent_id})-[r:RELATES_TO]->(e:Entity)
        WHERE r.is_current = true
        RETURN r.relation_label as relation_label, r.object_value as object_value,
               r.edge_id as edge_id, r.relation_embedding as relation_embedding
        """
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: falkordb_graph.query(query, {
                "user_id": user_id,
                "agent_id": agent_id,
            })
        )

        if not result or not result.result_set:
            return {"action": "new", "supersedes_edge_id": None, "existing_object": None}

        # Embed the new relation_label
        from app.services.extraction.pipeline import _embed
        new_embedding = await _embed(relation_label)

        # Compare with existing relation embeddings
        best_match = None
        best_score = 0.0
        SIMILARITY_THRESHOLD = 0.75

        for row in result.result_set:
            existing_relation = row[0]
            existing_object = row[1]
            existing_edge_id = row[2]
            existing_embedding = row[3]

            if not existing_embedding:
                continue

            # Cosine similarity
            import numpy as np
            a = np.array(new_embedding)
            b = np.array(existing_embedding)
            score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

            if score > best_score:
                best_score = score
                best_match = {
                    "edge_id": existing_edge_id,
                    "relation_label": existing_relation,
                    "object_value": existing_object,
                    "score": score,
                }

        if best_match and best_score >= SIMILARITY_THRESHOLD:
            if best_match["object_value"].lower() == object_value.lower():
                return {
                    "action": "duplicate",
                    "supersedes_edge_id": best_match["edge_id"],
                    "existing_object": best_match["object_value"],
                }
            else:
                return {
                    "action": "update",
                    "supersedes_edge_id": best_match["edge_id"],
                    "existing_object": best_match["object_value"],
                }

        return {"action": "new", "supersedes_edge_id": None, "existing_object": None}

    except Exception as e:
        # On any error, treat as new to avoid blocking
        return {"action": "new", "supersedes_edge_id": None, "existing_object": None}
