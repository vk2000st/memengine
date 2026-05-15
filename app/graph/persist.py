import asyncio
import uuid
from typing import Any


async def graph_persist(
    relation_label: str,
    object_value: str,
    user_id: str,
    agent_id: str,
    company_id: str,
    memory_id: str,
    action: str,
    supersedes_edge_id: str | None,
    falkordb_graph: Any,
) -> str:
    """
    Write a structured memory to FalkorDB.
    Returns the new edge_id.
    action: "new" | "update"
    """
    from app.services.extraction.pipeline import _embed

    # Embed the relation label for future similarity comparison
    relation_embedding = await _embed(relation_label)
    edge_id = str(uuid.uuid4())

    try:
        # If update, mark old edge as superseded
        if action == "update" and supersedes_edge_id:
            supersede_query = """
            MATCH (u:User {user_id: $user_id, agent_id: $agent_id})-[r:RELATES_TO {edge_id: $edge_id}]->(e:Entity)
            SET r.is_current = false, r.superseded_at = timestamp()
            """
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: falkordb_graph.query(supersede_query, {
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "edge_id": supersedes_edge_id,
                })
            )

        # Create new edge
        create_query = """
        MERGE (u:User {user_id: $user_id, agent_id: $agent_id, company_id: $company_id})
        MERGE (e:Entity {value: $object_value})
        CREATE (u)-[r:RELATES_TO {
            edge_id: $edge_id,
            relation_label: $relation_label,
            object_value: $object_value,
            relation_embedding: $relation_embedding,
            memory_id: $memory_id,
            is_current: true,
            created_at: timestamp(),
            superseded_at: null
        }]->(e)
        RETURN r.edge_id as edge_id
        """
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: falkordb_graph.query(create_query, {
                "user_id": user_id,
                "agent_id": agent_id,
                "company_id": company_id,
                "object_value": object_value,
                "edge_id": edge_id,
                "relation_label": relation_label,
                "relation_embedding": relation_embedding,
                "memory_id": memory_id,
            })
        )
        return edge_id

    except Exception as e:
        raise RuntimeError(f"graph_persist failed: {e}")
