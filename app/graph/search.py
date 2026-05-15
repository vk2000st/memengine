import asyncio
from typing import Any


async def graph_search(
    query: str,
    user_id: str,
    agent_id: str,
    company_id: str,
    falkordb_graph: Any,
) -> list[dict]:
    """
    Retrieve all current structured memories for a user from FalkorDB.
    Returns list of dicts with memory_id, relation_label, object_value.
    """
    try:
        cypher = """
        MATCH (u:User {user_id: $user_id, agent_id: $agent_id})-[r:RELATES_TO]->(e:Entity)
        WHERE r.is_current = true
        RETURN r.memory_id as memory_id,
               r.relation_label as relation_label,
               r.object_value as object_value,
               r.edge_id as edge_id
        """
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: falkordb_graph.query(cypher, {
                "user_id": user_id,
                "agent_id": agent_id,
            })
        )

        if not result or not result.result_set:
            return []

        return [
            {
                "memory_id": row[0],
                "relation_label": row[1],
                "object_value": row[2],
                "edge_id": row[3],
            }
            for row in result.result_set
            if row[0]  # skip rows with no memory_id
        ]

    except Exception:
        return []
