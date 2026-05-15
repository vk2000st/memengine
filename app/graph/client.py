import os
from falkordb import FalkorDB

_client = None
_graph = None

def get_falkordb_client():
    global _client
    if _client is None:
        url = os.getenv("FALKORDB_URL", "falkordb://localhost:6379")
        # Parse falkordb://host:port
        host_port = url.replace("falkordb://", "")
        host, port = host_port.split(":")
        _client = FalkorDB(host=host, port=int(port))
    return _client

def get_graph(graph_name: str):
    client = get_falkordb_client()
    return client.select_graph(graph_name)

def get_user_graph(company_id: str, agent_id: str):
    """Each company+agent combination gets its own graph for isolation."""
    graph_name = f"mem_{company_id}_{agent_id}"
    return get_graph(graph_name)
