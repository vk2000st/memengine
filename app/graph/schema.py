INIT_QUERIES = [
    # Create indexes for fast lookups
    "CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.user_id)",
    "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.value)",
]

def init_graph(graph):
    """Initialize schema and indexes for a graph."""
    for query in INIT_QUERIES:
        try:
            graph.query(query)
        except Exception:
            pass  # Index may already exist
