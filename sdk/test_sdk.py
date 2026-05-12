from memengine import MemEngine

BASE_URL = "http://localhost:8000"

client = MemEngine(BASE_URL)

print("1. Registering company...")
api_key = client.register("Test Company")
print(f"   API key: {api_key}\n")

print("2. Creating agent...")
agent = client.create_agent(
    agent_slug="demo-agent",
    name="Demo Agent",
    extraction_instructions=(
        "Extract long-term facts about the user: preferences, habits, goals, "
        "dietary needs, hobbies, and personal details. Skip small talk."
    ),
)
print(f"   Agent: {agent.agent_slug} ({agent.id})\n")

print("3. Adding memory from conversation...")
result = client.add(
    agent_slug="demo-agent",
    user_id="alice",
    session_id="session-1",
    messages=[
        {"role": "user", "content": "I just moved to Amsterdam and I'm learning Dutch. I'm also vegetarian and I love cycling."},
        {"role": "assistant", "content": "That's great! Amsterdam is perfect for cycling. I'll remember your preferences."},
        {"role": "user", "content": "Yeah, and I work as a product designer. My goal this year is to run a half-marathon."},
    ],
)
print(f"   Trace ID : {result.trace_id}")
print(f"   Persisted: {len(result.persisted)} memories")
for m in result.persisted:
    print(f"     [{m.memory_type:12s}  {m.importance_score:.2f}]  {m.content}")
print(f"   Rejected : {len(result.rejected)} candidates\n")

print("4. Searching memories...")
resp = client.search(agent_slug="demo-agent", user_id="alice", query="lifestyle and hobbies")
print(f"   Rewritten query: {resp.rewritten_query}")
print(f"   Results ({len(resp.results)}):")
for r in resp.results:
    print(f"     [{r.similarity_score:.3f}]  {r.content}")
print()

print("5. Fetching pipeline trace...")
trace = client.get_trace(result.trace_id)
print(f"   Status     : {trace.status}")
print(f"   LLM calls  : {trace.llm_calls_total}")
print(f"   Tokens used: {trace.tokens_used}")
print(f"   Candidates : {len(trace.candidates)}\n")

print("6. Listing rejected candidates...")
rejected = client.list_rejected(result.trace_id)
if rejected:
    for c in rejected:
        print(f"   REJECTED: {c.content!r}  — {c.rejection_reason}")
else:
    print("   None rejected.")
print()

print("Done.")
