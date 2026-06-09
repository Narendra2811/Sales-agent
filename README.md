# SaaSify Sales Assistant Agent

A production-grade persistent B2B SaaS sales assistant API built with FastAPI, LangGraph, LangChain, and OpenAI gpt-3.5-turbo.

**Live URL:** `https://your-app.railway.app` ← replace after Railway deploy

---

## Architecture Diagram


![Architecture Illustration](media/flow.png)

---

## Memory Design Decision

### Two-Tier Architecture

**Short-term memory** (conversation flow):
- Raw messages: last 10 turns injected as conversation history
- Summaries: when message count > 20, the model auto-compresses older messages into a paragraph
- Bounded token cost — no matter how long the conversation, we inject at most ~10 raw messages + 1 summary

**Long-term memory** (user profile):
- Extracted facts: after every turn, a lightweight model call scans the exchange for new user facts
- Stored as key-value pairs: `team_size=50`, `budget=$500/mo`, `plan_interest=Enterprise`
- Persists across ALL sessions forever (until GDPR delete)
- Injected into the system prompt on every request

**Why this design?**
> A sales assistant's superpower is remembering context. "We discussed Enterprise pricing last week" should be available this week. SQLite with SQLAlchemy (abstracted behind a `MemoryBackend` interface) lets us ship fast while keeping the swap-to-Postgres path trivial.

**What we'd use at scale:**
> `Mem0` or a dedicated vector store (Pinecone/Weaviate) for semantic memory search. The `MemoryBackend` abstraction means this change touches exactly ONE file (`chat_service.py` import).

---

## Eval Design

Every response is scored by a **second model call** immediately after the agent responds.

Three dimensions (0.0 -> 1.0):
| Score | What it measures |
|---|---|
| `groundedness` | Are all facts backed by the catalog? (detects hallucination) |
| `relevance` | Does it actually answer the user's question? |
| `confidence` | Overall quality — triggers flagging if < 0.7 |

**Limitations:**
- LLM self-scoring is not perfectly calibrated. the model can be generous.
- Adds ~1-2s latency and ~300 tokens per request.
- Scores are relative, not absolute — useful for trend analysis, not hard thresholds.

**What we'd replace it with at scale:**
> A fine-tuned reward model trained on human-labeled response pairs. Or integrate `Ragas` for RAG-specific eval (faithfulness + answer relevancy + context precision).

---

## Cross-Session Memory Demo (curl)

### Call 1 — Set context in session 1
```bash
curl -X POST https://your-app.railway.app/chat/demo_user \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi! We are a 50-person fintech company considering Enterprise. Our CTO will make the final call."}'
```

Expected: Agent answers about Enterprise pricing. Facts extracted: `team_size=50`, `industry=fintech`, `decision_maker=CTO`.

### Call 2 — New session, agent remembers
```bash
curl -X POST https://your-app.railway.app/chat/demo_user \
  -H "Content-Type: application/json" \
  -d '{"message": "Does the plan we discussed include audit logs?"}'
```

Expected: Agent says "Yes, Enterprise (which we discussed for your 50-person team) includes full audit logs" — without being told the plan or team size again. Memory persists.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat/{user_id}` | Send message, get response + eval |
| `GET` | `/chat/{user_id}/history` | Full conversation history |
| `DELETE` | `/chat/{user_id}/memory` | GDPR wipe |
| `GET` | `/chat/{user_id}/evals` | Aggregated quality stats (bonus) |
| `GET` | `/catalog` | Raw product catalog |
| `GET` | `/health` | Service health check |
| `GET` | `/docs` | Swagger UI |

---

## Local Setup

```bash
git clone <repo>
cd sales_agent

# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run database migrations
python3 -m alembic -c alembic.ini upgrade head

# 4. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI.

---

## Deploy to Railway

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "initial commit"
gh repo create sales-agent --public --push

# 2. Go to railway.app -> New Project -> Deploy from GitHub repo
# 3. Add environment variables in Railway dashboard:
#    OPENAI_API_KEY = sk-your-openai-key
#    DATABASE_URL = sqlite:///./sales_agent.db  (or Railway Postgres URL)
# 4. Railway auto-detects Dockerfile and deploys
# 5. Copy the Railway URL -> update this README
```

---

## Project Structure

```
sales_agent/
├── app/
│   ├── api/routes.py          ← HTTP endpoints (thin layer only)
│   ├── agents/agent_loop.py   ← LangGraph StateGraph (ReAct loop)
│   ├── memory/
│   │   ├── base.py            ← Abstract interface (swap in 1 file)
│   │   └── sqlite_backend.py  ← SQLite implementation
│   ├── tools/search_catalog.py ← BM25 + Semantic + RRF hybrid search
│   ├── services/
│   │   ├── chat_service.py    ← Orchestration
│   │   └── eval_service.py    ← Self-scoring + fact extraction
│   ├── models/schemas.py      ← Pydantic request/response models
│   └── db/
│       ├── models.py          ← SQLAlchemy ORM models (6 tables)
│       └── database.py        ← Engine + session factory
├── alembic/                   ← DB migration scripts
├── catalog.json               ← Mock product catalog
├── main.py                    ← FastAPI app + lifespan startup
├── Dockerfile                 ← Railway deployment
└── MASTER_GUIDE.md            ← Deep-dive documentation
```
