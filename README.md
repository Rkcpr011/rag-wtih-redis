# RAG + Valkey + BetterDB

## Project structure

```
.
├── Dockerfile            ← Python app ka image
├── docker-compose.yml    ← Valkey + BetterDB + RAG app
├── app.py                ← FastAPI server
├── rag.py                ← RAG pipeline (cache integrated)
├── semantic_cache.py     ← Valkey semantic cache logic
├── requirements.txt
└── .env.example
```

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        docker-compose network    │
                    │                                  │
  User              │  ┌──────────┐   redis://valkey   │
  POST /ask  ──────►│  │ rag-app  │──────────────────► │
                    │  │ :8000    │                    │  ┌────────┐
                    │  └──────────┘                    │  │ Valkey │
                    │                                  │  │ :6379  │
                    │  ┌──────────┐   redis://valkey   │  └────────┘
                    │  │ betterdb │──────────────────► │       │
  Browser           │  │ :3001    │◄──────────────────►│       │
  Dashboard ───────►│  └──────────┘   metrics pull     │       │
                    │                                  │       │
                    └──────────────────────────────────┘       │
                                                     persisted │
                                                     volume ───┘
```

## Quick start

```bash
# 1. API key set karo
cp .env.example .env
# OPENAI_API_KEY fill karo

# 2. Sab kuch start karo
docker-compose up -d

# 3. Logs dekho
docker-compose logs -f rag-app

# 4. Test karo
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'

# 5. BetterDB dashboard
# Browser mein: http://localhost:3001
```

## Important: localhost vs container name

| Context | Valkey URL |
|---|---|
| Local development (Python bahar) | `redis://localhost:6379` |
| Docker Compose ke andar | `redis://valkey:6379` |

Compose file mein `VALKEY_URL: redis://valkey:6379` set hai — containers ek hi network mein hain toh container name se baat karte hain.

## API endpoints

| Endpoint | Method | Kya karta hai |
|---|---|---|
| `/ask` | POST | RAG query |
| `/health` | GET | App + Valkey health check |
| `/cache/stats` | GET | Cache entries count, memory |
| `/cache/clear` | DELETE | Cache flush |
| `/docs` | GET | FastAPI auto-generated docs |

## BetterDB dashboard (localhost:3001)

Yahan dekho:
- Live commands per second Valkey pe
- Slowlog — kaunse commands slow hain
- Memory usage over time
- Connected clients
- Cache HIT/MISS patterns (SET vs GET ratio se samjho)
