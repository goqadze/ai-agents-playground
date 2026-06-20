# Database

PostgreSQL 16 running in Docker.

## Connection details

| Field    | Value      |
|----------|------------|
| Host     | `localhost` |
| Port     | `5432`     |
| Database | `chatdb`   |
| User     | `chatuser` |
| Password | `chatpass` |

---

## Connect with pgAdmin

### 1. Start the stack

```bash
docker compose up -d
```

### 2. Open pgAdmin

pgAdmin runs as part of the Docker stack. Open http://localhost:5050 in your browser.

Login with:
- **Email:** `admin@admin.com`
- **Password:** `admin`

### 3. Register a new server

1. Right-click **Servers** in the left panel → **Register → Server…**

2. **General** tab:
   - Name: `DeepChat` (any label you like)

3. **Connection** tab:

   | Field                | Value      |
   |----------------------|------------|
   | Host name / address  | `localhost` |
   | Port                 | `5432`     |
   | Maintenance database | `chatdb`   |
   | Username             | `chatuser` |
   | Password             | `chatpass` |

   Check **Save password** so you don't have to re-enter it.

4. Click **Save**.

### 4. Browse the schema

```
Servers
└── DeepChat
    └── Databases
        └── chatdb
            └── Schemas
                └── public
                    └── Tables
                        ├── conversations
                        └── messages
```

### 5. Run a query

Right-click `chatdb` → **Query Tool**, then try:

```sql
-- List all conversations
SELECT * FROM conversations ORDER BY created_at DESC;

-- List all messages for a conversation
SELECT * FROM messages WHERE conversation_id = 1 ORDER BY created_at;

-- Full conversation with messages
SELECT
    c.id,
    c.title,
    m.role,
    m.content,
    m.agent_steps,
    m.created_at
FROM conversations c
JOIN messages m ON m.conversation_id = c.id
ORDER BY c.id, m.created_at;
```

---

## Schema

```sql
conversations
  id          SERIAL PRIMARY KEY
  title       VARCHAR(255)
  created_at  TIMESTAMPTZ
  updated_at  TIMESTAMPTZ

messages
  id               SERIAL PRIMARY KEY
  conversation_id  INTEGER → conversations(id)
  role             VARCHAR(20)   -- 'user' | 'assistant'
  content          TEXT
  agent_steps      JSONB         -- e.g. ["Analyzing question...", "Researching deeply..."]
  created_at       TIMESTAMPTZ
```
