# Kiến trúc Hermes Memory Stack

Tài liệu kỹ thuật chi tiết. Hướng dẫn cài đặt/sử dụng nhanh: xem [README.md](README.md).

## 1. Tổng quan

Memory backend đóng gói bằng Docker, cấp cho Hermes Desktop bộ nhớ dài hạn
(long-term memory) theo mô hình **single-user, local-first**: mỗi người dùng
chạy một stack độc lập, dữ liệu nằm hoàn toàn trên máy của họ, không sync,
không chia sẻ.

```
┌─ MÁY NGƯỜI DÙNG ────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌─ Hermes Desktop (native app — LLM chat chạy ở đây) ────────────────┐  │
│  │                                                                     │  │
│  │   hooks.post_llm_call ──────────────┐        MCP client ─────────┐ │  │
│  └──────────────────────────────────────┼───────────────────────────┼─┘  │
│                                         │ (sau mỗi lượt chat)       │    │
│                              hooks/post_llm_call.py                 │    │
│                                         │ POST /memory/append       │    │
│                                         ▼                           ▼    │
│  ┌─ Docker Compose ────────────────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  ┌─ llamaindex-service (FastAPI, host :8800 → container :8000) ───┐  │ │
│  │  │                                                                 │  │ │
│  │  │   REST API           MCP Streamable HTTP (/mcp)                 │  │ │
│  │  │      │                        │                                 │  │ │
│  │  │      └────────┬───────────────┘                                 │  │ │
│  │  │               ▼                                                 │  │ │
│  │  │   ┌─ Memory Engine ────────────────────────────────┐            │  │ │
│  │  │   │ L1 Working   ChatMemoryBuffer (dựng theo phiên)│            │  │ │
│  │  │   │ L2 Episodic  memory_store.py                   │            │  │ │
│  │  │   │ L3 Semantic  memories.py + consolidation.py    │            │  │ │
│  │  │   │ L4 Knowledge documents.py                      │            │  │ │
│  │  │   └────────────────────────────────────────────────┘            │  │ │
│  │  │               │                                                 │  │ │
│  │  │   Embedding: fastembed ONNX (bake trong image, local, no key)   │  │ │
│  │  │   LLM (optional): none|anthropic|openai|nvidia|ollama           │  │ │
│  │  └──────────────┬──────────────────────────────────────────────────┘  │ │
│  │                 │ HTTP :6333                                          │ │
│  │  ┌─ Qdrant ─────▼─────────────────────────────────────┐               │ │
│  │  │ hermes_chat_history │ hermes_memories │             │               │ │
│  │  │ hermes_documents    │ hermes_meta     │             │               │ │
│  │  └─── volume: qdrant_data ───────────────┘             │               │ │
│  │                                                        │               │ │
│  │  [ollama] — optional profile (--profile ollama)        │               │ │
│  └────────────────────────────────────────────────────────┘               │ │
│         volume: hermes_data (/data — file tài liệu gốc)                   │ │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. Bốn tầng bộ nhớ

| Tầng | Vai trò | Lưu ở | Module |
|---|---|---|---|
| **L1 Working** | Ngữ cảnh phiên hiện tại — `ChatMemoryBuffer` (giới hạn ~3000 token) dựng lại từ L2 mỗi lượt | RAM (dựng lại mỗi request) | `main.py` |
| **L2 Episodic** | Từng lượt hội thoại thô (user + assistant), có vector → tìm được theo ngữ nghĩa xuyên phiên | `hermes_chat_history` | `memory_store.py` |
| **L3 Semantic** | Fact đã chưng cất từ L2: quyết định, sở thích, thông tin dự án, task — thứ đáng nhớ vĩnh viễn | `hermes_memories` | `memories.py`, `consolidation.py` |
| **L4 Knowledge** | Tài liệu ingest (PDF/text/markdown) — RAG cổ điển | `hermes_documents` + file gốc trong `/data/documents` | `documents.py` |

## 3. Luồng dữ liệu

### 3a. Luồng ghi (mỗi lượt chat, tự động)

```
User chat trong Hermes Desktop
  → Hermes bắn event post_llm_call, pipe JSON vào stdin của hook:
      {session_id, extra: {user_message, assistant_response, ...}}
  → hooks/post_llm_call.py  (best-effort, không bao giờ làm hỏng lượt chat)
  → POST /memory/append
  → memory_store.add_message():
      - embed nội dung (fastembed, trong container)
      - point ID = uuid5(user_id : session_id : role : sha256(content))
        → IDEMPOTENT: retry/ghi trùng không tạo bản ghi mới
      - upsert vào hermes_chat_history
      - cập nhật last_written_at trong hermes_meta (để phát hiện hook chết)
```

### 3b. Luồng chưng cất (consolidation, L2 → L3)

```
Kết thúc phiên / theo yêu cầu:
  MCP tool consolidate_session(session_id)
    │
    ├─ Service CÓ LLM (LLM_PROVIDER != none):
    │    lấy turn chưa xử lý → LLM extract facts (JSON) → save_facts()
    │    → đánh dấu turn consolidated=true
    │
    └─ Service KHÔNG có LLM (mặc định):
         trả transcript + hướng dẫn extract cho CHÍNH model của Hermes
         → Hermes tự chưng cất → gọi MCP tool save_memories(facts)

save_facts() chống trùng lặp 2 lớp:
  1. Hash chính xác (point ID từ sha256 text chuẩn hoá) → bỏ qua nếu trùng
  2. Similarity ≥ 0.92 với fact đang hiệu lực → fact cũ bị đánh dấu
     superseded_by=<id mới> (giữ lại để truy vết, loại khỏi recall)
```

### 3c. Luồng đọc (recall)

```
POST /memory/recall {query, session_id?}   (hoặc MCP tool memory_recall)
  │
  ├─ L3: search hermes_memories (lọc superseded)
  │      điểm = similarity × 0.5^(tuổi/90 ngày) × (0.5 + 0.5×importance)
  ├─ L2: search hermes_chat_history xuyên phiên (loại phiên hiện tại)
  │      điểm = similarity × 0.5^(tuổi/30 ngày)
  └─ L1: N lượt gần nhất của phiên hiện tại (theo thời gian)
  │
  └→ context_block sẵn sàng inject vào system prompt:
       [Ghi nhớ dài hạn] … [Hội thoại cũ liên quan] … [Các lượt gần nhất] …
```

## 4. Schema Qdrant

Tất cả collection vector đều Cosine, dimension theo embedding model
(mặc định 384 — `paraphrase-multilingual-MiniLM-L12-v2`).

### `hermes_chat_history` (L2)
```jsonc
// point ID: uuid5("msg:{user_id}:{session_id}:{role}:{sha256(content)}")
{
  "user_id": "local",        // payload index — sẵn sàng multi-user tương lai
  "session_id": "…",         // payload index
  "project_id": "erp",       // payload index — project sidebar (xem mục 11)
  "role": "user|assistant",  // payload index
  "content": "…",
  "timestamp": 1783229012.9, // payload index (float)
  "consolidated": false      // payload index — đã chưng cất chưa
}
```

### `hermes_memories` (L3)
```jsonc
// point ID: uuid5("fact:{user_id}:{sha256(normalized_text)}")
{
  "user_id": "local",
  "session_id": "…",              // phiên nguồn
  "project_id": "erp",            // payload index — kế thừa từ phiên nguồn
  "type": "fact|preference|decision|task",
  "text": "…",
  "importance": 0.8,              // 0..1
  "created_at": 1783229012.9,
  "superseded_by": "<point-id>"   // chỉ có khi bị fact mới thay thế
}
```

### `hermes_documents` (L4)
Do LlamaIndex `QdrantVectorStore` quản lý (chunk + metadata `user_id`,
`source`, `stored_path` → file gốc trong `/data/documents/<sha12>_<tên>`).

### `hermes_meta` (guard)
1 point cố định, vector dim=1: `{schema_version, embed_provider, embed_model,
embed_dim, last_written_at}`. Lúc khởi động service so config hiện tại với
meta — **lệch embedding là từ chối chạy** (bảo vệ không gian vector).

## 5. Provider (cấu hình qua .env)

| | Embedding | LLM |
|---|---|---|
| Vai trò | Quyết định không gian vector — **chọn một lần** | Chỉ dùng cho consolidation + `/chat` — **đổi thoải mái** |
| Mặc định | `fastembed` (ONNX local, bake trong image, không cần key, không cần GPU) | `none` (model của Hermes tự chưng cất qua MCP) |
| Tuỳ chọn | `ollama`, `openai`, `nvidia` | `anthropic`, `openai`, `nvidia`, `ollama` |
| Đổi thế nào | Backup → `docker compose down -v` → đổi .env → re-ingest (meta guard chặn đổi ngầm) | Sửa `.env` + API key → `docker compose up -d` |

`fastembed` được **pin cứng phiên bản** trong requirements.txt — thư viện này
từng đổi cách pooling giữa các minor version, không pin là vector giữa hai
lần build image lệch nhau âm thầm.

## 6. Bề mặt API

### REST (`http://localhost:8800`)
| Endpoint | Chức năng |
|---|---|
| `GET /health` | Trạng thái + `last_written_at` + số điểm từng collection |
| `POST /memory/append` | Ghi 1 lượt hội thoại (hook gọi) — idempotent |
| `POST /memory/recall` | Truy hồi tổng hợp L1+L2+L3 → context_block |
| `POST /memory/facts` | Lưu facts đã chưng cất |
| `POST /memory/search` | Tìm trong L3 |
| `POST /memory/consolidate` | Chưng cất (cần LLM; `background: true` cho hook) |
| `GET /memory/pending-consolidation` | Phiên đang chờ chưng cất |
| `GET /memory/facts` · `DELETE /memory/facts/{id}` | Liệt kê / quên fact |
| `DELETE /sessions/{id}` | Xoá trọn một phiên |
| `POST /ingest/text` `/ingest/file` | Nạp tài liệu vào L4 |
| `POST /query` | Tìm trong L4 |
| `POST /chat` | Chat trực tiếp với service (cần LLM; 503 nếu `none`) |
| `GET /sessions` `/sessions/{id}/history` | Danh sách phiên / nội dung phiên |

### MCP tools (`http://localhost:8800/mcp`, Streamable HTTP)
`memory_recall` · `memory_append` · `consolidate_session` · `save_memories` ·
`list_memories` · `forget_about` · `forget_memory` · `search_history` ·
`list_sessions` · `list_projects` · `search_knowledge_base` ·
`add_to_knowledge_base` — các tool tìm kiếm/ghi đều nhận param `project` tuỳ chọn.

## 7. Tích hợp Hermes — 3 điểm chạm

`setup.sh` → `scripts/configure_hermes.py` tự động hoá toàn bộ (idempotent):

1. **`~/.hermes/config.yaml`**: 3 hooks + `hooks_auto_accept: true` + MCP server:
   - `post_llm_call` → ghi từng lượt chat vào memory (kèm project từ cwd)
   - `pre_llm_call` → auto-inject memory: recall theo user_message rồi trả
     `{"context": ...}` (contract chính thức của Hermes) — model không cần
     nhớ gọi tool nữa
   - `on_session_end` → trigger consolidation nền cho phiên vừa kết thúc
   Ngoài ra: tự mượn API key từ `~/.hermes/.env` (NVIDIA → Gemini) cho
   auto-consolidation, cài launchd backup 2:00 sáng (retention 7 bản), và
   thêm block **memory routing vào `~/.hermes/SOUL.md`** — Hermes có memory
   tool built-in (file text ~2200 ký tự) cạnh tranh với MCP tools khi user
   ra lệnh "nhớ/quên" tường minh; block này định tuyến facts dài hạn về MCP
   (Qdrant), built-in chỉ giữ định danh cốt lõi.
2. **`~/.hermes/shell-hooks-allowlist.json`**: consent cho hook (Hermes yêu cầu
   duyệt từng command; Desktop không có TTY để hỏi → phải ghi sẵn).
   ⚠️ Consent gắn với mtime của script — **sửa file hook là phải chạy lại setup.sh**.
3. **Vá bug Hermes** (`hermes_cli/main.py`): lệnh `serve` (backend Desktop) bị
   thiếu trong `_AGENT_COMMANDS` nên shell hooks không bao giờ được đăng ký cho
   chat từ Desktop (CLI hoạt động, Desktop âm thầm không). Patch thêm `"serve"`.
   ⚠️ **Update Hermes sẽ ghi đè patch — chạy lại `./setup.sh` sau mỗi lần update.**

## 8. Vận hành

- **Kiểm tra sống**: `curl localhost:8800/health` — `last_written_at` phải
  nhích sau mỗi lượt chat. Xem trực quan: `http://localhost:6333/dashboard`.
- **Chẩn đoán hook**: `hermes hooks doctor` · payload thô của mọi lần hook
  chạy nằm ở `logs/hook-debug.jsonl`.
- **Backup**: `./scripts/backup.sh` — snapshot mọi collection vào `./backups/`.
- **Dữ liệu**: named volumes `qdrant_data` (vector) + `hermes_data` (file gốc).
  `docker compose down -v` = xoá sạch làm lại.
- **Ports**: 8800 (service, giữ 8000 trống cho hindsight server của Hermes),
  6333/6334 (Qdrant), 11434 (Ollama nếu bật profile).

## 9. Cấu trúc mã nguồn

```
hermes-agent/
├── setup.sh                     # cài đặt một lệnh: Docker + cấu hình Hermes tự động
├── docker-compose.yml           # qdrant + llamaindex (+ ollama optional profile)
├── .env.example                 # cấu hình provider/collection
├── ARCHITECTURE.md              # tài liệu này
├── hooks/
│   └── post_llm_call.py         # hook ghi memory (parse đa format, debug log)
├── scripts/
│   ├── configure_hermes.py      # tự vá config.yaml + consent + bug serve + restart app
│   └── backup.sh                # snapshot Qdrant
├── logs/
│   └── hook-debug.jsonl         # payload thô mỗi lần hook chạy (chẩn đoán)
└── llamaindex-service/
    ├── Dockerfile               # bake sẵn model fastembed
    ├── requirements.txt         # fastembed pin cứng ==0.7.4
    └── app/
        ├── main.py              # FastAPI endpoints + lifespan + mount MCP
        ├── config.py            # đọc env, hằng số
        ├── providers.py         # factory embedding + LLM theo provider
        ├── qdrant_setup.py      # auto-init collections/indexes + schema guard
        ├── memory_store.py      # L2: ghi/đọc/search hội thoại (ID tất định)
        ├── memories.py          # L3: save_facts (dedup/supersede) + recall + decay
        ├── consolidation.py     # chưng cất L2→L3 (prompt + parse)
        ├── documents.py         # L4: ingest tài liệu + giữ file gốc
        ├── mcp_server.py        # 8 MCP tools (Streamable HTTP tại /mcp)
        └── runtime.py           # state chia sẻ giữa REST và MCP
```

## 10. Bộ nhớ theo dự án (project partitioning)

Memory được phân vùng theo **project trong sidebar của Hermes Desktop** —
không tách collection, mà bằng trường `project_id` (có payload index) trong
cả 3 tầng dữ liệu, đúng khuyến nghị multitenancy của Qdrant.

**Nguồn sự thật là chính Hermes.** Sidebar project của Hermes lưu trong
`~/.hermes/projects.db` (bảng `projects` + `project_folders`, mỗi project neo
một hoặc nhiều thư mục). Không có file mapping riêng phải bảo trì.

**Luồng gắn project (tự động 100%):**
```
Chat trong Hermes (bất kỳ đâu)
  → hook payload có sẵn "cwd" (thư mục phiên chat đang đứng)
  → hooks/post_llm_call.py tra projects.db: longest-prefix match
    cwd với các folder của project (project con thắng project cha,
    ví dụ /work/erp thắng /work) → slug, không khớp → "default"
  → project_id đi kèm /memory/append, lưu vào payload
```

**Luồng truy hồi:**
- `memory_recall(query, session_id)` — service tự suy project từ message đã
  lưu của chính phiên đó (không cần ai khai báo). Phiên mới tinh chưa có
  message thì truyền `project` tường minh (model của Hermes có thể gọi
  `list_projects` để biết slug).
- **Soft boost thay vì filter cứng** cho memory/hội thoại: điểm cùng project
  × `RECALL_PROJECT_BOOST` (mặc định 1.5). Ký ức dự án khác vẫn nổi lên được
  nếu thực sự liên quan — đúng giá trị của bộ nhớ xuyên dự án.
- **Tài liệu (L4) filter cứng** khi chỉ định project — tài liệu dự án A trả
  lời cho câu hỏi dự án B thường là nhiễu.

Dữ liệu cũ không có `project_id` được coi là `"default"` — không cần migrate.
Tạo project mới trong sidebar là tự nhận, không phải cấu hình gì thêm.

## 11. Định hướng mở rộng (đã trả trước chi phí)

- Mọi payload đều có `user_id` (mặc định `"local"`) và point ID tất định →
  chuyển lên server chung multi-user sau này chỉ là thêm auth + đổi giá trị,
  không phải migrate dữ liệu.
- Đổi embedding model: tạo collection mới → re-embed từ file gốc (L4) và
  content trong payload (L2/L3) → flip alias. Meta guard đảm bảo không bao
  giờ trộn hai không gian vector.
- Chưa làm (chủ đích, để đơn giản): TTL/quên tự động, hybrid search
  (dense+sparse), auth — bật khi có nhu cầu thật.
