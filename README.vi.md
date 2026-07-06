# Hermes Agent — Long-term Memory stack (LlamaIndex + Qdrant)

> 🇬🇧 English version: [README.md](README.md)

Memory backend đóng gói bằng Docker cho Hermes Desktop: mỗi người dùng chạy
một stack độc lập trên máy của mình, dữ liệu và bộ nhớ hoàn toàn riêng tư.
Mặc định **không cần API key, không cần Ollama, không cần Python trên host**.

## Kiến trúc

```
Hermes Desktop (host — LLM chạy ở đây)
   │  pre_llm_call  ──► tự tiêm memory liên quan vào mỗi lượt chat
   │  post_llm_call ──► tự ghi từng lượt chat (kèm project từ sidebar)
   │  on_session_end ─► tự chưng cất phiên thành facts dài hạn
   │  on_session_start ► quét bù phiên chưa chưng cất khi mở Desktop
   │  MCP (Streamable HTTP) ──► http://localhost:8800/mcp
   ▼
LlamaIndex service (Docker, 127.0.0.1:8800)
   ├── L1 Working memory   — ChatMemoryBuffer dựng lại theo session
   ├── L2 Episodic memory  — hermes_chat_history (từng lượt hội thoại,
   │                          tìm được theo ngữ nghĩa lẫn theo phiên)
   ├── L3 Semantic memory  — hermes_memories (fact/preference/decision/task
   │                          do consolidation chưng cất, có dedup/supersede)
   ├── L4 Knowledge base   — hermes_documents (RAG tài liệu)
   ├── Embedding: fastembed (ONNX local, bake sẵn trong image)
   └── LLM (cho consolidation): none | anthropic | openai | nvidia | gemini | ollama
   ▼
Qdrant (Docker, 127.0.0.1:6333) — named volume `qdrant_data`
```

![alt text](image.png)

Vòng đời memory chạy **tự động hoàn toàn**: ghi → tự nhắc lại → chưng cất →
quên có kiểm soát (tool `forget_about`) → backup đêm (2:00, giữ 7 bản).

## Cài đặt (3 bước)

1. Cài [Docker Desktop](https://docs.docker.com/get-docker/).
2. Cài Hermes Desktop.
3. Trong thư mục này chạy:

```bash
./setup.sh
```

**Không còn bước thủ công nào.** Script tự làm trọn: tạo `.env` → build &
khởi động containers → chờ healthcheck → đăng ký 4 hooks + consent vào
`~/.hermes/` → vá bug `serve` của Hermes (Desktop không đăng ký hook nếu
thiếu) → mượn API key sẵn có (NVIDIA/Gemini) cho auto-consolidation → cài
backup đêm → thêm định tuyến memory vào `~/.hermes/SOUL.md` (lệnh "nhớ/quên"
tường minh đi vào stack này thay vì built-in store nhỏ của Hermes) → restart
Hermes Desktop. Chạy lại bao nhiêu lần cũng an toàn (idempotent).

Kiểm tra sau vài lượt chat: `curl localhost:8800/health` — trường
`last_written_at` phải cập nhật sau mỗi lượt.

## Trình duyệt bộ nhớ (`http://localhost:8800/ui`)

Một trang tự chứa để khám phá và quản lý memory đã lưu — không cần thêm
container, không tải asset ngoài, có theme sáng/tối:

- **Graph view**: memory hiển thị thành đồ thị force-directed trên nền lưới
  chấm bám theo camera. Cạnh nối là độ tương đồng vector thật (đậm hơn =
  gần nghĩa hơn); màu = project, **hình dạng node = loại** (● fact,
  ▲ preference, ◆ decision, ■ task — bấm nút `?` để xem chú giải),
  kích thước = độ quan trọng, viền đứt = đã bị supersede. Rê chuột làm nổi
  vùng lân cận của node; click mở panel chi tiết kèm hiệu ứng gợn sóng.
  Kéo thả, pan, zoom mượt bằng con lăn, nút `Fit`, và click tiêu đề để
  reset khung nhìn.
- **Tìm kiếm Spotlight (⌘K)**: tìm ngữ nghĩa trực tiếp, có lưu truy vấn
  gần đây — kết quả khớp được làm nổi trên đồ thị và camera lướt tới kết
  quả tốt nhất.
- **Bộ lọc**: chip project và chip loại (click để solo/bật-tắt), công tắc
  hiện superseded, cùng điều khiển cạnh nối — thanh trượt độ tương đồng
  tối thiểu và công tắc "chỉ cùng project".
- **Panel chi tiết**: toàn văn, metadata, các memory liên quan (theo độ
  tương đồng, click để nhảy tới), và transcript của phiên nguồn render
  dạng markdown.
- **Chỉnh sửa** (qua modal chọn project — không phải gõ tay): chuyển một
  memory, gắn lại cả phiên (turns + facts chuyển theo, các lượt chat sau
  đi theo nhờ stickiness), gắn lại hàng loạt bằng đa chọn ⇧click
  ("Select linked" mở rộng ra trọn cụm liên thông), hoặc đổi tên project
  ở mọi nơi (nút ✎ trên chip project).
- **List view** (cùng bộ lọc, dạng bảng) và **xuất PNG** đồ thị.

Trang này cố ý KHÔNG có chức năng xoá — hãy quên qua Hermes
(`forget_about`) hoặc REST API, nơi có bước xác nhận.

## Chọn provider (.env)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `EMBED_PROVIDER` | `fastembed` | `fastembed` \| `ollama` \| `openai` \| `nvidia` |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Model embedding (đa ngôn ngữ, chạy CPU) |
| `LLM_PROVIDER` | `none`* | `none` \| `anthropic` \| `openai` \| `nvidia` \| `gemini` \| `ollama` |
| `LLM_MODEL` | theo provider | VD `models/gemini-2.5-flash`, `claude-sonnet-5` |
| `*_API_KEY` | — | `ANTHROPIC` / `OPENAI` / `NVIDIA` / `GOOGLE` — setup.sh tự mượn key có sẵn trong `~/.hermes/.env` khi provider đang là `none` |
| `HERMES_USER_ID` | `local` | Định danh trong payload (để tương lai lên server chung không phải migrate) |

- **LLM đổi thoải mái** — chỉ dùng cho consolidation và `/chat`.
  Với `none`, chính model của Hermes đảm nhận consolidation qua MCP tool
  `consolidate_session`.
- **Embedding chọn một lần** — đổi model là đổi không gian vector. Service
  ghi model + dimension vào collection meta và **từ chối khởi động** nếu
  config lệch với dữ liệu trên đĩa. Muốn đổi thật: backup → `docker compose
  down -v` → đổi `.env` → chạy lại và re-ingest, hoặc re-embed vào
  collection mới.
- Ollama local (tuỳ chọn): `docker compose --profile ollama up -d`, rồi đặt
  `LLM_PROVIDER=ollama` và `OLLAMA_BASE_URL=http://ollama:11434`.

## MCP tools (đăng ký tại `http://localhost:8800/mcp`)

| Tool | Chức năng |
|---|---|
| `memory_recall(query, session_id?, project?)` | Gộp trí nhớ liên quan (facts + hội thoại cũ + lượt gần nhất) thành một context block |
| `memory_append(session_id, user_message, assistant_response)` | Ghi một lượt hội thoại (idempotent) |
| `consolidate_session(session_id)` | Chưng cất phiên thành facts (server-side nếu có LLM, ngược lại trả transcript + hướng dẫn cho model của Hermes) |
| `save_memories(facts, session_id?, project?)` | Lưu facts đã chưng cất (tự dedup/supersede) |
| `search_history(query, top_k?, project?)` | Tìm ngữ nghĩa trên toàn bộ hội thoại cũ |
| `list_memories(project?)` | Liệt kê facts đã lưu (kèm id) |
| `forget_about(query)` → `forget_memory(id, confirm=true)` | Quên có kiểm soát: liệt kê ứng viên trước, xoá theo id — từ chối nếu thiếu `confirm=true` (chỉ đặt sau khi người dùng đã đồng ý) |
| `forget_session(session_id)` | Xoá trọn lịch sử một phiên |
| `forget_everything(confirm="DELETE ALL")` | Reset toàn bộ memory — bắt buộc đúng chuỗi xác nhận |
| `list_sessions()` / `list_projects()` | Liệt kê phiên / dự án đang có memory |
| `search_knowledge_base(query, top_k?, project?)` | Tìm trong tài liệu đã ingest |
| `add_to_knowledge_base(text, source?, project?)` | Thêm text vào knowledge base |

Các tool nhận `project` (slug dự án trong sidebar Hermes) để khoanh vùng tìm kiếm.

## REST API

```bash
# Trạng thái + kiểm tra memory có đang được ghi không (last_written_at)
curl localhost:8800/health

# Nạp tài liệu
curl -X POST localhost:8800/ingest/text -H 'Content-Type: application/json' \
  -d '{"text": "Nội dung...", "metadata": {"source": "faq.md"}}'
curl -X POST localhost:8800/ingest/file -F "file=@tai-lieu.pdf"

# Truy vấn knowledge base
curl -X POST localhost:8800/query -H 'Content-Type: application/json' \
  -d '{"query": "..."}'

# Memory
curl -X POST localhost:8800/memory/append -H 'Content-Type: application/json' \
  -d '{"session_id": "s1", "user_message": "...", "assistant_response": "..."}'
curl -X POST localhost:8800/memory/recall -H 'Content-Type: application/json' \
  -d '{"query": "dự án hermes dùng vector db gì?", "session_id": "s1"}'
curl -X POST localhost:8800/memory/consolidate -H 'Content-Type: application/json' \
  -d '{"session_id": "s1"}'          # cần LLM_PROVIDER != none
curl -X POST localhost:8800/memory/search -H 'Content-Type: application/json' \
  -d '{"query": "..."}'
curl "localhost:8800/memory/facts?project=erp"      # liệt kê facts
curl -X DELETE localhost:8800/memory/facts/<id>     # quên một fact
curl -X DELETE "localhost:8800/memory/all?confirm=DELETE%20ALL"  # reset toàn bộ

# Đồ thị memory (nodes + cạnh tương đồng, cấp dữ liệu cho trang /ui)
curl "localhost:8800/memory/graph?include_superseded=false&min_similarity=0.35"

# Chuyển máy (export/import — xem mục riêng bên dưới)
curl -o bundle.json localhost:8800/memory/export
curl -X POST localhost:8800/memory/import -H 'Content-Type: application/json' \
  --data-binary @bundle.json

# Gắn lại project (chỉnh sửa; slug chỉ gồm chữ thường [a-z0-9_-])
curl -X PATCH localhost:8800/memory/facts/<id> -H 'Content-Type: application/json' \
  -d '{"project_id": "erp"}'                       # chuyển một fact
curl -X PATCH localhost:8800/memory/facts -H 'Content-Type: application/json' \
  -d '{"ids": ["<id1>", "<id2>"], "project_id": "erp"}'   # chuyển hàng loạt
curl -X PATCH localhost:8800/sessions/<id>/project -H 'Content-Type: application/json' \
  -d '{"project_id": "erp"}'    # cả phiên: turns + facts; lượt chat sau đi theo
curl -X PATCH localhost:8800/memory/projects/<slug> -H 'Content-Type: application/json' \
  -d '{"project_id": "ten-moi"}'                   # đổi tên ở mọi nơi

# Sessions & projects
curl localhost:8800/sessions
curl localhost:8800/sessions/s1/history
curl -X DELETE localhost:8800/sessions/s1
curl localhost:8800/projects
```

## Bộ nhớ theo dự án

Memory tự phân vùng theo **project trong sidebar Hermes Desktop**: hook đọc
thư mục phiên chat (`cwd`) → tra `~/.hermes/projects.db` → gắn `project_id`
vào mọi bản ghi. Khi `cwd` không khớp thư mục của project nào, hook dùng
project đang được **chọn trong sidebar** — nhờ vậy project chỉ-để-chat
(không gắn thư mục) vẫn hoạt động bình thường. Truy hồi ưu tiên ký ức cùng
dự án (boost ×1.5) nhưng vẫn thấy được ký ức dự án khác khi thực sự liên
quan; tài liệu thì lọc cứng theo dự án. Tạo project mới trong sidebar là tự
nhận — không cần cấu hình.
Xem chi tiết: [ARCHITECTURE.md](ARCHITECTURE.md).

## Backup

Tự động chạy **2:00 sáng hàng ngày** (launchd; nhờ `RunAtLoad` còn chạy
thêm một lần mỗi khi boot/đăng nhập, phòng khi máy tắt lúc 2:00; giữ 7 bản
mới nhất; log tại `logs/backup.log`) — setup.sh đã cài. Chạy tay khi cần:

```bash
./scripts/backup.sh    # snapshot mọi collection hermes_* vào ./backups/
```

## Chuyển sang máy khác (export / import)

Snapshot hàng đêm là **dữ liệu nhị phân, gắn chặt với embedding model** —
chỉ khôi phục được đúng máy cũ, không mang bộ nhớ sang một bản cài mới
(có thể chạy model khác) được. Việc đó dùng bundle chuyển máy ở mức văn bản:

```bash
# máy cũ
./scripts/memory_transfer.sh export            # -> backups/memory-export-<stamp>.json

# máy mới (sau khi chạy setup.sh, service đang chạy)
./scripts/memory_transfer.sh import memory-export-<stamp>.json
```

Bundle chỉ chứa phần văn bản (facts, lượt chat, chunk tài liệu) — không chứa
vector. Import sẽ **re-embed toàn bộ bằng model hiện tại**, giữ nguyên
timestamps / liên kết supersede / cờ consolidated (nhờ đó suy giảm theo thời
gian và nguồn gốc fact vẫn hoạt động, phiên đã import không bị chưng cất
lại), và bỏ qua bản ghi đã tồn tại — chạy lại lần nữa vẫn an toàn.

## Cấu trúc repository

```
hermes-agent/
├── setup.sh                 # cài đặt một lệnh (Docker + cấu hình Hermes tự động)
├── docker-compose.yml       # qdrant + llamaindex (+ ollama optional profile)
├── .env.example
├── ARCHITECTURE.md          # tài liệu kiến trúc chi tiết
├── UPGRADE_PLAN.md          # lộ trình nâng cấp + tiến độ
├── hooks/
│   ├── post_llm_call.py     # ghi từng lượt chat (kèm project từ sidebar)
│   ├── pre_llm_call.py      # tự tiêm memory vào mỗi lượt chat
│   ├── on_session_end.py    # trigger chưng cất khi phiên kết thúc
│   └── on_session_start.py  # quét bù khi mở Desktop
├── scripts/
│   ├── configure_hermes.py  # tự cấu hình Hermes (hooks + consent + vá serve + key + backup)
│   ├── backup.sh            # snapshot Qdrant (launchd gọi hàng đêm)
│   └── memory_transfer.sh   # export/import mức văn bản để chuyển máy
└── llamaindex-service/      # memory service (FastAPI + LlamaIndex + MCP)
    └── tests/               # bộ test pytest (chạy trong container, xem bên dưới)
```

## Kiểm thử

```bash
docker compose run --rm --no-deps --entrypoint sh \
  -v "$PWD:/repo" llamaindex \
  -c "pip install -q 'pytest>=8,<9' && cd /repo/llamaindex-service && python -m pytest tests -q"
```

Bao phủ: tính idempotent của point-id, dedup/supersede facts, bộ lọc
min-score của recall (chạy với Qdrant in-process), parse output LLM (phân
biệt lỗi parse với `[]` chủ đích), cắt ngắn transcript, và phần trích
payload của hooks + phân giải cwd→project (longest prefix, project đã
archive, symlink).

## Lưu ý vận hành

- **Sau mỗi lần update Hermes: chạy lại `./setup.sh`** — bản update ghi đè
  patch `serve` (Desktop backend không đăng ký hook nếu thiếu patch này).
- **Sửa file nào trong `hooks/` xong cũng chạy lại `./setup.sh`** — consent
  hook gắn với mtime của script.
- **Muốn xoá memory: nói với Hermes ("quên chuyện X đi") hoặc dùng API**,
  đừng xoá trong Qdrant dashboard — dashboard có toàn quyền ghi/xoá không
  xác nhận, dễ tạo cảm giác "hệ thống tự mất dữ liệu".
