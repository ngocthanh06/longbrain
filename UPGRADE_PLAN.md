# Kế hoạch nâng cấp Memory Stack (cải tiến 1–7)

> Trạng thái nền: v2 + project partitioning đã chạy ổn định (2026-07-05).
> Kế hoạch chia 4 phase theo thứ tự phụ thuộc; mỗi phase deploy được độc lập.
>
> **TIẾN ĐỘ (2026-07-05 chiều):**
> - ✅ **Phase A hoàn thành** — A1 auto-consolidation (hook `on_session_end` +
>   sweep 30ph, LLM = NVIDIA deepseek-v4-pro qua OpenAILike vì adapter
>   `llama-index-llms-nvidia` không nhận model mới; đã thêm provider `gemini`
>   dự phòng), A2 auto-inject (`pre_llm_call` trả `{"context": ...}`,
>   recent_turns=0 tránh trùng history), A3 quản lý memory (list/forget REST +
>   MCP, DELETE session). Key NVIDIA tự sync từ ~/.hermes/.env.
> - ✅ **Phase B hoàn thành** — backup launchd 2:00 sáng, retention 7 bản,
>   log logs/backup.log (fix BSD head bằng awk).
> - ⬜ Phase C (bge-m3 + hybrid) — chưa làm, cần benchmark trước.
> - ⬜ Phase D (auto-ingest docs/) — chưa làm.

## Phát hiện nền tảng (đã xác minh trong source Hermes)

Hermes hỗ trợ nhiều hook event hơn `post_llm_call` đang dùng:

| Event | Dùng cho | Cơ chế |
|---|---|---|
| `pre_llm_call` | **#2 auto-inject memory** | stdout `{"context": "..."}` được Hermes inject chính thức vào lượt chat |
| `on_session_end` | **#1 auto-consolidation** | extra có `completed`, `model`, `platform` — bắn đúng lúc phiên kết thúc |
| `on_session_start` | tuỳ chọn: warm-up recall | extra có `model`, `platform` |

---

## Phase A — Hoàn thiện vòng đời memory (làm trước, giá trị cao nhất)

### A1. Tự động consolidation (#1)

**Vấn đề:** `hermes_memories` chỉ có dữ liệu khi ai đó chủ động gọi tool — thực tế sẽ không ai gọi.

**Thiết kế — 2 tầng trigger:**
1. **Hook `on_session_end`** (mới: `hooks/on_session_end.py`): phiên kết thúc
   (`completed=true`) → `POST /memory/consolidate {session_id}` (fire-and-forget,
   timeout ngắn, best-effort như hook hiện tại).
2. **Sweep định kỳ trong service** (mới: `app/scheduler.py`, asyncio task trong
   lifespan): mỗi `CONSOLIDATION_INTERVAL` (mặc định 30 phút) quét phiên có
   ≥ `CONSOLIDATION_MIN_TURNS` (4) turn chưa chưng cất VÀ im lặng >
   `CONSOLIDATION_IDLE_SECONDS` (15 phút) → consolidate. Bắt phiên mà hook bỏ lỡ
   (crash, tắt máy giữa chừng).

**Điều kiện:** service cần LLM (`LLM_PROVIDER != none`). Khuyến nghị:
`anthropic` + `claude-haiku` (rẻ, đủ cho extraction) hoặc `ollama` profile.
Khi `none`: scheduler chỉ log cảnh báo + expose `GET /memory/pending-consolidation`
để Hermes-side xử lý thủ công.

**Việc:** `app/scheduler.py` (mới ~80 dòng) · `hooks/on_session_end.py` (mới ~40 dòng)
· `config.py` +4 env · `main.py` lifespan +5 dòng · `configure_hermes.py` đăng ký
hook mới + consent · test: tạo phiên giả → chờ sweep → fact xuất hiện.

**Rủi ro:** LLM extract chất lượng kém với model quá nhỏ → prompt đã có sẵn,
test với 2-3 model trước khi chốt khuyến nghị. Chi phí API: ~1 call/phiên, không đáng kể.

### A2. Tự động inject memory vào lượt chat (#2)

**Vấn đề:** recall phụ thuộc model Hermes nhớ gọi tool — không đáng tin.

**Thiết kế:** hook mới `hooks/pre_llm_call.py`:
```
stdin: {session_id, cwd, extra: {user_message?, ...}}
  → POST /memory/recall {query: user_message, session_id, project: resolve(cwd)}
  → stdout: {"context": context_block}   ← Hermes inject chính thức
```
- **Bước 0 (discovery):** payload thật của `pre_llm_call` chưa được xác minh
  (docs không liệt kê extra keys) → bật debug-dump 1 ngày như đã làm với
  post_llm_call, chốt schema rồi mới code phần parse.
- **Kiểm soát latency** (hook chạy đồng bộ trước mỗi LLM call): timeout 3s,
  fail → trả rỗng; chỉ inject khi recall có kết quả thật (context_block ≠ rỗng);
  option `RECALL_INJECT_EVERY_N_TURNS` (mặc định: mọi turn, đo thực tế rồi chỉnh).
- Ước tính chi phí mỗi turn: 1 embed + 3 search ≈ 100–300ms trên máy M-series.

**Việc:** `hooks/pre_llm_call.py` (mới ~70 dòng, tái dùng `resolve_project`)
· `configure_hermes.py` đăng ký + consent · đo latency thực tế · test A/B:
hỏi lại thông tin phiên cũ mà không gọi tool — Hermes phải tự biết.

### A3. Công cụ quản lý memory (#3)

**Vấn đề:** memory nhớ sai thì không có cách sửa ngoài mò Qdrant dashboard.

**Thiết kế:**
- REST: `GET /memory/facts?project=&type=&limit=` (liệt kê, kèm id)
  · `DELETE /memory/facts/{id}` · `DELETE /sessions/{session_id}` (xoá cả phiên)
- MCP: `list_memories(project?)` · `forget_memory(id)` ·
  `forget_about(query)` — search top-5, trả danh sách kèm id để model chọn xoá
  (2 bước, tránh xoá nhầm theo similarity).
- Xoá fact = xoá cứng point (khác supersede — supersede là thay thế tự nhiên,
  forget là lệnh của người dùng).

**Việc:** `memories.py` +3 hàm · `main.py` +3 endpoint · `mcp_server.py` +3 tool
· test: save → forget_about → recall không còn thấy.

**Phase A tổng:** ~1 buổi làm việc. Không migration, không đổi schema.

---

## Phase B — Vận hành tự động (#7, làm cùng Phase A được)

### B1. Backup tự động

**Thiết kế:** launchd agent trên macOS (host mới truy cập được cả 2 volume qua API):
- `scripts/backup.sh` nâng cấp: thêm retention (giữ `BACKUP_KEEP=7` bản mới nhất),
  log ra `logs/backup.log`, snapshot cả 3 collection + copy `.env`.
- `scripts/com.hermes.memory-backup.plist` — chạy 2:00 sáng hàng ngày.
- `configure_hermes.py` (hoặc setup.sh) cài plist vào `~/Library/LaunchAgents`
  + `launchctl load`. Idempotent.

**Việc:** ~30 phút. Test: `launchctl start` thủ công → kiểm tra `backups/`.

---

## Phase C — Chất lượng truy hồi (#4 + #5, LÀM CHUNG một lần migration)

> Gộp 2 cải tiến vì cả hai đều yêu cầu re-create collection + re-embed.
> Làm riêng = trả giá migration 2 lần.

### C1. Nâng embedding lên BAAI/bge-m3 (#5)

- Đổi default `EMBED_MODEL=BAAI/bge-m3` (1024-dim, multilingual mạnh —
  cải thiện rõ cho tiếng Việt). Image nặng thêm ~2.2GB — chấp nhận, vẫn bake.
- Máy yếu giữ được MiniLM qua `.env` — mọi guard đã có sẵn.

### C2. Hybrid search dense + sparse BM25 (#4)

- **L4 documents:** `QdrantVectorStore(enable_hybrid=True, fastembed_sparse_model="Qdrant/bm25")`
  — LlamaIndex lo toàn bộ (named vectors + RRF fusion). Yêu cầu collection mới.
- **L2/L3 (collection tự quản):** thêm named sparse vector khi ghi
  (`fastembed SparseTextEmbedding`), search chuyển sang Query API
  `prefetch dense + sparse → fusion RRF`. Đây là phần code lớn nhất của cả kế hoạch
  (~150 dòng thay đổi trong `memory_store.py`/`memories.py`).

### C3. Script migration `scripts/reembed.py`

```
1. Backup tự động trước khi chạy
2. Tạo collections *_new với config mới (dense 1024 + sparse)
3. Re-embed: L2/L3 từ payload text (đủ 100%), L4 từ file gốc /data/documents
4. Verify số điểm khớp → đổi tên (delete cũ, rename mới) → cập nhật hermes_meta
5. Rollback: restore từ snapshot bước 1
```

**Phase C tổng:** ~1-2 buổi. Rủi ro chính: tương thích version
`llama-index-vector-stores-qdrant` 0.4.x với hybrid — kiểm tra sớm, nếu vướng
thì nâng minor version có kiểm soát. Điểm quyết định trước khi làm: đo chất
lượng bge-m3 vs MiniLM trên chính dữ liệu tiếng Việt của bạn (script so sánh
recall trên 10-20 câu hỏi mẫu — nửa buổi, tránh migration vô ích).

---

## Phase D — Auto-ingest tài liệu theo project (#6)

**Vấn đề:** muốn tài liệu vào knowledge base phải curl thủ công.

**Thiết kế:** watcher chạy trên **host** (container không thấy thư mục người dùng):
- `scripts/ingest_watcher.py`: đọc danh sách thư mục project từ
  `~/.hermes/projects.db` (tái dùng resolver) → theo dõi thư mục con `docs/`
  của mỗi project (opt-in theo thư mục con, tránh ingest nhầm cả repo code)
  → file mới/đổi (`.pdf .md .txt .docx`) → `POST /ingest/file` kèm đúng `project_id`.
- Dedup: service đã content-address file gốc (sha) — watcher chỉ cần gửi,
  trùng thì service bỏ qua (cần thêm check sha trước khi re-index ở service).
- Chạy bằng launchd agent (giống backup), poll 60s — không cần lib `watchdog`,
  giảm dependency.

**Việc:** `scripts/ingest_watcher.py` (~120 dòng) · check-sha trong `documents.py`
· plist + cài đặt trong setup.sh · README hướng dẫn quy ước thư mục `docs/`.
~1 buổi. Quyết định cần chốt trước: quy ước thư mục nào được watch
(đề xuất: `<project_folder>/docs/`).

---

## Tổng hợp thứ tự thực hiện

| Bước | Nội dung | Công sức | Phụ thuộc |
|---|---|---|---|
| A1 | Auto-consolidation (hook session_end + sweep) | nửa buổi | cần chốt LLM provider |
| A2 | Auto-inject recall (pre_llm_call) | nửa buổi | discovery payload trước |
| A3 | Quản lý memory (list/forget) | 2-3 giờ | — |
| B1 | Backup tự động (launchd) | 30 phút | — |
| C1-3 | bge-m3 + hybrid + migration | 1-2 buổi | benchmark trước khi migrate |
| D | Auto-ingest watcher | 1 buổi | chốt quy ước thư mục docs/ |

**Câu hỏi cần chốt trước khi code:**
1. A1: dùng LLM nào cho consolidation? (đề xuất: `anthropic` + Haiku nếu có key,
   không thì `ollama` profile + qwen3)
2. D: đồng ý quy ước watch `<thư mục project>/docs/`?
