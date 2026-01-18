# Refactoring Plan: Subprocess Daemons → Celery Tasks

## Current Architecture

The application uses **three daemon subprocesses** managed by a central **Watcher HTTP server**:

```
Watcher (HTTP Server, Port 8080)
├── Indexer Daemon      → Polls RSS feed, downloads episodes
├── Transcriber Daemon  → Calls Deepgram API for transcription
└── Segmentation Daemon → LLM segmentation + pgvector storage
```

**Communication mechanisms:**
- Filesystem markers (`segmentation_completed`, `in_progress`)
- PID files for daemon lifecycle synchronization
- Database for deduplication and data storage
- Polling loops (1-hour sleep intervals)

## Target Architecture

Replace subprocesses with **Celery tasks** orchestrated via chains/workflows:

```
Celery Beat (Scheduler)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   Celery Workers                    │
│                                                     │
│  check_rss_feed → download_metadata → download_mp3  │
│                          │                          │
│                          ▼                          │
│                   transcribe_episode                │
│                          │                          │
│                          ▼                          │
│                   segment_episode                   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
                  PostgreSQL + pgvector

HTTP API Server (FastAPI/Flask) ← Search endpoints only
```

## Benefits of Migration

1. **Event-driven** vs polling - tasks trigger immediately on completion
2. **Scalability** - spin up multiple workers for parallel processing
3. **Retry logic** - built-in retry with exponential backoff
4. **Monitoring** - Flower dashboard, task status inspection
5. **Simpler orchestration** - no PID files, no subprocess management
6. **Fault tolerance** - tasks persist in broker if worker crashes

---

## Phase 1: Refactor Existing Business Logic into Reusable Modules

### 1.1 Extract Indexer Logic

Create `src/components/indexer/operations.py`:

```python
# Extract from daemon.py into pure functions
def fetch_rss_feed(feed_url: str) -> List[EpisodeData]
def download_episode_metadata(episode: EpisodeData, storage_dir: Path) -> Path
def download_episode_mp3(episode_path: Path, mp3_url: str) -> Path
```

### 1.2 Extract Transcriber Logic

Create `src/components/indexer/transcription_operations.py`:

```python
# Extract from transcriber_daemon.py / transcriber_deepgram.py
def transcribe_episode(
    episode_path: Path,
    provider: str,
    api_key: str
) -> TranscriptionResult
```

### 1.3 Extract Segmentation Logic

The `SegmentationBuilder` is already well-factored. Ensure it can be called without daemon context:

```python
# Already exists in segmentation_builder.py
def get_segments(transcription_path: Path) -> SegmentationResult
```

---

## Phase 2: Implement Celery Tasks

### 2.1 Update `app/tasks.py`

Replace placeholder implementations with real logic:

```python
from celery import chain, group
from celery_app import app
from src.components.indexer.operations import (
    fetch_rss_feed,
    download_episode_metadata,
    download_episode_mp3
)
from src.components.indexer.transcription_operations import transcribe_episode
from src.components.segmentation.segmentation_builder import SegmentationBuilder
from src.components.segmentation.pgvector_repository import PgVectorRepository

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def check_rss_feed(self):
    """Scheduled task: check RSS feed and trigger chains for new episodes."""
    episodes = fetch_rss_feed('https://devzen.ru/feed/')
    for ep in episodes:
        if not episode_exists(ep.number):
            process_new_episode.delay(ep.dict())
    return {'new_episodes': len(episodes)}

@app.task(bind=True, max_retries=3)
def process_new_episode(self, episode_data: dict):
    """Chain: download → transcribe → segment"""
    workflow = chain(
        download_metadata_task.s(episode_data),
        download_mp3_task.s(),
        transcribe_task.s(),
        segment_task.s()
    )
    return workflow.apply_async()

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def download_metadata_task(self, episode_data: dict):
    """Download episode metadata and HTML."""
    path = download_episode_metadata(episode_data, STORAGE_DIR)
    return {**episode_data, 'episode_path': str(path)}

@app.task(bind=True, max_retries=3, default_retry_delay=300)
def download_mp3_task(self, episode_data: dict):
    """Download episode MP3 file."""
    path = download_episode_mp3(
        Path(episode_data['episode_path']),
        episode_data['mp3_link']
    )
    return {**episode_data, 'mp3_path': str(path)}

@app.task(bind=True, max_retries=2, default_retry_delay=600)
def transcribe_task(self, episode_data: dict):
    """Transcribe episode audio via Deepgram."""
    try:
        result = transcribe_episode(
            Path(episode_data['episode_path']),
            provider='deepgram',
            api_key=settings.DEEPGRAM_API_KEY
        )
        return {**episode_data, 'transcription_path': str(result.path)}
    except Exception as e:
        raise self.retry(exc=e)

@app.task(bind=True, max_retries=2, default_retry_delay=300)
def segment_task(self, episode_data: dict):
    """Segment transcription and store in pgvector."""
    builder = SegmentationBuilder(settings)
    result = builder.get_segments(Path(episode_data['transcription_path']))

    repo = PgVectorRepository(settings.db_connection)
    repo.insert_segmentation_result(episode_data['episode_number'], result)

    return {'status': 'completed', 'episode': episode_data['episode_number']}
```

### 2.2 Configure Celery Beat Schedule

Update `celery_app.py`:

```python
app.conf.beat_schedule = {
    'check-rss-feed-hourly': {
        'task': 'app.tasks.check_rss_feed',
        'schedule': crontab(minute=0),  # Every hour at :00
    },
}

# Task routing for different queue priorities
app.conf.task_routes = {
    'app.tasks.check_rss_feed': {'queue': 'scheduler'},
    'app.tasks.download_*': {'queue': 'downloads'},
    'app.tasks.transcribe_task': {'queue': 'transcription'},
    'app.tasks.segment_task': {'queue': 'segmentation'},
}

# Retry configuration
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
```

---

## Phase 3: Refactor HTTP API (Watcher)

### 3.1 Simplify Watcher to HTTP API Only

The watcher currently manages daemon lifecycles. After migration:

1. Remove `DaemonWrapper` usage
2. Remove daemon startup/shutdown logic
3. Keep only HTTP endpoints:
   - `/ping` - health check
   - `/search` - vector similarity search
   - `/v2/search` - search with episode data
   - `/episodes/{num}` - episode metadata

### 3.2 Option: Migrate to FastAPI

Consider migrating from aiohttp to FastAPI for:
- Automatic OpenAPI documentation
- Dependency injection
- Better async support
- Pydantic integration (already used)

```python
# src/api/main.py
from fastapi import FastAPI
from src.components.segmentation.pgvector_repository import PgVectorRepository

app = FastAPI()

@app.get("/search")
async def search(q: str, limit: int = 10):
    repo = PgVectorRepository(settings.db_connection)
    return repo.search(q, limit)

@app.get("/episodes/{episode_num}")
async def get_episode(episode_num: int):
    ...
```

---

## Phase 4: Update Configuration

### 4.1 Environment Variables

```bash
# .env additions
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
DEEPGRAM_API_KEY=...
SEGMENTATION_LLM_API_KEY=...
SEGMENTATION_LLM_API_URL=...
```

### 4.2 Settings Module

Create unified settings using Pydantic:

```python
# src/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    celery_broker_url: str = 'redis://localhost:6379/0'
    celery_result_backend: str = 'redis://localhost:6379/1'
    episodes_storage_dir: str = 'episodes'
    deepgram_api_key: str
    segmentation_llm_api_url: str
    segmentation_llm_api_key: str
    db_host: str = 'localhost'
    db_port: int = 5432
    db_name: str = 'podcast'
    db_user: str
    db_password: str

    class Config:
        env_file = '.env'
```

---

## Phase 5: Update Deployment

### 5.1 Systemd Services

Replace single watcher service with multiple services:

```ini
# /etc/systemd/system/podcast-celery-worker.service
[Unit]
Description=Podcast Celery Worker
After=network.target redis.service postgresql.service

[Service]
Type=simple
User=podcast
WorkingDirectory=/opt/podcast-shownotes
ExecStart=/opt/podcast-shownotes/envs/bin/celery -A celery_app worker --loglevel=info
Restart=always

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/podcast-celery-beat.service
[Unit]
Description=Podcast Celery Beat Scheduler
After=network.target redis.service

[Service]
Type=simple

User=podcast
WorkingDirectory=/opt/podcast-shownotes
ExecStart=/opt/podcast-shownotes/envs/bin/celery -A celery_app beat --loglevel=info
Restart=always

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/podcast-api.service
[Unit]
Description=Podcast HTTP API
After=network.target postgresql.service

[Service]
Type=simple
User=podcast
WorkingDirectory=/opt/podcast-shownotes
ExecStart=/opt/podcast-shownotes/envs/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

### 5.2 Docker Compose (Alternative)

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  worker:
    build: .
    command: celery -A celery_app worker --loglevel=info
    depends_on:
      - redis
      - postgres
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  beat:
    build: .
    command: celery -A celery_app beat --loglevel=info
    depends_on:
      - redis

  api:
    build: .
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8080
    ports:
      - "8080:8080"
    depends_on:
      - postgres

  flower:
    build: .
    command: celery -A celery_app flower --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis
```

---

## Phase 6: Migration Steps

### Step-by-step execution:

1. **Create operations modules** (Phase 1)
   - Extract business logic from daemons
   - Add unit tests for extracted functions

2. **Implement Celery tasks** (Phase 2)
   - Replace placeholder tasks with real implementations
   - Test task chains locally

3. **Run both systems in parallel**
   - Keep daemons running
   - Start Celery workers
   - Verify both produce same results

4. **Refactor HTTP API** (Phase 3)
   - Remove daemon management code
   - Test search endpoints

5. **Switch traffic**
   - Stop daemon-based watcher
   - Start Celery workers + API server

6. **Cleanup**
   - Remove daemon files after validation period
   - Remove `DaemonWrapper`, PID file logic

---

## Files to Create

```
app/
├── tasks.py              # Update with real implementations
├── __init__.py

src/
├── settings.py           # New: unified Pydantic settings
├── api/
│   ├── __init__.py
│   └── main.py           # New: FastAPI HTTP server
├── components/
│   └── indexer/
│       ├── operations.py           # New: extracted indexer logic
│       └── transcription_operations.py  # New: extracted transcriber logic
```

## Files to Delete (after migration)

```
src/components/indexer/daemon.py
src/components/indexer/transcriber_daemon.py
src/components/segmentation/daemon.py
src/shared/daemon_wrapper.py
```

## Files to Modify

```
src/components/indexer/watcher.py  # Remove daemon management, keep HTTP only
celery_app.py                      # Add task routing, configuration
```

---

## Testing Strategy

1. **Unit tests** for extracted operations
2. **Integration tests** for Celery task chains (use `celery.contrib.testing`)
3. **Parallel run validation** - compare outputs of both systems
4. **Load testing** - verify worker scaling behavior

---

## Rollback Plan

If issues arise:
1. Stop Celery workers and beat
2. Restart original watcher with daemons
3. Debug and fix Celery implementation
4. Re-attempt migration

Keep daemon code intact until migration is validated in production for at least 1 week.
