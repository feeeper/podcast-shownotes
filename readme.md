# Podcast Shownotes

Podcast shownotes processing system for DevZen podcast (devzen.ru). Handles RSS feed monitoring, episode downloading, audio transcription (via Deepgram), semantic segmentation, vector storage (pgvector), and search with a Telegram bot interface.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/)
- [Docker](https://docs.docker.com/get-docker/) (for Redis and PostgreSQL)
- API keys: Deepgram, OpenAI-compatible LLM API

## Getting Started

### 1. Install Conda/Mamba

Install Miniconda:
```bash
# Download and install Miniconda (Linux)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or use the official installer links:
# https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links
```

Install Mamba (recommended for faster dependency resolution):
```bash
conda install mamba -n base -c conda-forge
```

### 2. Create Environment

```bash
make env-create
```

This creates a conda environment in `./envs` directory.

To update the environment after changing `env.yml`:
```bash
make env-update
```

To activate the environment manually:
```bash
conda activate ./envs
```

### 3. Start PostgreSQL with pgvector

Run PostgreSQL with the pgvector extension:
```bash
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=podcast_shownotes \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

To start an existing container:
```bash
docker start postgres-pgvector
```

### 4. Start Redis

Redis is used as the Celery broker and backend:
```bash
# First time - create and run container
make run-redis

# Or manually:
docker run -d --name redis -p 6379:6379 redis
```

To start an existing container:
```bash
make start-redis

# Or manually:
docker start redis
```

### 5. Set Environment Variables

Create a `.env` file or export the following variables:
```bash
export DEEPGRAM_API_KEY=your_deepgram_api_key
export SEGMENTATION_API_KEY=your_openai_compatible_api_key
export SEGMENTATION_API_URL=https://api.openai.com/v1  # or compatible endpoint
```

### 6. Database Setup

Initialize the database schema (creates database, tables, and pgvector extension):
```bash
make db-init
```

Run migrations (adds FTS indexes, pg_trgm extension, search history tables):
```bash
make db-migrate
```

## Running the Application

### Celery Worker

Processes tasks for downloading, transcription, and segmentation:
```bash
make run-worker

# Or manually:
celery -A src.app.app worker --loglevel=info -Q scheduler,downloads,transcription,segmentation
```

### Celery Beat

Scheduler that periodically checks the RSS feed for new episodes:
```bash
make run-beat

# Or manually:
celery -A src.app.app beat --loglevel=info
```

### Trigger RSS Check Manually

```bash
make trigger-check-rss-task
```

## Development

### Running Tests

```bash
make test              # Run unit and contract tests
make test-unit         # Unit tests only
make test-integration  # Integration tests (requires external services)
make test-k-FILTER     # Run tests matching FILTER
```

### Code Quality

```bash
make format            # Format code with brunette
make lint              # Run flake8 and mypy
```

### Jupyter Lab

```bash
make lab
```
