# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Podcast shownotes processing system for DevZen podcast (devzen.ru). Handles RSS feed monitoring, episode downloading, audio transcription (via Deepgram), semantic segmentation, vector storage (pgvector), and search with a Telegram bot interface.

## Development Commands

```bash
# Environment setup (requires conda/mamba)
make env-create        # Create conda environment in ./envs
make env-update        # Update environment
make env-remove        # Remove environment

# Testing
make test              # Run unit and contract tests
make test-k-FILTER     # Run tests matching FILTER (e.g., make test-k-segmentation)
make test-vvsx         # Run with verbose output and stop on first failure
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-failed       # Rerun failed tests

# Code quality
make format            # Format code with brunette (black fork)
make lint              # Run flake8 and mypy
make flake8            # Linting only
make mypy              # Type checking only

# Services
make run-indexer-deepgram  # Run indexer with Deepgram transcription
make lab               # Start Jupyter Lab

# Celery (requires Redis)
celery -A src.app.app worker --loglevel=info -Q scheduler,downloads,transcription,segmentation  # Run worker
celery -A src.app.app beat --loglevel=info    # Run scheduler
```

## Architecture

### Core Pipeline
1. **RSS Monitoring** (`app/tasks.py`) - Celery Beat checks devzen.ru/feed for new episodes
2. **Downloading** - Episode metadata and MP3 files saved to storage directory
3. **Transcription** (`src/components/indexer/transcriber_deepgram.py`) - Deepgram nova-2 model for Russian audio
4. **Segmentation** (`src/components/segmentation/`) - LLM-based and semantic text segmentation
5. **Vector Storage** (`pgvector_repository.py`) - PostgreSQL with pgvector for embeddings
6. **Search** - Vector similarity search via watcher.py HTTP API
7. **Telegram Bot** (`src/tgbot/bot.py`) - Inline search interface

### Key Components
- `src/components/indexer/watcher.py` - Main HTTP server managing daemons for indexing, transcription, and segmentation
- `src/components/segmentation/segmentation_builder.py` - Coordinates segmentation with `SegmentationResult` containing segments and sentences
- `src/components/segmentation/embedding_builder.py` - Uses `deepvk/USER-bge-m3` model for embeddings
- `src/components/segmentation/llm_segmetation.py` - OpenAI-compatible API for text segmentation
- `src/shared/args.py` - Pydantic-based CLI argument parsing (`IndexerServerArgs`, `DaemonArgs`)

### Data Flow
Episodes stored in directories named by episode number (e.g., `episodes/472/`):
- `episode.mp3` - Audio file
- `episode.json` - Episode data
- `metadata.json` - Parsed metadata with shownotes
- `transcription-*.json` - Transcription results
- `segmentation_completed` / `segmentation_in_progress` - Status markers

### Database Schema
- `episodes` - Episode metadata with shownotes embedding
- `speakers` - Speaker information
- `segments` - Text segments with embeddings for similarity search

## External Dependencies
- PostgreSQL with pgvector extension
- Redis (for Celery broker/backend)
- Deepgram API (transcription)
- OpenAI-compatible API (segmentation LLM)

## Code Style
- Single quotes, 79 char line length (brunette formatter)
- Type hints throughout
- Pydantic models for configuration
