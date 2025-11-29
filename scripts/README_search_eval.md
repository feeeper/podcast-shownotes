# Search Evaluation Dataset Builder

This script generates search queries for podcast segments to create evaluation datasets for search functionality.

## Features

- Connects to PostgreSQL database to fetch podcast segments
- Selects one random segment per episode (excluding first and last segments)
- Sends parallel requests to OpenAI API to generate search queries
- Saves results to JSON files organized by episode

## Requirements

- Python 3.12+
- PostgreSQL database with `segments` and `episodes` tables
- OpenAI API key
- Required Python packages (already included in the project's `env.yml`):
  - `openai==1.65.4`
  - `psycopg2`
  - `httpx` (included with openai package)

## Usage

### Basic Usage

```bash
python scripts/build_search_eval_dataset.py \
    --dbname your_database_name \
    --openai-api-key your_openai_api_key
```

**Note**: The script processes one random segment per episode in batches. Use `--batch-size` to control how many requests are sent in parallel.

### Full Example

```bash
python scripts/build_search_eval_dataset.py \
    --dbhost localhost \
    --dbport 5432 \
    --dbname podcast_shownotes \
    --dbuser postgres \
    --dbpassword your_password \
    --openai-api-key sk-your-openai-api-key \
    --openai-base-url https://api.openai.com/v1 \
    --batch-size 5 \
    --batch-delay 15 \
    --output-dir ./search_eval_results
```

### Command Line Arguments

#### Database Connection
- `--dbhost`: Database host (default: localhost)
- `--dbport`: Database port (default: 5432)
- `--dbname`: Database name (**required**)
- `--dbuser`: Database user (default: postgres)
- `--dbpassword`: Database password (default: postgres)

#### OpenAI Configuration
- `--openai-api-key`: OpenAI API key (**required**)
- `--openai-base-url`: OpenAI API base URL (default: https://api.openai.com/v1)

#### Script Configuration
- `--batch-size`: Number of requests to send in parallel per batch (default: 5)
- `--batch-delay`: Delay in seconds between batches (default: 15)
- `--output-dir`: Output directory for results (default: ./search_eval_results)

## Database Schema Requirements

The script expects the following tables:

### `episodes` table
```sql
CREATE TABLE episodes (
    id UUID PRIMARY KEY,
    episode_number INTEGER,
    -- other fields...
);
```

### `segments` table
```sql
CREATE TABLE segments (
    id UUID PRIMARY KEY,
    episode_id UUID REFERENCES episodes(id),
    text TEXT,
    segment_number INTEGER,
    -- other fields...
);
```

## Output Format

The script creates JSON files named `{episode_id}.json` in the output directory. Each file contains an array of segment results:

```json
[
  {
    "segment_id": "uuid-string",
    "episode_id": "uuid-string", 
    "episode_number": 123,
    "segment_number": 5,
    "text": "Original segment text...",
    "generated_query": {
      "query": "search query here"
    }
  }
]
```

## OpenAI API Configuration

The script uses the following OpenAI API settings:
- **Model**: `gpt-4o-mini`
- **Temperature**: 0.1 (for consistent results)
- **Max tokens**: 100
- **Response format**: JSON object

The system prompt instructs the AI to:
- Generate search queries of 5 words or fewer
- Focus on the main idea of the input text
- Return results in a specific JSON format
- Handle cases where a suitable query cannot be formed

## Error Handling

- Database connection errors are handled gracefully
- OpenAI API errors are captured and included in the results
- The script continues processing even if individual requests fail
- Results are saved even if some segments fail to process

## Performance

- Uses `asyncio` for parallel API requests within batches
- Processes segments in configurable batches (default: 5 requests per batch)
- Implements throttling with configurable delays between batches (default: 15 seconds)
- Database queries are optimized to exclude first/last segments efficiently
- Results are grouped by episode for efficient file output

## Example Output

```
Connecting to database: localhost:5432/podcast_shownotes
Batch size: 5, Delay between batches: 15s
Output directory: ./search_eval_results
Fetching random segments from database...
Found 10 segments to process (one per episode)
Generating search queries using OpenAI API...
Processing batch 1/2 (5 segments)...
Waiting 15 seconds before next batch...
Processing batch 2/2 (5 segments)...
Saved 6 results to ./search_eval_results/123.json
Saved 4 results to ./search_eval_results/124.json
Search evaluation dataset generation completed!
```
