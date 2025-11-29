#!/bin/bash

# Example usage of the search evaluation dataset script
# Make sure to activate your conda environment first

# Activate the conda environment (adjust path as needed)
# conda activate podcast-shownotes

# Basic usage with required parameters
python scripts/build_search_eval_dataset.py \
    --dbname podcast_shownotes \
    --openai-api-key "your-openai-api-key-here"

# Full usage with all parameters
python scripts/build_search_eval_dataset.py \
    --dbhost localhost \
    --dbport 5432 \
    --dbname podcast_shownotes \
    --dbuser postgres \
    --dbpassword your_password \
    --openai-api-key "your-openai-api-key-here" \
    --openai-base-url "https://api.openai.com/v1" \
    --batch-size 5 \
    --batch-delay 15 \
    --output-dir "./search_eval_results"

# Usage with environment variables
export OPENAI_API_KEY="your-openai-api-key-here"
export DB_PASSWORD="your-database-password"

python scripts/build_search_eval_dataset.py \
    --dbname podcast_shownotes \
    --dbpassword "$DB_PASSWORD" \
    --openai-api-key "$OPENAI_API_KEY" \
    --batch-size 5 \
    --batch-delay 15
