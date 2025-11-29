#!/usr/bin/env python3
"""
Script to build search evaluation dataset by generating search queries for podcast segments.

This script:
1. Connects to PostgreSQL database
2. Selects random segments (excluding first and last) for each episode
3. Sends parallel requests to OpenAI API to generate search queries
4. Saves results to JSON files per episode
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
import re
import time
from typing import List, Dict, Any
import uuid
import traceback

import psycopg2
import psycopg2.extras
from openai import AsyncOpenAI, RateLimitError


class DatabaseConnection:
    """Handles PostgreSQL database connection and queries."""
    
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        self.connection_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
    
    def get_random_segments(self) -> List[Dict[str, Any]]:
        """
        Get one random segment per episode from the database, excluding first and last segments.
        
        Returns:
            List of segment dictionaries with episode_id, text, and other fields
        """
        query = """
        WITH episode_segments AS (
            SELECT 
                s.id,
                s.episode_id,
                s.text,
                s.segment_number,
                e.episode_number,
                ROW_NUMBER() OVER (PARTITION BY s.episode_id ORDER BY s.segment_number) as rn,
                COUNT(*) OVER (PARTITION BY s.episode_id) as total_segments
            FROM segments s
            JOIN episodes e ON s.episode_id = e.id
        ),
        eligible_segments AS (
            SELECT 
                id,
                episode_id,
                text,
                segment_number,
                episode_number,
                ROW_NUMBER() OVER (PARTITION BY episode_id ORDER BY RANDOM()) as random_rank
            FROM episode_segments
            WHERE rn > 1 AND rn < total_segments  -- Exclude first and last segments
        )
        SELECT 
            id,
            episode_id,
            text,
            segment_number,
            episode_number
        FROM eligible_segments
        WHERE random_rank = 1  -- Select only one random segment per episode
        ORDER BY episode_number
        """
        
        with psycopg2.connect(**self.connection_params) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(query)
                segments = [dict(row) for row in cursor.fetchall()]
                return segments


class OpenAIClient:
    """Handles OpenAI API requests for generating search queries."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.system_prompt = """# Role and Objective
- You are an assistant designed to convert user-supplied text into concise, natural-language search queries.

# Instructions
- When the user provides input, generate a search query of no more than 5 words that accurately represents the main idea of the input.
- Ensure the search query reads naturally and could be used directly in a search engine.
- If an appropriate query of 5 words or fewer cannot be formed, return an error message explaining this.
- Output only the search query or error message, never include the original user input.

# Output Format
- For a valid search query, output:
```json
{ "query": "<search query>" }
```
- For an error, output:
```json
{ "error": "Unable to form a suitable search query of 5 words or fewer." }
```
- Respond strictly using the specified JSON format. Do not include any additional text.

# Process Checklist
- Begin with a concise checklist (3-7 bullets) of the steps you will take to generate the output; keep checklist items conceptual.

# Reasoning Effort
- Set reasoning_effort = minimal, as the task is straightforward.

# Verbosity
- Keep outputs brief and to the point.

# Stop Conditions
- After processing, respond only with the required JSON object structure.
- Do not include any additional commentary or user input."""

#         self.system_prompt = """# Роль и цель
# - Вы — помощник, призванный преобразовывать текст, введенный пользователем, в краткие поисковые запросы на естественном языке.

# # Инструкции
# - Когда пользователь вводит данные, сформулируйте поисковый запрос длиной не более 5 слов, точно отражающий основную идею.
# - Убедитесь, что поисковый запрос читается естественно и может быть использован непосредственно в поисковой системе.
# - Если не удается сформировать подходящий запрос длиной не более 5 слов, верните сообщение об ошибке с пояснением.
# - Выводите только поисковый запрос или сообщение об ошибке, никогда не включайте исходный пользовательский ввод.

# # Формат вывода
# - Для корректного поискового запроса выведите:
# ```json
# { "query": "<search query>" }
# ```
# - Для ошибки выведите:
# ```json
# { "error": "Не удалось сформировать подходящий поисковый запрос длиной не более 5 слов." }
# ```
# - Ответ должен быть строго в указанном формате JSON. Не добавляйте дополнительный текст.

# # Контрольный список процесса
# - Начните с краткого контрольного списка (3–7 пунктов) шагов, которые вы выполните для получения выходных данных; пункты контрольного списка должны быть концептуальными.

# # Усилия на обоснование
# - Установите reasoning_effort = minimal, поскольку задача проста.

# # Многословность
# - Выходные данные должны быть краткими и по существу.

# # Условия остановки
# - После обработки ответьте только необходимой структурой JSON-объекта.
# - Не включайте никаких дополнительных комментариев или пользовательского ввода.

# Ваш ответ должен быть на том же языке, что и текст, предоставленный пользователем."""

        self.json_schema = {
            "name": "query_object",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search or database query as a string"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    
    async def generate_search_query(self, text: str) -> Dict[str, Any]:
        """
        Generate a search query for the given text using OpenAI API.
        
        Args:
            text: The segment text to generate a query for
            
        Returns:
            Dictionary containing the generated query or error
        """
        retries = 0
        max_retries = 10
        err = None
        while retries < max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ],
                    response_format={
                        'type': 'json_schema',
                        'json_schema': self.json_schema
                    },
                    temperature=1,
                    max_completion_tokens=500
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
            
            except RateLimitError as e:
                error_msg = str(e)
                wait_time = None

                # Look for "Please try again in Xs" pattern
                match = re.search(r"Please try again in ([0-9.]+)s", error_msg)
                if match:
                    wait_time = float(match.group(1))
                else:
                    # fallback: exponential backoff
                    wait_time = min(2 ** retries, 60)

                print(f"Rate limit hit. Waiting {wait_time:.2f}s before retry...")
                time.sleep(wait_time)

                retries += 1
                err = e
                
            except Exception as e:
                return {"error": f"API request failed: {str(e)}", "stack_trace": traceback.format_exc()}
        
        return {"error": f"API request failed: {str(err)}", "stack_trace": traceback.format_exc()}

    
    async def generate_search_queries_parallel(self, segments: List[Dict[str, Any]], batch_size: int = 5, delay_seconds: int = 15) -> List[Dict[str, Any]]:
        """
        Generate search queries for multiple segments in batches with throttling.
        
        Args:
            segments: List of segment dictionaries
            batch_size: Number of requests to send in parallel per batch (default: 5)
            delay_seconds: Delay in seconds between batches (default: 15)
            
        Returns:
            List of results with segment info and generated queries
        """
        combined_results = []
        total_segments = len(segments)
        total_batches = (total_segments + batch_size - 1) // batch_size
        
        for i in range(0, total_segments, batch_size):
            batch = segments[i:i + batch_size]
            batch_number = (i // batch_size) + 1            
            print(f"Processing batch {batch_number}/{total_batches} ({len(batch)} segments)...")

            # Create tasks for current batch
            tasks = []
            for segment in batch:
                task = self.generate_search_query(segment["text"])
                tasks.append(task)
            
            # Execute current batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine segment info with results for current batch
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    result = {"error": f"Request failed: {str(result)}"}
                    print(f"Batch {batch_number} failed: {result}")
                batch_element = batch[j]
                combined_results.append({
                    "segment_id": str(batch_element["id"]),
                    "episode_id": str(batch_element["episode_id"]),
                    "episode_number": batch_element["episode_number"],
                    "segment_number": batch_element["segment_number"],
                    "text": batch_element["text"],
                    "result": result
                })
            
            # Add delay between batches (except for the last batch)
            if i + batch_size < total_segments:
                print(f"Waiting {delay_seconds} seconds before next batch...")
                await asyncio.sleep(delay_seconds)
        
        return combined_results


def save_results_to_files(results: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Save results to JSON files organized by episode.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save the files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group results by episode_id
    episode_results = {}
    for result in results:
        episode_id = result['episode_id']
        if episode_id not in episode_results:
            episode_results[episode_id] = []
        episode_results[episode_id].append(result)
    
    # Save each episode's results to a separate file
    for episode_id, episode_data in episode_results.items():
        episode_number = episode_data[0]['episode_number']
        filename = f"{episode_number}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(episode_data)} results to {filepath}")


async def main():
    """Main function to orchestrate the search query generation process."""
    parser = argparse.ArgumentParser(
        description="Build search evaluation dataset by generating search queries for podcast segments"
    )
    
    # Database connection arguments
    parser.add_argument("--dbhost", type=str, default="localhost", help="Database host")
    parser.add_argument("--dbport", type=int, default=5432, help="Database port")
    parser.add_argument("--dbname", type=str, required=True, help="Database name")
    parser.add_argument("--dbuser", type=str, default="postgres", help="Database user")
    parser.add_argument("--dbpassword", type=str, default="postgres", help="Database password")
    
    # OpenAI arguments
    parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--openai-base-url", type=str, default="https://api.openai.com/v1", 
                       help="OpenAI API base URL")
    
    # Script arguments
    parser.add_argument("--batch-size", type=int, default=5, 
                       help="Number of requests to send in parallel per batch (default: 5)")
    parser.add_argument("--batch-delay", type=int, default=15, 
                       help="Delay in seconds between batches (default: 15)")
    parser.add_argument("--output-dir", type=str, default="./search_eval_results", 
                       help="Output directory for results (default: ./search_eval_results)")
    
    args = parser.parse_args()
    
    print(f"Connecting to database: {args.dbhost}:{args.dbport}/{args.dbname}")
    print(f"Batch size: {args.batch_size}, Delay between batches: {args.batch_delay}s")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize database connection
    db = DatabaseConnection(
        host=args.dbhost,
        port=args.dbport,
        dbname=args.dbname,
        user=args.dbuser,
        password=args.dbpassword
    )
    
    # Get random segments
    print("Fetching random segments from database...")
    segments = db.get_random_segments()
    
    if not segments:
        print("No segments found in the database.")
        return
    
    print(f"Found {len(segments)} segments to process")
    
    # Initialize OpenAI client
    openai_client = OpenAIClient(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url
    )
    
    # Generate search queries in batches with throttling
    print("Generating search queries using OpenAI API...")
    results = await openai_client.generate_search_queries_parallel(
        segments, 
        batch_size=args.batch_size, 
        delay_seconds=args.batch_delay
    )
    
    # Save results to files
    output_dir = Path(args.output_dir)
    save_results_to_files(results, output_dir)
    
    print("Search evaluation dataset generation completed!")


if __name__ == "__main__":
    asyncio.run(main())
