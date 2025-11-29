import feedparser
import re
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from celery import chain
import logging

logger = logging.getLogger(__name__)

# Import app - this will work because celery_app is already initialized
from celery_app import app

# Get storage directory from environment or use default
STORAGE_DIR = Path(os.getenv('EPISODES_STORAGE_DIR', 'episodes'))


@app.task(name="app.tasks.check_rss_feed")
def check_rss_feed():
    """
    Celery Beat task that checks the RSS feed at https://devzen.ru/feed/
    and triggers processing chains for new episodes.
    """
    feed_url = 'https://devzen.ru/feed/'
    
    try:
        logger.info(f'Checking RSS feed: {feed_url}')
        feed = feedparser.parse(feed_url)
        
        if feed.bozo:
            logger.warning(f'Feed parsing error: {feed.bozo_exception}')
            return {'status': 'error', 'message': str(feed.bozo_exception)}
        
        feed_info = {
            'title': feed.feed.get('title', 'Unknown'),
            'link': feed.feed.get('link', ''),
            'updated': feed.feed.get('updated', ''),
            'entry_count': len(feed.entries)
        }
        
        logger.info(f'Feed checked successfully. Found {feed_info["entry_count"]} entries.')
        logger.info(f'Feed title: {feed_info["title"]}')
        logger.info(f'Feed updated: {feed_info["updated"]}')
        
        # Detect new episodes and trigger processing chains
        new_episodes = []
        for entry in feed.entries:
            try:
                # Skip non-episode entries
                non_episode_link_prefixes = (
                    'https://devzen.ru/themes',
                    'https://devzen.ru/no-themes'
                )
                if entry.link.startswith(non_episode_link_prefixes):
                    continue

                # Extract episode number
                matches = re.findall('/episode-(\\d+)', entry.link)
                if not matches:
                    continue
                
                episode_num = int(matches[0])
                episode_path = STORAGE_DIR / str(episode_num)
                
                # Check if episode is new (doesn't exist or missing MP3)
                is_new = False
                if not episode_path.exists():
                    is_new = True
                elif not (episode_path / 'episode.mp3').exists():
                    is_new = True
                
                if is_new:
                    episode_data = {
                        'episode_number': episode_num,
                        'title': entry.title,
                        'episode_link': entry.link,
                        'mp3_link': entry.enclosures[0].href if entry.enclosures else None,
                        'published': datetime(*entry.published_parsed[:6]).isoformat() if entry.published_parsed else None,
                        'summary': entry.summary,
                        'authors': [author.name for author in entry.authors] if hasattr(entry, 'authors') else [],
                        'html_content': entry.content[0].value if entry.content else '',
                    }
                    new_episodes.append(episode_data)
                    logger.info(f'Found new episode: {episode_num} - {entry.title}')
            except Exception as e:
                logger.error(f'Error processing entry {entry.link}: {e}', exc_info=True)
                continue
        
        # Trigger processing chain for each new episode
        for episode_data in new_episodes:
            logger.info(f'Triggering processing chain for episode {episode_data["episode_number"]}')
            # Create and execute the chain
            workflow = chain(
                download_episode_metadata.s(episode_data),
                download_episode_mp3.s(),
                process_episode.s()
            )
            workflow.apply_async()
        
        return {
            'status': 'success',
            'feed_info': feed_info,
            'new_episodes_count': len(new_episodes),
            'new_episodes': [ep['episode_number'] for ep in new_episodes],
            'checked_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f'Error checking RSS feed: {e}', exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'checked_at': datetime.now().isoformat()
        }


@app.task(name="app.tasks.download_episode_metadata")
def download_episode_metadata(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Downloads and saves episode metadata (episode.json, episode.html, metadata.json).
    """
    episode_num = episode_data['episode_number']
    logger.info(f'[STEP 1] Starting metadata download for episode {episode_num}')
    logger.info(f'[STEP 1] Episode title: {episode_data.get("title", "Unknown")}')
    logger.info(f'[STEP 1] Episode link: {episode_data.get("episode_link", "N/A")}')
    
    # Simulate work with delay
    time.sleep(2)
    
    logger.info(f'[STEP 1] Metadata download completed for episode {episode_num}')
    
    # Return episode_data with path for next task
    episode_data['episode_path'] = str(STORAGE_DIR / str(episode_num))
    return episode_data


@app.task(name="app.tasks.download_episode_mp3")
def download_episode_mp3(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Downloads the MP3 file for an episode.
    """
    episode_num = episode_data['episode_number']
    mp3_link = episode_data.get('mp3_link', 'N/A')
    
    logger.info(f'[STEP 2] Starting MP3 download for episode {episode_num}')
    logger.info(f'[STEP 2] MP3 link: {mp3_link}')
    logger.info(f'[STEP 2] Episode path: {episode_data.get("episode_path", "N/A")}')
    
    # Simulate work with delay
    time.sleep(3)
    
    logger.info(f'[STEP 2] MP3 download completed for episode {episode_num}')
    
    episode_data['mp3_path'] = str(Path(episode_data.get('episode_path', STORAGE_DIR / str(episode_num))) / 'episode.mp3')
    return episode_data


@app.task(name="app.tasks.process_episode")
def process_episode(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final processing step for an episode (can trigger transcription, segmentation, etc.).
    This is a placeholder that can be extended with additional processing steps.
    """
    episode_num = episode_data['episode_number']
    
    logger.info(f'[STEP 3] Starting final processing for episode {episode_num}')
    logger.info(f'[STEP 3] Episode path: {episode_data.get("episode_path", "N/A")}')
    logger.info(f'[STEP 3] MP3 path: {episode_data.get("mp3_path", "N/A")}')
    
    # Simulate work with delay
    time.sleep(2)
    
    logger.info(f'[STEP 3] Final processing completed for episode {episode_num}')
    logger.info(f'[CHAIN COMPLETE] Episode {episode_num} processing chain finished successfully!')
    
    return {
        'status': 'success',
        'episode_number': episode_num,
        'episode_path': episode_data.get('episode_path', 'N/A'),
        'processed_at': datetime.now().isoformat()
    }


