from argparse import ArgumentParser
import re

import feedparser
import psycopg2
import psycopg2.extras


def extract_episode_number(link: str) -> int | None:
    """Extract episode number from devzen link."""
    match = re.search(r'/episode-(\d+)/?', link)
    if match:
        return int(match.group(1))
    return None


def fetch_rss_guid_mapping(rss_url: str) -> dict[int, tuple[str, str]]:
    """
    Fetch RSS feed and build mapping of episode_number
    to (rss_guid, episode_link).
    """
    print(f'Fetching RSS feed: {rss_url}')
    feed = feedparser.parse(rss_url)

    if feed.bozo:
        print(f'Warning: Feed parsing error: {feed.bozo_exception}')

    mapping = {}
    for entry in feed.entries:
        link = getattr(entry, 'link', '')
        guid = (
            getattr(entry, 'id', None)
            or getattr(entry, 'guid', None)
            or link
        )
        episode_num = extract_episode_number(link)
        if episode_num is not None:
            mapping[episode_num] = (guid, link)

    print(f'Found {len(mapping)} episodes in RSS feed')
    return mapping


def run(
    host: str = 'localhost',
    port: int = 5432,
    user: str = 'postgres',
    password: str = 'postgres',
    dbname: str = 'podcast_shownotes',
) -> None:
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
    )
    cur = conn.cursor(
        cursor_factory=psycopg2.extras.DictCursor
    )

    # Get devzen podcast
    cur.execute("""
        SELECT id, rss_url FROM podcasts WHERE slug = 'devzen';
    """)
    row = cur.fetchone()
    if not row:
        print('devzen podcast not found')
        cur.close()
        conn.close()
        return

    podcast_id = row['id']
    rss_url = row['rss_url']

    # Fetch RSS and build guid mapping
    guid_mapping = fetch_rss_guid_mapping(rss_url)

    # Get episodes that need backfill
    cur.execute("""
        SELECT id, episode_number
        FROM episodes
        WHERE podcast_id = %s
          AND rss_guid IS NULL
          AND episode_number IS NOT NULL;
    """, (podcast_id,))
    episodes = cur.fetchall()
    print(f'Found {len(episodes)} episodes needing backfill')

    updated = 0
    not_found = []
    for ep in episodes:
        ep_id = ep['id']
        ep_num = ep['episode_number']

        if ep_num in guid_mapping:
            rss_guid, episode_link = guid_mapping[ep_num]
            cur.execute("""
                UPDATE episodes
                SET rss_guid = %s, episode_link = %s
                WHERE id = %s;
            """, (rss_guid, episode_link, ep_id))
            updated += 1
        else:
            not_found.append(ep_num)

    conn.commit()
    print(f'Updated {updated} episodes')

    if not_found:
        print(
            f'Episodes not found in RSS '
            f'(may be too old): {not_found[:10]}'
            f'{"..." if len(not_found) > 10 else ""}'
        )

    cur.close()
    conn.close()
    print('Migration complete')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--host', type=str, default='localhost'
    )
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument(
        '--user', type=str, default='postgres'
    )
    parser.add_argument(
        '--password', type=str, default='postgres'
    )
    parser.add_argument(
        '--dbname', type=str, default='podcast_shownotes'
    )
    args = parser.parse_args()

    print(f'Running migration with arguments: {args}')
    run(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
    )
