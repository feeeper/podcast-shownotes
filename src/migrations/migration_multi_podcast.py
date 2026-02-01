from argparse import ArgumentParser

import psycopg2
import psycopg2.extras


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
    cur = conn.cursor()

    # Create podcasts table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS podcasts (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            slug VARCHAR(128) NOT NULL UNIQUE,
            name VARCHAR(256) NOT NULL,
            rss_url VARCHAR(512) NOT NULL,
            language VARCHAR(10) NOT NULL DEFAULT 'en'
        );
    """)
    conn.commit()
    print('Created podcasts table')

    # Add new columns to episodes table
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'episodes'
                AND column_name = 'podcast_id'
            ) THEN
                ALTER TABLE episodes
                    ADD COLUMN podcast_id UUID
                    REFERENCES podcasts(id);
            END IF;
        END $$;
    """)
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'episodes'
                AND column_name = 'rss_guid'
            ) THEN
                ALTER TABLE episodes
                    ADD COLUMN rss_guid VARCHAR(512);
            END IF;
        END $$;
    """)
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'episodes'
                AND column_name = 'episode_link'
            ) THEN
                ALTER TABLE episodes
                    ADD COLUMN episode_link VARCHAR(512);
            END IF;
        END $$;
    """)
    conn.commit()
    print('Added podcast_id, rss_guid, episode_link to episodes')

    # Make episode_number nullable
    cur.execute("""
        ALTER TABLE episodes
            ALTER COLUMN episode_number DROP NOT NULL;
    """)
    conn.commit()
    print('Made episode_number nullable')

    # Insert devzen podcast and migrate existing data
    cur.execute("""
        INSERT INTO podcasts (slug, name, rss_url, language)
        VALUES ('devzen', 'DevZen Podcast',
                'https://devzen.ru/feed/', 'ru')
        ON CONFLICT (slug) DO NOTHING;
    """)
    conn.commit()

    cur.execute("""
        UPDATE episodes
        SET podcast_id = (
            SELECT id FROM podcasts WHERE slug = 'devzen'
        )
        WHERE podcast_id IS NULL;
    """)
    conn.commit()
    print('Migrated existing episodes to devzen podcast')

    # Set podcast_id NOT NULL after migration
    cur.execute("""
        ALTER TABLE episodes
            ALTER COLUMN podcast_id SET NOT NULL;
    """)
    conn.commit()

    # Drop old unique index on episode_number, create new one
    cur.execute("""
        DROP INDEX IF EXISTS ix_episode_number;
    """)
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_podcast_rss_guid
            ON episodes (podcast_id, rss_guid);
    """)
    conn.commit()
    print('Created ix_podcast_rss_guid index')

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
