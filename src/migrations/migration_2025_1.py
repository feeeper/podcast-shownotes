from argparse import ArgumentParser
import psycopg2.extras



# install pg_trgm extension for trigram similarity
def run(
        host: str = 'localhost',
        port: int = 5432,
        user: str = 'postgres',
        password: str = 'postgres',
        dbname: str = 'podcast_shownotes'
) -> None:
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname
    )
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    conn.commit()
    print("pg_trgm extension installed")

    # Index for Full-Text Search (FTS), create only if it does not exist
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'ix_fts_en_text' AND n.nspname = 'public'
            ) THEN
                CREATE INDEX ix_fts_en_text ON sentences USING GIN (to_tsvector('english', text));
            END IF;
        END
        $$;
    """)
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'ix_fts_ru_text' AND n.nspname = 'public'
            ) THEN
                CREATE INDEX ix_fts_ru_text ON sentences USING GIN (to_tsvector('russian', text));
            END IF;
        END
        $$;
    """)
    conn.commit()
    print("Indexes for FTS created")

    # Index for trigram search (pg_trgm), create only if it does not exist
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'trgm_idx' AND n.nspname = 'public'
            ) THEN
                CREATE INDEX trgm_idx ON sentences USING GIN (text gin_trgm_ops);
            END IF;
        END
        $$;
    """)
    conn.commit()
    print("Index for trigram search created")

    # Create search history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            comment TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    print("Search history table created")

    # Create search results table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_results (
            id SERIAL PRIMARY KEY,
            search_history_id INT NOT NULL REFERENCES search_history(id) ON DELETE CASCADE,
            sentence_id UUID NOT NULL,
            similarity FLOAT NOT NULL
        );
    """)
    conn.commit()
    print("Search results table created")

    cur.close()
    conn.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--user', type=str, default='postgres')
    parser.add_argument('--password', type=str, default='postgres')
    parser.add_argument('--dbname', type=str, default='podcast_shownotes')
    args = parser.parse_args()

    print(f'Running migration with arguments: {args}')
    run(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )
