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

    # Index for Full-Text Search (FTS)
    cur.execute("CREATE INDEX IX_FTS_EN_TEXT ON sentences USING GIN (to_tsvector('english', text));")
    cur.execute("CREATE INDEX IX_FTS_RU_TEXT ON sentences USING GIN (to_tsvector('russian', text));")
    conn.commit()

    # Index for trigram search (pg_trgm)
    cur.execute("CREATE INDEX trgm_idx ON sentences USING GIN (text gin_trgm_ops);")
    conn.commit()
    cur.close()
    conn.close()
    print("pg_trgm extension installed")


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
