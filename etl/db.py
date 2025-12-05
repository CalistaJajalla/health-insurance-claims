from sqlalchemy import create_engine

def get_engine(secrets):
    # Cloud: use Supabase Pooler
    if "DATABASE_URL" in secrets:
        return create_engine(secrets["DATABASE_URL"], pool_pre_ping=True)

    # Local: use normal Postgres connection
    user = secrets["DB_USER"]
    password = secrets["DB_PASSWORD"]
    host = secrets["POSTGRES_HOST"]
    port = secrets["POSTGRES_PORT"]
    db = secrets["POSTGRES_DB"]

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)
