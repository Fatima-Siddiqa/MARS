from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine():
    # Lazy so importing the app (e.g. in tests) never requires a live DB.
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def get_db():
    """FastAPI dependency yielding a DB session."""
    session = sessionmaker(bind=get_engine(), autoflush=False)()
    try:
        yield session
    finally:
        session.close()
