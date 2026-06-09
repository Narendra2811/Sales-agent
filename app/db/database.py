
import logging
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings        # Our typed settings object
from app.db.models import Base         # The Base all models inherit from

logger = logging.getLogger(__name__)



_connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,

    echo=False,
)

logger.info(f"Database engine created. URL: {settings.DATABASE_URL}")



@event.listens_for(engine, "connect")
def set_sqlite_pragmas(dbapi_connection, connection_record):
   
    if not settings.DATABASE_URL.startswith("sqlite"):
        return  # Not SQLite, do nothing

    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")   # Enable Write-Ahead Logging
    cursor.execute("PRAGMA foreign_keys=ON")    # Enforce foreign key constraints
    cursor.execute("PRAGMA synchronous=NORMAL") # Faster writes, still safe with WAL
    cursor.close()



SessionLocal: sessionmaker = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)



def init_db() -> None:
    """
    Creates all database tables if they don't already exist.
    """
    logger.info("Initializing database — creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialization complete.")


def get_db() -> Session:
    """
    FastAPI dependency that provides a database session per request.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def verify_db_connection() -> bool:
    """
    Tests that the database is reachable and the schema is set up.
    Used by the /health endpoint to verify database connectivity.

    Returns:
        True if database is accessible, False if there's a problem.
    """
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
