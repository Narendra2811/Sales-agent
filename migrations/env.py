import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context


from app.config import settings
from app.db.models import Base  # SQLAlchemy Base with all table definitions
from app.db import models as _models

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
logger.info(f"Alembic using database: {settings.DATABASE_URL}")

target_metadata = Base.metadata


# used when doing major schema changes
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    In offline mode, Alembic generates SQL scripts WITHOUT connecting to the DB.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,  # Render values literally instead of as placeholders
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()
        logger.info("Offline migrations completed.")


# used when smaller changes
def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In online mode, Alembic connects to the real database and applies migrations
    directly.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()
            logger.info("Online migrations completed successfully.")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
