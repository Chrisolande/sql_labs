from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    configure_mappers,  # noqa: E402
)
from sqlalchemy.pool import AsyncAdaptedQueuePool

from app.core.config.settings import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    poolclass=AsyncAdaptedQueuePool,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


configure_mappers()
