import uuid
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from app.schemas import Job, JobRequest


class JobStore:
    """Stores job state as JSON in Redis so the API and the worker share it."""

    def __init__(self, redis: Redis, ttl_s: int):
        self._redis = redis
        self._ttl_s = ttl_s

    @staticmethod
    def _key(job_id: str) -> str:
        return f"article-image-gen:job:{job_id}"

    async def create(self, request: JobRequest) -> Job:
        job = Job(id=uuid.uuid4().hex, request=request)
        await self.save(job)
        return job

    async def get(self, job_id: str) -> Job | None:
        raw = await self._redis.get(self._key(job_id))
        return Job.model_validate_json(raw) if raw else None

    async def save(self, job: Job) -> None:
        job.updated_at = datetime.now(timezone.utc)
        await self._redis.set(self._key(job.id), job.model_dump_json(), ex=self._ttl_s)

    async def update(self, job_id: str, **fields: Any) -> Job:
        job = await self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job = job.model_copy(update=fields)
        await self.save(job)
        return job
