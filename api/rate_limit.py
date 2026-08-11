import asyncio
import math
import time
from collections import deque
from collections.abc import Callable

from fastapi import HTTPException, Request, status


class RateLimiter:
    """Process-local sliding-window rate limiter keyed by client IP."""

    def __init__(
        self,
        requests: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive integers.")

        self.requests = requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._clients: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = clock()

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",", maxsplit=1)[0].strip()

        if request.client:
            return request.client.host

        return "unknown"

    async def __call__(self, request: Request) -> None:
        client_ip = self._client_ip(request)
        now = self._clock()
        cutoff = now - self.window_seconds

        async with self._lock:
            if now - self._last_cleanup >= self.window_seconds:
                self._clients = {
                    ip: timestamps
                    for ip, timestamps in self._clients.items()
                    if timestamps and timestamps[-1] > cutoff
                }
                self._last_cleanup = now

            timestamps = self._clients.setdefault(client_ip, deque())
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.requests:
                retry_after = max(
                    1,
                    math.ceil(self.window_seconds - (now - timestamps[0])),
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )

            timestamps.append(now)
