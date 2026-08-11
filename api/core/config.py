import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer.") from error

    if value < 1:
        raise RuntimeError(f"{name} must be greater than zero.")

    return value


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    pinecone_api_key: str
    pinecone_index: str
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        required = {
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
            "PINECONE_INDEX": os.getenv("PINECONE_INDEX"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        return cls(
            groq_api_key=required["GROQ_API_KEY"],  # type: ignore[arg-type]
            pinecone_api_key=required["PINECONE_API_KEY"],  # type: ignore[arg-type]
            pinecone_index=required["PINECONE_INDEX"],  # type: ignore[arg-type]
            rate_limit_requests=_positive_int("RATE_LIMIT_REQUESTS", 10),
            rate_limit_window_seconds=_positive_int(
                "RATE_LIMIT_WINDOW_SECONDS", 60
            ),
        )
