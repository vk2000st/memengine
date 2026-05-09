class MemEngineError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class AuthError(MemEngineError):
    """Invalid or missing API key."""


class NotFoundError(MemEngineError):
    """Requested resource does not exist."""


class ValidationError(MemEngineError):
    """Request payload failed server-side validation."""


class ConflictError(MemEngineError):
    """Resource already exists (e.g. duplicate agent slug)."""
