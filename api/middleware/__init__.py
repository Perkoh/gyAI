"""gyAI API middleware package: authentication and global error handling."""

from api.middleware.auth import require_api_key, require_admin_key, API_KEY_HEADER
from api.middleware.error_handlers import (
    APIError,
    BadRequestError,
    InvalidDomainError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    PayloadTooLargeError,
    RateLimitError,
    InternalServerError,
    register_error_handlers,
)

__all__ = [
    "require_api_key",
    "require_admin_key",
    "API_KEY_HEADER",
    "APIError",
    "BadRequestError",
    "InvalidDomainError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "PayloadTooLargeError",
    "RateLimitError",
    "InternalServerError",
    "register_error_handlers",
]