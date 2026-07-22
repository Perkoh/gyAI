"""
config/
───────
Centralised ADIS configuration package.

Exports:
    settings   — runtime settings object (reads from .env / environment vars)
    constants  — static lookup tables and enumerated values

Quick import:
    from config import settings, constants
    from config.settings import settings
    from config.constants import PHISHING_KEYWORDS, SUSPICIOUS_TLDS, LABEL
"""

from config.settings import settings
from config import constants

__all__ = ["settings", "constants"]
