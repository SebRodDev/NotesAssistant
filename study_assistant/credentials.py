"""Gemini API key lookup and macOS Keychain storage."""

from __future__ import annotations

import os

import keyring

SERVICE_NAME = "Purdue Study Assistant"
ACCOUNT_NAME = "gemini-api-key"
_EPHEMERAL_KEY = False


def get_api_key() -> str | None:
    """Prefer an explicit environment variable, then the user's Keychain value."""
    environment_key = os.environ.get("GEMINI_API_KEY")
    if environment_key:
        return environment_key
    try:
        return keyring.get_password(SERVICE_NAME, ACCOUNT_NAME)
    except keyring.errors.KeyringError:
        return None


def save_api_key(api_key: str) -> bool:
    """Save to Keychain, or keep it in this app process if Keychain is unavailable."""
    global _EPHEMERAL_KEY
    try:
        keyring.set_password(SERVICE_NAME, ACCOUNT_NAME, api_key.strip())
        return True
    except keyring.errors.KeyringError:
        os.environ["GEMINI_API_KEY"] = api_key.strip()
        _EPHEMERAL_KEY = True
        return False


def delete_api_key() -> None:
    global _EPHEMERAL_KEY
    if _EPHEMERAL_KEY:
        os.environ.pop("GEMINI_API_KEY", None)
        _EPHEMERAL_KEY = False
    try:
        keyring.delete_password(SERVICE_NAME, ACCOUNT_NAME)
    except keyring.errors.KeyringError:
        pass
