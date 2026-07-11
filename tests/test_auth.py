"""Tests for JWT authentication and RBAC permissions."""

from __future__ import annotations

import os
import time

import pytest

from kairn.auth.jwt import TokenExpiredError, TokenInvalidError, create_token, verify_token
from kairn.auth.permissions import can_admin, can_read, can_write, check_permission


_TEST_SECRET = "unit-test-signing-secret-32-bytes!!"


@pytest.fixture(autouse=True)
def _jwt_secret(monkeypatch):
    """create_token fails closed without KAIRN_JWT_SECRET (weakness-audit
    rank 37); every JWT test signs with an explicit test secret."""
    monkeypatch.setenv("KAIRN_JWT_SECRET", _TEST_SECRET)


class TestJWT:
    """Test JWT token creation and validation."""

    def test_create_token_basic(self) -> None:
        token = create_token("user-123", "org-456")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token_valid(self) -> None:
        token = create_token("user-123", "org-456", exp_minutes=1)
        payload = verify_token(token, _TEST_SECRET)

        assert payload["sub"] == "user-123"
        assert payload["org"] == "org-456"
        assert "exp" in payload

    def test_verify_token_custom_expiry(self) -> None:
        token = create_token("user-789", "org-101", exp_minutes=120)
        payload = verify_token(token, _TEST_SECRET)

        assert payload["sub"] == "user-789"
        exp_time = payload["exp"]
        now = int(time.time())
        assert exp_time > now + 3600

    def test_verify_token_expired(self) -> None:
        token = create_token("user-123", "org-456", exp_minutes=-1)

        with pytest.raises(TokenExpiredError):
            verify_token(token, _TEST_SECRET)

    def test_verify_token_invalid_signature(self) -> None:
        token = create_token("user-123", "org-456")
        wrong_secret = "wrong-secret"

        with pytest.raises(TokenInvalidError):
            verify_token(token, wrong_secret)

    def test_verify_token_malformed(self) -> None:
        secret = os.environ.get("KAIRN_JWT_SECRET", "test-secret-key-do-not-use")

        with pytest.raises(TokenInvalidError):
            verify_token("not.a.valid.jwt", secret)


class TestPermissions:
    """Test RBAC permission checks."""

    def test_role_hierarchy(self) -> None:
        assert check_permission("owner", "reader")
        assert check_permission("owner", "contributor")
        assert check_permission("owner", "maintainer")
        assert check_permission("owner", "owner")

        assert check_permission("maintainer", "reader")
        assert check_permission("maintainer", "contributor")
        assert check_permission("maintainer", "maintainer")
        assert not check_permission("maintainer", "owner")

        assert check_permission("contributor", "reader")
        assert check_permission("contributor", "contributor")
        assert not check_permission("contributor", "maintainer")
        assert not check_permission("contributor", "owner")

        assert check_permission("reader", "reader")
        assert not check_permission("reader", "contributor")
        assert not check_permission("reader", "maintainer")
        assert not check_permission("reader", "owner")

    def test_can_read(self) -> None:
        assert can_read("reader")
        assert can_read("contributor")
        assert can_read("maintainer")
        assert can_read("owner")

    def test_can_write(self) -> None:
        assert not can_write("reader")
        assert can_write("contributor")
        assert can_write("maintainer")
        assert can_write("owner")

    def test_can_admin(self) -> None:
        assert not can_admin("reader")
        assert not can_admin("contributor")
        assert can_admin("maintainer")
        assert can_admin("owner")

    def test_check_permission_invalid_role(self) -> None:
        assert not check_permission("invalid", "reader")
        assert not check_permission("reader", "invalid")
