"""RBAC permission checks for Kairn workspaces.

STATUS (2026-07-11, weakness-audit rank 17): these checks are NOT wired into
any execution path - no MCP tool, CLI command, or engine consults them, so
they provide no isolation today. They are the intended vocabulary for a
future multi-user mode, nothing more. If you wire a caller, update this
docstring AND the tripwire test
(tests/test_weakness_audit_wave3.py::test_rbac_unenforced_tripwire).
"""

from __future__ import annotations

_ROLE_HIERARCHY = {
    "owner": 4,
    "maintainer": 3,
    "contributor": 2,
    "reader": 1,
}


def check_permission(role: str, required_role: str) -> bool:
    """Check if a role has permission to perform an action requiring a specific role."""
    if role not in _ROLE_HIERARCHY or required_role not in _ROLE_HIERARCHY:
        return False
    role_level = _ROLE_HIERARCHY[role]
    required_level = _ROLE_HIERARCHY[required_role]
    return role_level >= required_level


def can_read(role: str) -> bool:
    """Check if a role can read workspace content."""
    return check_permission(role, "reader")


def can_write(role: str) -> bool:
    """Check if a role can write to workspace content."""
    return check_permission(role, "contributor")


def can_admin(role: str) -> bool:
    """Check if a role can perform administrative actions."""
    return check_permission(role, "maintainer")
