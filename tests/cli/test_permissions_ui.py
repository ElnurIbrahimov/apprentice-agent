"""Tests for permission tier UI."""
import pytest
from aura.cli.permissions_ui import (
    PermissionMode, cycle_permission_mode, get_mode_description,
    get_mode_indicator, get_mode_short,
    should_auto_approve_edit, should_auto_approve_command, should_block_mutations,
)

def test_permission_modes_exist():
    assert PermissionMode.PLAN == "plan"
    assert PermissionMode.CAREFUL == "careful"
    assert PermissionMode.AUTO_EDIT == "auto_edit"
    assert PermissionMode.FULL_AUTO == "full_auto"

def test_cycle_forward():
    # auto_edit is deferred from the cycle (enforcement not wired up yet)
    assert cycle_permission_mode("plan") == "careful"
    assert cycle_permission_mode("careful") == "full_auto"
    assert cycle_permission_mode("full_auto") == "plan"

def test_cycle_invalid_defaults_to_careful():
    assert cycle_permission_mode("garbage") == "careful"

def test_mode_description():
    desc = get_mode_description("plan")
    assert "read-only" in desc.lower() or "plan" in desc.lower()

def test_mode_indicator():
    indicator = get_mode_indicator("careful")
    assert len(indicator) > 0

def test_mode_short():
    short = get_mode_short("careful")
    assert "CARE" in short

def test_auto_approve_edit():
    assert not should_auto_approve_edit("plan")
    assert not should_auto_approve_edit("careful")
    assert should_auto_approve_edit("auto_edit")
    assert should_auto_approve_edit("full_auto")

def test_auto_approve_command():
    assert not should_auto_approve_command("plan")
    assert not should_auto_approve_command("careful")
    assert not should_auto_approve_command("auto_edit")
    assert should_auto_approve_command("full_auto")

def test_block_mutations():
    assert should_block_mutations("plan")
    assert not should_block_mutations("careful")
    assert not should_block_mutations("auto_edit")
    assert not should_block_mutations("full_auto")
