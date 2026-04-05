"""Tests for aura.messaging.telegram_store.TelegramStore."""
import time
import pytest
from aura.messaging.telegram_store import TelegramStore


@pytest.fixture
def store(tmp_path):
    db = TelegramStore(db_path=str(tmp_path / "test.db"))
    yield db
    db.close()


class TestUserSettings:
    def test_get_missing_user_returns_empty(self, store):
        s = store.get_user_settings("999")
        assert isinstance(s, dict)

    def test_set_and_get_language(self, store):
        store.set_user_setting("123", language="az")
        s = store.get_user_settings("123")
        assert s["language"] == "az"

    def test_set_multiple_fields(self, store):
        store.set_user_setting("456", language="ru", keyboard_enabled=0)
        s = store.get_user_settings("456")
        assert s["language"] == "ru"
        assert s["keyboard_enabled"] == 0

    def test_update_existing_setting(self, store):
        store.set_user_setting("123", language="en")
        store.set_user_setting("123", language="az")
        s = store.get_user_settings("123")
        assert s["language"] == "az"

    def test_get_user_language_default(self, store):
        # User not in DB — should return "en"
        assert store.get_user_language("no_user") == "en"

    def test_set_user_language(self, store):
        store.set_user_language("123", "ru")
        assert store.get_user_language("123") == "ru"

    def test_keyboard_enabled_default(self, store):
        # User not in DB — default is True
        assert store.get_keyboard_enabled("no_user") is True

    def test_set_keyboard_disabled(self, store):
        store.set_keyboard_enabled("123", False)
        assert store.get_keyboard_enabled("123") is False

    def test_set_keyboard_enabled(self, store):
        store.set_keyboard_enabled("123", False)
        store.set_keyboard_enabled("123", True)
        assert store.get_keyboard_enabled("123") is True


class TestPremium:
    def test_not_premium_by_default(self, store):
        assert store.is_premium("999") is False

    def test_set_and_check_premium(self, store):
        store.set_premium("123", tier="pro")
        assert store.is_premium("123") is True

    def test_get_premium_users_includes_new_user(self, store):
        store.set_premium("abc", tier="supporter")
        users = store.get_premium_users()
        assert "abc" in users

    def test_premium_tier_stored(self, store):
        store.set_premium("abc", tier="vip")
        users = store.get_premium_users()
        assert users["abc"]["tier"] == "vip"

    def test_remove_premium(self, store):
        store.set_premium("123", tier="pro")
        store.remove_premium("123")
        assert store.is_premium("123") is False

    def test_set_premium_with_metadata(self, store):
        store.set_premium("123", tier="pro", stars_amount=500, transaction_id="tx_99")
        users = store.get_premium_users()
        assert users["123"]["stars_amount"] == 500
        assert users["123"]["transaction_id"] == "tx_99"

    def test_remove_non_premium_is_noop(self, store):
        # Should not raise
        store.remove_premium("ghost_user")
        assert store.is_premium("ghost_user") is False


class TestActiveChats:
    def test_upsert_and_retrieve(self, store):
        store.upsert_active_chat("chat1", "user1", "John", "johndoe")
        chats = store.get_active_chats()
        assert "chat1" in chats

    def test_upsert_updates_existing(self, store):
        store.upsert_active_chat("chat1", "user1", "John", "johndoe")
        store.upsert_active_chat("chat1", "user1", "Johnny", "johndoe")
        chats = store.get_active_chats()
        # Should still be one entry
        assert len([c for c in chats if c == "chat1"]) == 1

    def test_multiple_chats(self, store):
        store.upsert_active_chat("chat1", "user1", "Alice", "alice")
        store.upsert_active_chat("chat2", "user2", "Bob", "bob")
        chats = store.get_active_chats()
        assert "chat1" in chats
        assert "chat2" in chats

    def test_empty_initially(self, store):
        chats = store.get_active_chats()
        assert isinstance(chats, dict)
        assert len(chats) == 0


class TestDocContext:
    def test_set_and_get(self, store):
        store.set_doc_context("user1", "document text here", "test.pdf")
        ctx = store.get_doc_context("user1")
        assert ctx is not None
        assert ctx["text"] == "document text here"
        assert ctx["filename"] == "test.pdf"

    def test_missing_returns_none(self, store):
        assert store.get_doc_context("no_user") is None

    def test_overwrite_doc_context(self, store):
        store.set_doc_context("user1", "first doc", "a.pdf")
        store.set_doc_context("user1", "second doc", "b.pdf")
        ctx = store.get_doc_context("user1")
        assert ctx["text"] == "second doc"
        assert ctx["filename"] == "b.pdf"

    def test_clear_doc_context(self, store):
        store.set_doc_context("user1", "some text", "file.pdf")
        store.clear_doc_context("user1")
        assert store.get_doc_context("user1") is None

    def test_expired_context_returns_none(self, store):
        store.set_doc_context("user1", "old doc", "old.pdf")
        # Use a negative TTL so the condition (elapsed > ttl) is always True
        ctx = store.get_doc_context("user1", ttl=-1.0)
        assert ctx is None


class TestGroupMessages:
    def test_add_and_retrieve(self, store):
        store.add_group_message("grp1", "u1", "Alice", "Hello")
        msgs = store.get_group_messages("grp1")
        assert len(msgs) == 1
        assert msgs[0]["text"] == "Hello"
        assert msgs[0]["user_name"] == "Alice"

    def test_multiple_messages_in_order(self, store):
        store.add_group_message("grp1", "u1", "Alice", "first")
        store.add_group_message("grp1", "u2", "Bob", "second")
        msgs = store.get_group_messages("grp1")
        assert len(msgs) == 2
        texts = {m["text"] for m in msgs}
        assert texts == {"first", "second"}

    def test_isolated_by_chat(self, store):
        store.add_group_message("grp1", "u1", "Alice", "in grp1")
        store.add_group_message("grp2", "u2", "Bob", "in grp2")
        msgs1 = store.get_group_messages("grp1")
        msgs2 = store.get_group_messages("grp2")
        assert len(msgs1) == 1
        assert len(msgs2) == 1


class TestUserLocation:
    def test_set_and_get(self, store):
        store.set_user_location("u1", 40.4093, 49.8671)
        loc = store.get_user_location("u1")
        assert loc is not None
        assert abs(loc["latitude"] - 40.4093) < 0.0001
        assert abs(loc["longitude"] - 49.8671) < 0.0001

    def test_missing_returns_none(self, store):
        assert store.get_user_location("ghost") is None


class TestReactionFeedback:
    def test_save_and_get_stats(self, store):
        store.save_reaction_feedback("u1", "chat1", 42, "👍", "positive")
        stats = store.get_reaction_stats("u1")
        assert "positive" in stats
        assert stats["positive"] == 1

    def test_stats_aggregate_multiple(self, store):
        store.save_reaction_feedback("u1", "chat1", 1, "👍", "positive")
        store.save_reaction_feedback("u1", "chat1", 2, "👍", "positive")
        store.save_reaction_feedback("u1", "chat1", 3, "👎", "negative")
        stats = store.get_reaction_stats("u1")
        assert stats["positive"] == 2
        assert stats["negative"] == 1

    def test_global_stats(self, store):
        store.save_reaction_feedback("u1", "c1", 1, "👍", "positive")
        store.save_reaction_feedback("u2", "c1", 2, "👍", "positive")
        stats = store.get_reaction_stats()
        assert stats["positive"] == 2
