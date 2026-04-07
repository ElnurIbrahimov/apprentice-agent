import pytest
import json


class TestModelProfiles:
    def test_load_profiles_from_json(self, tmp_path):
        data = {
            "nemotron-3-super:cloud": {
                "code": 0.60, "reason": 0.73, "speed": 1.00,
                "context": 1.00, "quality": 0.70, "vision": 0,
            }
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        p = store.get("nemotron-3-super:cloud")
        assert p["speed"] == 1.00
        assert p["code"] == 0.60

    def test_get_unknown_model_returns_neutral(self, tmp_path):
        path = tmp_path / "profiles.json"
        path.write_text("{}")
        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        p = store.get("unknown-model:cloud")
        assert p["code"] == 0.5
        assert p["speed"] == 0.5

    def test_update_profile_clamps_values(self, tmp_path):
        data = {"test:cloud": {"code": 0.95, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0}}
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        store.update("test:cloud", {"code": 0.1})
        assert store.get("test:cloud")["code"] == 1.0

    def test_update_profile_persists(self, tmp_path):
        data = {"test:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0}}
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        store.update("test:cloud", {"code": 0.05})
        store2 = ProfileStore(str(path))
        assert store2.get("test:cloud")["code"] == 0.55

    def test_list_available_filters_by_dimension(self, tmp_path):
        data = {
            "model-a:cloud": {"code": 0.9, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 1},
            "model-b:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0},
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        from aura.routing.profiles import ProfileStore
        store = ProfileStore(str(path))
        vision_models = store.list_available(require={"vision": 0.5})
        assert "model-a:cloud" in vision_models
        assert "model-b:cloud" not in vision_models


class TestClassifier:
    def test_extract_features_short_message(self):
        from aura.routing.classifier import extract_features
        f = extract_features("hello how are you")
        assert f["word_count"] == 4
        assert f["code_ratio"] == 0.0
        assert f["has_attachment"] is False

    def test_extract_features_code_message(self):
        from aura.routing.classifier import extract_features
        f = extract_features("fix the bug in def calculate(x): return x * 2")
        assert f["code_ratio"] > 0.1
        assert len(f["language_markers"]) > 0

    def test_extract_features_with_attachment(self):
        from aura.routing.classifier import extract_features
        f = extract_features("analyze this image", has_attachment=True)
        assert f["has_attachment"] is True

    def test_score_task_short_query_high_speed(self):
        from aura.routing.classifier import score_task
        needs = score_task("hi there")
        assert needs["speed"] >= 0.8
        assert needs["reason"] < 0.3

    def test_score_task_code_query(self):
        from aura.routing.classifier import score_task
        needs = score_task("debug this Python function that throws TypeError")
        assert needs["code"] >= 0.7

    def test_score_task_complex_reasoning(self):
        from aura.routing.classifier import score_task
        needs = score_task("write a comprehensive analysis of the pros and cons of microservices vs monolithic architecture for a startup")
        assert needs["reason"] >= 0.7
        assert needs["speed"] <= 0.4

    def test_score_task_vision(self):
        from aura.routing.classifier import score_task
        needs = score_task("what's in this screenshot", has_attachment=True)
        assert needs["vision"] >= 0.9

    def test_score_task_long_context(self):
        from aura.routing.classifier import score_task
        needs = score_task("continue", conversation_tokens=120_000)
        assert needs["context"] >= 0.7

    def test_score_task_after_regen(self):
        from aura.routing.classifier import score_task
        needs = score_task("try again", recent_regen_count=2)
        assert needs["quality"] >= 0.8


class TestMatcher:
    def _make_store(self, tmp_path):
        import json
        from aura.routing.profiles import ProfileStore
        data = {
            "fast-model:cloud": {"code": 0.3, "reason": 0.4, "speed": 1.0, "context": 0.5, "quality": 0.5, "vision": 0},
            "code-model:cloud": {"code": 0.95, "reason": 0.5, "speed": 0.2, "context": 0.5, "quality": 0.8, "vision": 0},
            "vision-model:cloud": {"code": 0.5, "reason": 0.7, "speed": 0.3, "context": 0.5, "quality": 0.8, "vision": 1},
            "allround-model:cloud": {"code": 0.7, "reason": 0.8, "speed": 0.5, "context": 0.5, "quality": 0.85, "vision": 0},
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        return ProfileStore(str(path))

    def test_match_speed_query_picks_fast_model(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.0, "reason": 0.2, "speed": 0.9, "context": 0.0, "quality": 0.3, "vision": 0.0}
        result = match(task_needs, "balanced", store)
        assert result == "fast-model:cloud"

    def test_match_code_query_picks_code_model(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.9, "reason": 0.3, "speed": 0.2, "context": 0.0, "quality": 0.8, "vision": 0.0}
        result = match(task_needs, "balanced", store)
        assert result == "code-model:cloud"

    def test_match_vision_filters_non_vision(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.0, "reason": 0.5, "speed": 0.3, "context": 0.0, "quality": 0.5, "vision": 1.0}
        result = match(task_needs, "balanced", store)
        assert result == "vision-model:cloud"

    def test_prefer_fast_boosts_speed(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.0, "quality": 0.5, "vision": 0.0}
        result_fast = match(task_needs, "prefer-fast", store)
        result_quality = match(task_needs, "prefer-quality", store)
        assert result_fast == "fast-model:cloud"
        assert result_quality != "fast-model:cloud"

    def test_match_returns_string(self, tmp_path):
        from aura.routing.matcher import match
        store = self._make_store(tmp_path)
        task_needs = {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.0, "quality": 0.5, "vision": 0.0}
        result = match(task_needs, "balanced", store)
        assert isinstance(result, str)
        assert result.endswith(":cloud")


class TestConversationTracker:
    def test_new_conversation_returns_empty_profile(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        profile = tracker.get_profile("conv-1")
        assert profile.turn_count == 0
        assert profile.in_code_mode is False

    def test_update_increments_turn_count(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        tracker.update("conv-1", code_ratio=0.0, complexity=0.3, model_used="test:cloud", tokens=100)
        p = tracker.get_profile("conv-1")
        assert p.turn_count == 1

    def test_code_mode_activates_after_3_code_messages(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        for _ in range(3):
            tracker.update("conv-1", code_ratio=0.5, complexity=0.5, model_used="test:cloud", tokens=200)
        p = tracker.get_profile("conv-1")
        assert p.in_code_mode is True

    def test_code_mode_deactivates(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        for _ in range(3):
            tracker.update("conv-1", code_ratio=0.5, complexity=0.5, model_used="test:cloud", tokens=200)
        for _ in range(3):
            tracker.update("conv-1", code_ratio=0.0, complexity=0.3, model_used="test:cloud", tokens=100)
        p = tracker.get_profile("conv-1")
        assert p.in_code_mode is False

    def test_complexity_trend_escalating(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        for i in range(5):
            tracker.update("conv-1", code_ratio=0.0, complexity=0.2 + i * 0.15, model_used="test:cloud", tokens=200)
        p = tracker.get_profile("conv-1")
        assert p.complexity_trend > 0.2

    def test_record_regen(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        tracker.record_feedback("conv-1", "regeneration")
        p = tracker.get_profile("conv-1")
        assert p.regen_count == 1

    def test_record_model_switch(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        tracker.record_feedback("conv-1", "model_switch", model="new:cloud")
        p = tracker.get_profile("conv-1")
        assert p.model_switches == 1

    def test_adjust_boosts_code_in_code_mode(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        for _ in range(3):
            tracker.update("conv-1", code_ratio=0.5, complexity=0.5, model_used="test:cloud", tokens=200)
        task_needs = {"code": 0.1, "reason": 0.3, "speed": 0.5, "context": 0.0, "quality": 0.5, "vision": 0.0}
        adjusted = tracker.adjust("conv-1", task_needs)
        assert adjusted["code"] > task_needs["code"]

    def test_adjust_boosts_quality_after_regen(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker()
        tracker.record_feedback("conv-1", "regeneration")
        task_needs = {"code": 0.0, "reason": 0.3, "speed": 0.5, "context": 0.0, "quality": 0.5, "vision": 0.0}
        adjusted = tracker.adjust("conv-1", task_needs)
        assert adjusted["quality"] > task_needs["quality"]

    def test_stale_conversations_evicted(self):
        from aura.routing.conversation import ConversationTracker
        tracker = ConversationTracker(max_conversations=2)
        tracker.update("conv-1", code_ratio=0.0, complexity=0.3, model_used="a:cloud", tokens=100)
        tracker.update("conv-2", code_ratio=0.0, complexity=0.3, model_used="a:cloud", tokens=100)
        tracker.update("conv-3", code_ratio=0.0, complexity=0.3, model_used="a:cloud", tokens=100)
        p = tracker.get_profile("conv-1")
        assert p.turn_count == 0


class TestLearning:
    def _make_store(self, tmp_path):
        import json
        from aura.routing.profiles import ProfileStore
        data = {"test:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0}}
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        return ProfileStore(str(path))

    def test_positive_rating_boosts_profile(self, tmp_path):
        from aura.routing.learning import process_feedback
        store = self._make_store(tmp_path)
        task_dims = {"code": 0.8, "reason": 0.2, "speed": 0.0, "context": 0.0, "quality": 0.5, "vision": 0.0}
        process_feedback("thumbs_up", "test:cloud", task_dims, store)
        p = store.get("test:cloud")
        assert p["code"] > 0.5
        assert p["reason"] == 0.5  # below threshold, untouched

    def test_regeneration_penalizes_profile(self, tmp_path):
        from aura.routing.learning import process_feedback
        store = self._make_store(tmp_path)
        task_dims = {"code": 0.8, "reason": 0.2, "speed": 0.0, "context": 0.0, "quality": 0.5, "vision": 0.0}
        process_feedback("regeneration", "test:cloud", task_dims, store)
        p = store.get("test:cloud")
        assert p["code"] < 0.5

    def test_model_switch_updates_both(self, tmp_path):
        import json
        from aura.routing.profiles import ProfileStore
        from aura.routing.learning import process_feedback
        data = {
            "old:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0},
            "new:cloud": {"code": 0.5, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0},
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        store = ProfileStore(str(path))
        task_dims = {"code": 0.8, "reason": 0.2, "speed": 0.0, "context": 0.0, "quality": 0.5, "vision": 0.0}
        process_feedback("model_switch", "old:cloud", task_dims, store, switched_to="new:cloud")
        assert store.get("old:cloud")["code"] < 0.5
        assert store.get("new:cloud")["code"] > 0.5

    def test_feedback_respects_clamp(self, tmp_path):
        import json
        from aura.routing.profiles import ProfileStore
        from aura.routing.learning import process_feedback
        data = {"test:cloud": {"code": 0.02, "reason": 0.5, "speed": 0.5, "context": 0.5, "quality": 0.5, "vision": 0}}
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        store = ProfileStore(str(path))
        task_dims = {"code": 0.9, "reason": 0.0, "speed": 0.0, "context": 0.0, "quality": 0.0, "vision": 0.0}
        process_feedback("regeneration", "test:cloud", task_dims, store)
        assert store.get("test:cloud")["code"] >= 0.0


class TestRouter:
    def _make_router(self, tmp_path):
        import json
        from aura.routing.router import Router
        data = {
            "fast:cloud": {"code": 0.3, "reason": 0.4, "speed": 1.0, "context": 0.5, "quality": 0.5, "vision": 0},
            "code:cloud": {"code": 0.95, "reason": 0.5, "speed": 0.2, "context": 0.5, "quality": 0.8, "vision": 0},
            "vision:cloud": {"code": 0.5, "reason": 0.7, "speed": 0.3, "context": 0.5, "quality": 0.8, "vision": 1},
        }
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps(data))
        return Router(profiles_path=str(path))

    def test_explicit_model_bypasses_router(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("anything", model="my-explicit:cloud")
        assert result.model == "my-explicit:cloud"
        assert result.reason == "explicit_override"

    def test_auto_routes_short_to_fast(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("hello")
        assert result.model == "fast:cloud"

    def test_auto_routes_code_to_code_model(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("debug this Python function: def foo(): pass")
        assert result.model == "code:cloud"

    def test_auto_routes_vision_with_attachment(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("what is this", has_attachment=True)
        assert result.model == "vision:cloud"

    def test_preference_affects_result(self, tmp_path):
        router = self._make_router(tmp_path)
        fast_result = router.route("explain something moderately complex to me", preference="prefer-fast")
        quality_result = router.route("explain something moderately complex to me", preference="prefer-quality")
        assert fast_result.model in ("fast:cloud", "code:cloud", "vision:cloud")
        assert quality_result.model in ("fast:cloud", "code:cloud", "vision:cloud")

    def test_route_returns_metadata(self, tmp_path):
        router = self._make_router(tmp_path)
        result = router.route("hello there")
        assert hasattr(result, "model")
        assert hasattr(result, "reason")
        assert hasattr(result, "task_dimensions")
        assert hasattr(result, "alternatives")

    def test_conversation_context_influences_routing(self, tmp_path):
        router = self._make_router(tmp_path)
        conv = "test-conv"
        for _ in range(3):
            router.route("fix def calc(): return x * 2", conversation_id=conv)
        result = router.route("now test it", conversation_id=conv)
        assert result.model == "code:cloud"
