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
