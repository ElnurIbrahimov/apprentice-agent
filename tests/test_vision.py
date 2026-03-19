"""Tests for VisionTool - multi-model fallback, UI analysis, multilingual OCR."""

import base64
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from aura.tools.vision import VisionTool
from aura.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_png(tmp_path):
    """Create a minimal valid PNG file for testing."""
    # 1x1 red pixel PNG
    png_data = (
        b'\x89PNG\r\n\x1a\n'  # signature
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx'
        b'\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00'
        b'\x00\x00\x00IEND\xaeB`\x82'
    )
    p = tmp_path / "test.png"
    p.write_bytes(png_data)
    return str(p)


@pytest.fixture
def tiny_tiff(tmp_path):
    """Create a minimal file with .tiff extension."""
    p = tmp_path / "test.tiff"
    p.write_bytes(b'\x00' * 16)
    return str(p)


@pytest.fixture
def mock_brain():
    """Create a mock brain with an ollama client."""
    brain = MagicMock()
    brain.client = MagicMock()
    brain.client.chat.return_value = {
        'message': {'content': 'A test image description'}
    }
    return brain


@pytest.fixture
def mock_client():
    """Create a mock ollama client."""
    client = MagicMock()
    client.chat.return_value = {
        'message': {'content': 'A test image description'}
    }
    return client


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestVisionToolInit:
    def test_default_init_no_args(self):
        """VisionTool() with no args uses Config model and creates local client."""
        with patch('aura.tools.vision.ollama.Client') as mock_cls:
            tool = VisionTool()
            assert tool.model == Config.get_model("vision")
            assert tool._brain is None
            assert tool._local_client is not None
            mock_cls.assert_called_once_with(host=Config.OLLAMA_HOST)

    def test_init_with_explicit_model(self):
        """VisionTool(model='minicpm-v') uses specified model."""
        with patch('aura.tools.vision.ollama.Client'):
            tool = VisionTool(model="minicpm-v")
            assert tool.model == "minicpm-v"

    def test_init_with_brain(self, mock_brain):
        """VisionTool(brain=brain) reuses brain's client, no local client."""
        tool = VisionTool(brain=mock_brain)
        assert tool._brain is mock_brain
        assert tool._local_client is None

    def test_init_with_brain_and_model(self, mock_brain):
        """VisionTool(model='x', brain=brain) uses both."""
        tool = VisionTool(model="minicpm-v", brain=mock_brain)
        assert tool.model == "minicpm-v"
        assert tool._brain is mock_brain


# ---------------------------------------------------------------------------
# _get_client tests
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_local_client_without_brain(self, mock_client):
        """Without brain, returns local client and unchanged model."""
        with patch('aura.tools.vision.ollama.Client', return_value=mock_client):
            tool = VisionTool()
        client, model = tool._get_client("kimi-k2.5:cloud")
        assert client is mock_client
        assert model == "kimi-k2.5:cloud"

    def test_brain_client_when_brain_provided(self, mock_brain):
        """With brain, returns brain's client."""
        tool = VisionTool(brain=mock_brain)
        client, model = tool._get_client("kimi-k2.5:cloud")
        assert client is mock_brain.client
        assert model == "kimi-k2.5:cloud"

    def test_cloud_suffix_stripped(self, mock_brain):
        """Cloud-suffixed model names with -cloud are stripped to base names."""
        tool = VisionTool(brain=mock_brain)
        client, model = tool._get_client("some-model-cloud")
        assert model == "some-model"

    def test_non_cloud_model_unchanged(self, mock_brain):
        """Non-cloud model names pass through unchanged."""
        tool = VisionTool(brain=mock_brain)
        _, model = tool._get_client("minicpm-v")
        assert model == "minicpm-v"


# ---------------------------------------------------------------------------
# _analyze_with_fallback tests
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_primary_model_succeeds(self, mock_brain):
        """When primary model works, no fallback needed."""
        tool = VisionTool(model="kimi-k2.5:cloud", brain=mock_brain)
        content, model_used = tool._analyze_with_fallback("base64data", "describe")
        assert content == "A test image description"
        assert model_used == "kimi-k2.5:cloud"
        mock_brain.client.chat.assert_called_once()

    def test_fallback_on_primary_failure(self, mock_brain):
        """When primary fails, tries next model in chain."""
        # First call fails, second succeeds
        mock_brain.client.chat.side_effect = [
            Exception("model not found"),
            {'message': {'content': 'Fallback description'}},
        ]
        tool = VisionTool(model="nonexistent-model", brain=mock_brain)
        content, model_used = tool._analyze_with_fallback("base64data", "describe")
        assert content == "Fallback description"
        assert mock_brain.client.chat.call_count == 2

    def test_all_models_fail_raises_runtime_error(self, mock_brain):
        """When all models fail, raises RuntimeError."""
        mock_brain.client.chat.side_effect = Exception("model not found")
        tool = VisionTool(model="bad-model", brain=mock_brain)
        with pytest.raises(RuntimeError, match="All vision models failed"):
            tool._analyze_with_fallback("base64data", "describe")

    def test_chain_deduplication(self, mock_brain):
        """Primary model isn't tried twice if it's also in the chain."""
        mock_brain.client.chat.side_effect = Exception("fail")
        # Use a model that's in MODEL_VISION_CHAIN
        tool = VisionTool(model="kimi-k2.5:cloud", brain=mock_brain)
        with pytest.raises(RuntimeError):
            tool._analyze_with_fallback("base64data", "describe")
        # The primary model should appear only once in the tried list
        models_tried = [
            c.kwargs['model'] if 'model' in c.kwargs else c.args[0]
            for c in mock_brain.client.chat.call_args_list
        ]
        # No model name should appear more than once
        assert len(models_tried) == len(set(models_tried))


# ---------------------------------------------------------------------------
# analyze_image tests
# ---------------------------------------------------------------------------

class TestAnalyzeImage:
    def test_file_not_found(self, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_image("/nonexistent/path.png")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_not_a_file(self, tmp_path, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_image(str(tmp_path))  # directory, not file
        assert result["success"] is False
        assert "not a file" in result["error"]

    def test_unsupported_format(self, tmp_path, mock_brain):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_bytes(b'\x00')
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_image(str(bad_file))
        assert result["success"] is False
        assert "Unsupported" in result["error"]

    def test_tiff_supported(self, tiny_tiff, mock_brain):
        """TIFF files should be accepted as a supported format."""
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_image(tiny_tiff)
        assert result["success"] is True

    def test_successful_analysis(self, tiny_png, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_image(tiny_png, "What is this?")
        assert result["success"] is True
        assert result["description"] == "A test image description"
        assert result["question"] == "What is this?"
        assert "model" in result

    def test_returns_model_used(self, tiny_png, mock_brain):
        """Result dict includes which model actually handled the request."""
        tool = VisionTool(model="kimi-k2.5:cloud", brain=mock_brain)
        result = tool.analyze_image(tiny_png)
        assert result["model"] == "kimi-k2.5:cloud"

    def test_all_models_fail_returns_error(self, tiny_png, mock_brain):
        mock_brain.client.chat.side_effect = Exception("fail")
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_image(tiny_png)
        assert result["success"] is False
        assert "All vision models failed" in result["error"]


# ---------------------------------------------------------------------------
# describe_screen tests
# ---------------------------------------------------------------------------

class TestDescribeScreen:
    def test_delegates_to_analyze_image(self, tiny_png, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.describe_screen(tiny_png)
        assert result["success"] is True
        # Verify the prompt mentions screen/UI
        called_messages = mock_brain.client.chat.call_args[1].get(
            'messages', mock_brain.client.chat.call_args[0][0] if mock_brain.client.chat.call_args[0] else None
        )
        if called_messages is None:
            called_messages = mock_brain.client.chat.call_args.kwargs.get('messages', [])
        question = called_messages[0]['content'] if called_messages else ""
        assert "screen" in question.lower() or "application" in question.lower()


# ---------------------------------------------------------------------------
# read_text tests
# ---------------------------------------------------------------------------

class TestReadText:
    def test_basic_ocr(self, tiny_png, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.read_text(tiny_png)
        assert result["success"] is True

    def test_language_hint_included_in_prompt(self, tiny_png, mock_brain):
        """Language hint modifies the prompt sent to the model."""
        tool = VisionTool(brain=mock_brain)
        tool.read_text(tiny_png, language_hint="japanese")
        # Check that the prompt includes the language hint
        call_args = mock_brain.client.chat.call_args
        messages = call_args.kwargs.get('messages') or call_args[1].get('messages', [])
        prompt = messages[0]['content']
        assert "japanese" in prompt
        assert "non-Latin" in prompt

    def test_no_language_hint_default_prompt(self, tiny_png, mock_brain):
        """Without language hint, prompt doesn't mention non-Latin."""
        tool = VisionTool(brain=mock_brain)
        tool.read_text(tiny_png)
        call_args = mock_brain.client.chat.call_args
        messages = call_args.kwargs.get('messages') or call_args[1].get('messages', [])
        prompt = messages[0]['content']
        assert "non-Latin" not in prompt


# ---------------------------------------------------------------------------
# analyze_ui tests
# ---------------------------------------------------------------------------

class TestAnalyzeUI:
    def test_returns_ui_analysis_key(self, tiny_png, mock_brain):
        """analyze_ui adds a ui_analysis dict to the result."""
        mock_brain.client.chat.return_value = {
            'message': {'content': 'APPLICATION: VS Code\nUI_ELEMENTS:\n- Button at top-right (close)\nTEXT_CONTENT: Hello World\nACTIVE_STATE: Editor tab\nERRORS: None\nSUGGESTED_ACTIONS: Save file'}
        }
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_ui(tiny_png)
        assert result["success"] is True
        assert "ui_analysis" in result
        ui = result["ui_analysis"]
        assert ui["application"] == "VS Code"
        assert len(ui["ui_elements"]) >= 1
        assert "Hello World" in ui["text_content"]

    def test_ui_analysis_with_empty_sections(self, tiny_png, mock_brain):
        """Handles model output with missing sections gracefully."""
        mock_brain.client.chat.return_value = {
            'message': {'content': 'APPLICATION: Chrome\nUI_ELEMENTS:\nTEXT_CONTENT:\nACTIVE_STATE:\nERRORS:\nSUGGESTED_ACTIONS:'}
        }
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_ui(tiny_png)
        assert result["success"] is True
        ui = result["ui_analysis"]
        assert ui["application"] == "Chrome"
        assert ui["ui_elements"] == []

    def test_ui_analysis_parses_numbered_elements(self, tiny_png, mock_brain):
        """Parses numbered list items under UI_ELEMENTS."""
        mock_brain.client.chat.return_value = {
            'message': {'content': (
                'APPLICATION: Firefox\n'
                'UI_ELEMENTS:\n'
                '1. Search bar at top-center (input)\n'
                '2. Back button at top-left (button)\n'
                'TEXT_CONTENT: Google\n'
                'ACTIVE_STATE: Search bar focused\n'
                'ERRORS: None\n'
                'SUGGESTED_ACTIONS: Type a search query'
            )}
        }
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_ui(tiny_png)
        ui = result["ui_analysis"]
        assert len(ui["ui_elements"]) == 2
        assert "Search bar" in ui["ui_elements"][0]

    def test_ui_analysis_failure_passthrough(self, mock_brain):
        """If the image doesn't exist, returns error without ui_analysis."""
        tool = VisionTool(brain=mock_brain)
        result = tool.analyze_ui("/nonexistent.png")
        assert result["success"] is False
        assert "ui_analysis" not in result


# ---------------------------------------------------------------------------
# execute dispatcher tests
# ---------------------------------------------------------------------------

class TestExecuteDispatcher:
    def test_read_action(self, tiny_png, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.execute(f"read text from {tiny_png}")
        assert result["success"] is True

    def test_ocr_action(self, tiny_png, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.execute(f"ocr {tiny_png}")
        assert result["success"] is True

    def test_ui_action(self, tiny_png, mock_brain):
        mock_brain.client.chat.return_value = {
            'message': {'content': 'APPLICATION: Test\nUI_ELEMENTS:\nTEXT_CONTENT:\nACTIVE_STATE:\nERRORS:\nSUGGESTED_ACTIONS:'}
        }
        tool = VisionTool(brain=mock_brain)
        result = tool.execute(f"analyze ui {tiny_png}")
        assert result["success"] is True
        assert "ui_analysis" in result

    def test_dom_action(self, tiny_png, mock_brain):
        mock_brain.client.chat.return_value = {
            'message': {'content': 'APPLICATION: Test\nUI_ELEMENTS:\nTEXT_CONTENT:\nACTIVE_STATE:\nERRORS:\nSUGGESTED_ACTIONS:'}
        }
        tool = VisionTool(brain=mock_brain)
        result = tool.execute(f"dom analysis {tiny_png}")
        assert result["success"] is True
        assert "ui_analysis" in result

    def test_element_action(self, tiny_png, mock_brain):
        mock_brain.client.chat.return_value = {
            'message': {'content': 'APPLICATION: Test\nUI_ELEMENTS:\nTEXT_CONTENT:\nACTIVE_STATE:\nERRORS:\nSUGGESTED_ACTIONS:'}
        }
        tool = VisionTool(brain=mock_brain)
        result = tool.execute(f"find elements in {tiny_png}")
        assert result["success"] is True
        assert "ui_analysis" in result

    def test_screen_action(self, tiny_png, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.execute(f"screen capture {tiny_png}")
        assert result["success"] is True

    def test_default_analyze_action(self, tiny_png, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.execute(f"describe {tiny_png}")
        assert result["success"] is True

    def test_no_image_path_error(self, mock_brain):
        tool = VisionTool(brain=mock_brain)
        result = tool.execute("analyze something")
        assert result["success"] is False
        assert "No image path" in result["error"]

    def test_language_hint_passthrough(self, tiny_png, mock_brain):
        """language_hint kwarg is passed to read_text."""
        tool = VisionTool(brain=mock_brain)
        tool.execute(f"read text {tiny_png}", language_hint="arabic")
        call_args = mock_brain.client.chat.call_args
        messages = call_args.kwargs.get('messages') or call_args[1].get('messages', [])
        prompt = messages[0]['content']
        assert "arabic" in prompt


# ---------------------------------------------------------------------------
# _extract_path tests
# ---------------------------------------------------------------------------

class TestExtractPath:
    def setup_method(self):
        with patch('aura.tools.vision.ollama.Client'):
            self.tool = VisionTool()

    def test_quoted_path(self):
        assert self.tool._extract_path('analyze "C:/images/test.png"') == "C:/images/test.png"

    def test_single_quoted_path(self):
        assert self.tool._extract_path("analyze '/tmp/img.jpg'") == "/tmp/img.jpg"

    def test_unquoted_png(self):
        assert self.tool._extract_path("analyze screenshot.png") == "screenshot.png"

    def test_unquoted_tiff(self):
        assert self.tool._extract_path("analyze scan.tiff") == "scan.tiff"

    def test_unquoted_tif(self):
        assert self.tool._extract_path("analyze scan.tif") == "scan.tif"

    def test_windows_path(self):
        result = self.tool._extract_path("look at C:\\Users\\test\\img")
        assert result is not None
        assert result.startswith("C:\\")

    def test_no_path_returns_none(self):
        assert self.tool._extract_path("just analyze something") is None


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_model_vision_chain_exists(self):
        assert hasattr(Config, 'MODEL_VISION_CHAIN')
        assert isinstance(Config.MODEL_VISION_CHAIN, list)
        assert len(Config.MODEL_VISION_CHAIN) >= 2

    def test_get_model_vision(self):
        result = Config.get_model("vision")
        assert result == Config.MODEL_VISION

    def test_get_model_unknown_returns_default(self):
        result = Config.get_model("nonexistent_role")
        assert result == Config.MODEL_NAME

    def test_chain_contains_cloud_models(self):
        """The chain should contain cloud models we actually have."""
        assert any("cloud" in m for m in Config.MODEL_VISION_CHAIN)
