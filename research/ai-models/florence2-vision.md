# Florence-2 Vision Model

## Overview
Microsoft's Florence-2 is used for all vision tasks in AURA — image analysis, OCR, object detection, captioning.

## Model Details
- **Model**: `microsoft/Florence-2-large`
- **Size**: ~1.5GB
- **Tasks**: OCR, captioning, object detection, dense region captioning, referring expression
- **Loading**: Lazy-loaded singleton pattern in `vision.py`

## Task Types Used
| Task | Florence-2 Tag | Use Case |
|------|---------------|----------|
| OCR | `<OCR>` | Screen reader, text extraction |
| Caption | `<CAPTION>` | Image description |
| Detailed Caption | `<DETAILED_CAPTION>` | Rich image analysis |
| Object Detection | `<OD>` | Finding objects in images |
| Dense Region Caption | `<DENSE_REGION_CAPTION>` | Region-by-region descriptions |

## Implementation
```python
# Lazy loading pattern
_florence_model = None
_florence_processor = None

def _load_florence():
    global _florence_model, _florence_processor
    from transformers import AutoProcessor, AutoModelForCausalLM
    _florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    _florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
```

## Used By
- `VisionTool` - Primary image analysis
- `ScreenReaderTool` - OCR from screenshots
- Screen monitoring / change detection

## Performance
- Fast inference on CPU (~2-5s per image)
- GPU acceleration available if CUDA present
- OCR quality is good for printed text, acceptable for handwriting
