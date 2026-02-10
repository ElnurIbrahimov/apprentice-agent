# OpenAI Whisper - Speech to Text

## Overview
Whisper is used for audio/video transcription in AURA via the AudioTranscriberTool.

## Model Sizes
| Size | Parameters | English-only | Multilingual | VRAM |
|------|-----------|-------------|-------------|------|
| tiny | 39M | Yes | Yes | ~1GB |
| base | 74M | Yes | Yes | ~1GB |
| small | 244M | Yes | Yes | ~2GB |
| medium | 769M | Yes | Yes | ~5GB |
| large | 1550M | No | Yes | ~10GB |

## AURA Default: `base` (good speed/accuracy tradeoff)

## Features Used
- **Transcribe**: Audio/video to text with timestamps
- **Translate**: Any language to English
- **Language Detection**: Identify spoken language from audio snippet

## Supported Formats
- Audio: .mp3, .wav, .flac, .ogg, .m4a, .aac, .wma, .opus, .webm
- Video: .mp4, .mkv, .avi, .mov, .wmv, .flv, .webm (requires ffmpeg)

## Dependencies
- `openai-whisper` (pip install)
- `ffmpeg` (system install, required for video files)

## Implementation Pattern
- Lazy-loaded model singleton (`_whisper_model`)
- Transcripts saved to `data/transcripts/` with timestamps
- Segment-level timestamps for precise navigation
