# Ollama Integration Research

## Overview
AURA uses Ollama as its primary LLM backend, running models locally for privacy and speed.

## Brain Architecture (`brain.py`)
- Uses `ollama.chat()` for all LLM calls
- Supports streaming responses
- Configurable model via `Config.OLLAMA_MODEL`
- Default timeout: 60s for LLM calls, 10s for warmup

## Models Used
- **Primary**: Whatever model is configured (typically llama3, mistral, qwen2.5)
- **Vision**: Florence-2 (via HuggingFace transformers) for image analysis
- **TTS**: Sesame CSM (Conversational Speech Model) for voice synthesis
- **STT**: OpenAI Whisper for speech-to-text
- **Embeddings**: sentence-transformers for semantic search in memory systems

## Neuromodulator System
Brain parameters dynamically adjusted based on ALMA emotional state:
- **temperature**: 0.7 base, modulated by dopamine (creativity)
- **top_p**: 0.9 base, modulated by norepinephrine (focus)
- **num_predict**: 1024 base, modulated by serotonin (verbosity)
- All bounded by safety multipliers (0.7x - 1.4x of defaults)

## Prompt Engineering
- System prompt includes: identity, tool descriptions, current context
- Action selection prompt format: `TOOL: <name>\nACTION: <what to do>\nREASONING: <why>`
- Tool descriptions are dynamically generated based on loaded tools
- `_parse_action_response()` handles messy local model outputs with multiple fallbacks

## Key Functions
- `brain.decide_action()` - Main tool selection
- `brain.chat()` - Direct conversation
- `brain.evaluate_result()` - Check if tool result answers query
- `brain._clean_action()` - Strip markdown, quotes, prefixes from LLM output

## Challenges with Local Models
- Output format inconsistency (local models don't always follow TOOL:/ACTION: format)
- Multiple fallback parsing strategies needed
- Code extraction requires special handling (triple backticks, indentation)
- Tool name normalization (models say "file system" instead of "filesystem")
