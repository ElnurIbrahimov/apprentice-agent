# MirrorMind - Self-Critique System

## Overview
Before AURA sends a response, MirrorMind reviews it for quality, accuracy, completeness, and tone. Acts as an internal editor.

## Critique Dimensions
- **Accuracy**: Is the information correct?
- **Completeness**: Does it fully answer the question?
- **Clarity**: Is it easy to understand?
- **Tone**: Does it match the emotional context?
- **Relevance**: Does it stay on topic?

## How It Works
1. Response is generated
2. MirrorMind evaluates against critique dimensions
3. Returns `CritiqueResult` with scores and suggestions
4. If score is below threshold, response is revised

## Key Classes
- `MirrorMind` - Main critique engine
- `CritiqueResult` - Evaluation outcome with scores

## File Location
- `aura/tools/mirrormind.py`
