"""Test script for hybrid voice system."""

import os
os.environ['TORCHAO_NO_TRITON'] = '1'
# HF_TOKEN must be set in environment for PersonaPlex
if not os.getenv('HF_TOKEN'):
    print("Warning: HF_TOKEN not set. Some features may not work.")

import sys
from pathlib import Path

# Add the parent directory to enable imports
sys.path.insert(0, str(Path(__file__).parent / 'apprentice_agent' / 'tools'))
sys.path.insert(0, str(Path.home() / 'sesame-csm'))

from sesame_tts import SesameTTS

# Import VoiceManager components manually to avoid relative import issues
import torch
import requests

class VoiceManagerTest:
    """Simplified VoiceManager for testing."""

    PIPELINE_KEYWORDS = [
        "read this", "say this", "speak", "read aloud", "read it",
        "tell me", "announce", "narrate", "pronounce"
    ]
    DUPLEX_KEYWORDS = [
        "talk to me", "let's chat", "voice chat", "conversation mode",
        "speak with me", "have a conversation", "real-time chat"
    ]

    def __init__(self):
        self.current_mode = None
        self.sesame = SesameTTS()

    def route_voice(self, user_input):
        lower = user_input.lower()
        if any(kw in lower for kw in self.DUPLEX_KEYWORDS):
            return "duplex"
        if any(kw in lower for kw in self.PIPELINE_KEYWORDS):
            return "pipeline"
        return None

    def status(self):
        vram_free = "N/A"
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram_free = f"{free / 1024**3:.1f}GB"
        return {
            "mode": self.current_mode,
            "sesame_loaded": self.sesame.is_loaded(),
            "vram_free": vram_free
        }

VoiceManager = VoiceManagerTest

def main():
    print("=" * 60)
    print("Hybrid Voice System Test")
    print("=" * 60)

    # Test SesameTTS
    print("\n1. Testing SesameTTS...")
    tts = SesameTTS()
    status = tts.status()
    print(f"   Device: {status['device']}")
    print(f"   VRAM Free: {status['vram_free']}")
    print(f"   Loaded: {status['loaded']}")

    # Test VoiceManager
    print("\n2. Testing VoiceManager...")
    vm = VoiceManager()
    vm_status = vm.status()
    print(f"   Current Mode: {vm_status['mode']}")
    print(f"   Sesame Loaded: {vm_status['sesame_loaded']}")
    print(f"   VRAM Free: {vm_status['vram_free']}")

    # Test keyword routing
    print("\n3. Testing keyword routing...")
    test_phrases = [
        "read this document aloud",
        "let's have a voice chat",
        "tell me about the weather",
        "speak with me in real-time",
    ]
    for phrase in test_phrases:
        mode = vm.route_voice(phrase)
        print(f"   '{phrase[:30]}...' -> {mode or 'no voice action'}")

    print("\n" + "=" * 60)
    print("To test actual TTS generation:")
    print("  1. Pipeline mode (Sesame): vm.switch_to_pipeline()")
    print("  2. Speak: vm.speak('Hello from Aura!')")
    print("  3. Duplex mode (PersonaPlex): vm.switch_to_duplex()")
    print("=" * 60)

if __name__ == "__main__":
    main()
