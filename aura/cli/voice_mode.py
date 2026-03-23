def run_voice_mode(agent, enable_barge_in: bool = True):
    from aura.tools.voice import VoiceConversation
    conversation = VoiceConversation(agent, whisper_model="base", enable_barge_in=enable_barge_in)
    conversation.start()
