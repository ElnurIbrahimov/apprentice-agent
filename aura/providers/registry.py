"""Provider configurations — base URLs, env vars, and default model lists."""

PROVIDER_CONFIGS = {
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "env_var": "ANTHROPIC_API_KEY",
        "display_name": "Anthropic",
        "default_models": [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-5",
        ],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
        "display_name": "OpenAI",
        "default_models": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o4-mini",
            "o3",
        ],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_var": "GEMINI_API_KEY",
        "display_name": "Google Gemini",
        "default_models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
    },
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "env_var": "GROK_API_KEY",
        "display_name": "xAI (Grok)",
        "default_models": [
            "grok-3",
            "grok-3-mini",
            "grok-3-fast",
        ],
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "env_var": "PERPLEXITY_API_KEY",
        "display_name": "Perplexity",
        "default_models": [
            "sonar-pro",
            "sonar",
            "sonar-reasoning-pro",
        ],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_var": "DEEPSEEK_API_KEY",
        "display_name": "DeepSeek",
        "default_models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
    "minimax": {
        "base_url": "https://api.minimax.chat/v1",
        "env_var": "MINIMAX_API_KEY",
        "display_name": "MiniMax",
        "default_models": [
            "MiniMax-M1-80k",
        ],
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_var": "QWEN_API_KEY",
        "display_name": "Qwen (Alibaba)",
        "default_models": [
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
        ],
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "env_var": "KIMI_API_KEY",
        "display_name": "Kimi (Moonshot)",
        "default_models": [
            "moonshot-v1-128k",
            "moonshot-v1-32k",
            "moonshot-v1-8k",
        ],
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_var": "GLM_API_KEY",
        "display_name": "GLM (Zhipu)",
        "default_models": [
            "glm-4-plus",
            "glm-4-flash",
            "glm-4-long",
        ],
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "env_var": "MISTRAL_API_KEY",
        "display_name": "Mistral AI",
        "default_models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "codestral-latest",
        ],
    },
    "cohere": {
        "base_url": "https://api.cohere.ai/v2",
        "env_var": "COHERE_API_KEY",
        "display_name": "Cohere",
        "default_models": [
            "command-r-plus",
            "command-r",
        ],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_var": "GROQ_API_KEY",
        "display_name": "Groq",
        "default_models": [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ],
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_var": "TOGETHER_API_KEY",
        "display_name": "Together AI",
        "default_models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        ],
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_var": "FIREWORKS_API_KEY",
        "display_name": "Fireworks AI",
        "default_models": [
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
        ],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_var": "OPENROUTER_API_KEY",
        "display_name": "OpenRouter",
        "default_models": [
            "anthropic/claude-sonnet-4",
            "openai/gpt-4.1",
            "google/gemini-2.5-flash",
        ],
    },
}

# Providers that use OpenAI-compatible API format
OPENAI_COMPATIBLE_PROVIDERS = [
    "openai", "grok", "perplexity", "deepseek",
    "minimax", "qwen", "kimi", "glm",
    "mistral", "cohere", "groq", "together", "fireworks", "openrouter",
]
