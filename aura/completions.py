"""Shell completion script generation for Aura CLI.

Mirrors Hermes Agent's `hermes completion bash|zsh` pattern.

Generates shell completion scripts that register aura as a command
with auto-completion for subcommands and options.
"""
from __future__ import annotations



# All aura subcommands
_SUBCOMMANDS = [
    "init", "setup", "doctor", "config", "models", "commit", "cost",
    "mcp-serve", "acp-serve", "exec", "ide", "log", "status", "recall",
    "start", "stop", "why", "heatmap", "worktree",
    "profile", "auth", "tools", "skills", "cron", "sessions", "insights",
]

# All aura flags
_FLAGS = [
    "--version", "-V", "--resume", "-r", "--continue", "-c",
    "--profile", "--worktree", "-w", "--skills", "-s", "--yolo",
    "--tier", "--budget", "--trust", "--model", "--format",
    "-v", "--verbose", "-q", "--quiet", "--routing-trace",
    "--sandboxed", "--workspace-write", "--unrestricted",
    "--max-iterations", "--dream", "--dream-date",
    "--no-fastpath", "--fast", "--voice", "--speak",
    "--no-barge-in", "--resume", "-p", "--prompt",
    "--login", "--logout", "--preference", "--mode",
    "--all", "-a", "--by-model", "--by-provider",
    "--session", "--limit", "--format", "--clone",
    "--remove", "--force", "--open", "--list",
    "--branch", "--timeout", "--output-failures",
]


def generate_bash_completion() -> str:
    """Generate a bash completion script for aura."""
    cmds = " ".join(f'"{c}"' for c in _SUBCOMMANDS)
    flags = " ".join(f'"{f}"' for f in _FLAGS)

    return f"""# Aura CLI bash completion
# Source this file or add to ~/.bashrc:
#   source <(aura completion bash)

_aura_completion() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

    # Subcommand completion for the first positional
    if [ $COMP_CWORD -eq 1 ]; then
        opts="{cmds}"
        COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
        return 0
    fi

    # Flag completion
    opts="{flags}"
    COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
    return 0
}}

complete -F _aura_completion aura
"""


def generate_zsh_completion() -> str:
    """Generate a zsh completion script for aura."""
    cmds = "\n      ".join(f'"{c}:{c} subcommand"' for c in _SUBCOMMANDS)

    return f"""# Aura CLI zsh completion
# Source this file or add to ~/.zshrc:
#   source <(aura completion zsh)

#compdef aura

_aura() {{
    local -a subcommands
    subcommands=(
      {cmds}
    )

    _arguments -C \\
        '1: :->subcommand' \\
        '*:: :->args'

    case $state in
        subcommand)
            _describe 'aura subcommand' subcommands
            ;;
    esac
}}

_aura "$@"
"""


def generate_powershell_completion() -> str:
    """Generate a PowerShell completion script for aura."""
    cmds = ", ".join(f"'{c}'" for c in _SUBCOMMANDS)

    return f"""# Aura CLI PowerShell completion
# Add to your PowerShell profile:
#   aura completion powershell | Out-String | Invoke-Expression

Register-ArgumentCompleter -Native -CommandName 'aura' -ScriptBlock {{
    param($wordToComplete, $commandAst, $cursorPosition)

    $subcommands = @({cmds})

    $subcommands | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }}
}}
"""


def generate_completion(shell: str) -> str:
    """Generate completion script for the given shell."""
    shell = shell.lower().strip()
    if shell == "bash":
        return generate_bash_completion()
    if shell == "zsh":
        return generate_zsh_completion()
    if shell in ("powershell", "pwsh"):
        return generate_powershell_completion()
    raise ValueError(f"Unsupported shell: {shell}. Use bash, zsh, or powershell.")
