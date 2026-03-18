"""Interactive model picker for AURA CLI — arrow-key navigable, shows ALL models."""

import os

# Model roles with display info (updated from Config on startup)
MODEL_ROLES = [
    ("fast", "gemini-3-flash-preview:cloud", "1M ctx"),
    ("reason", "kimi-k2.5:cloud", "256K ctx"),
    ("code", "minimax-m2.7:cloud", "1M ctx, SWE-Pro 56.2%"),
    ("think", "kimi-k2-thinking:cloud", "256K ctx"),
    ("vision", "qwen3-vl:235b-cloud", "256K ctx"),
    ("longctx", "minimax-m2.7:cloud", "1M ctx, self-evolving"),
]

# Cache for all available models
_all_models_cache = []


def _fetch_all_models() -> list:
    """Fetch all available models from Ollama."""
    global _all_models_cache
    if _all_models_cache:
        return _all_models_cache
    try:
        import requests
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            _all_models_cache = [m["name"] for m in models]
    except Exception:
        pass
    return _all_models_cache


def _fetch_chatgpt_models() -> list:
    """Get available ChatGPT models if authenticated."""
    try:
        from aura.auth.chatgpt_oauth import is_authenticated
        if is_authenticated():
            from aura.auth.chatgpt_client import ALL_CHATGPT_MODELS
            return list(ALL_CHATGPT_MODELS)
    except ImportError:
        pass
    return []


def _build_model_list(current_model: str) -> list:
    """Build the full model list: auto first, then ChatGPT, then role models, then all others."""
    items = []

    # First item: auto
    role_tag = "auto-route"
    items.append(("auto", role_tag, ""))

    # ChatGPT OAuth models (if authenticated)
    chatgpt_models = _fetch_chatgpt_models()
    seen = set()
    for m in chatgpt_models:
        if m not in seen:
            seen.add(m)
            items.append((m, "chatgpt", "$0"))

    # Role-mapped models (deduplicated)
    for role, model, ctx in MODEL_ROLES:
        if model not in seen:
            seen.add(model)
            items.append((model, role, ctx))

    # Cloud models from config (these don't appear in /api/tags)
    try:
        from aura.config import VERIFIED_CLOUD_MODELS
        for m in sorted(VERIFIED_CLOUD_MODELS):
            if m not in seen:
                seen.add(m)
                items.append((m, "cloud", ""))
    except ImportError:
        pass

    # All Ollama local models not already listed
    all_models = _fetch_all_models()
    for m in all_models:
        if m not in seen:
            seen.add(m)
            if m.startswith("chatgpt:"):
                tag, ctx = "chatgpt", "$0"
            elif ":cloud" in m or "-cloud" in m:
                tag, ctx = "cloud", ""
            else:
                tag, ctx = "local", ""
            items.append((m, tag, ctx))

    return items


def pick_model(console, current_model: str = "auto") -> "str | None":
    """Show interactive model picker with arrow-key navigation.

    Returns model name, 'auto', or None (cancelled).
    """
    try:
        return _pick_model_interactive(current_model)
    except Exception:
        # Fallback to simple input if prompt_toolkit fails
        return _pick_model_fallback(console, current_model)


def _pick_model_interactive(current_model: str) -> "str | None":
    """Full interactive picker using prompt_toolkit Application."""
    from prompt_toolkit import Application
    from prompt_toolkit.layout import Layout, HSplit, Window, FormattedTextControl
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.dimension import Dimension

    items = _build_model_list(current_model)
    if not items:
        return None

    # State
    selected_idx = 0
    filter_text = [""]  # mutable container for closure
    result = [None]  # mutable container for closure

    # Find current model index
    for i, (model, _, _) in enumerate(items):
        if model == current_model:
            selected_idx = i
            break

    state = {"idx": selected_idx, "scroll_offset": 0}

    def _get_filtered_items():
        ft = filter_text[0].lower()
        if not ft:
            return list(enumerate(items))
        return [(i, item) for i, item in enumerate(items)
                if ft in item[0].lower() or ft in item[1].lower()]

    def _get_display_text():
        filtered = _get_filtered_items()
        if not filtered:
            return [("class:dim", "\n  No models match filter.\n")]

        # Terminal height budget: leave room for header/footer (~8 lines)
        max_visible = 20

        # Find where selected_idx is in filtered list
        sel_pos = 0
        for j, (orig_i, _) in enumerate(filtered):
            if orig_i == state["idx"]:
                sel_pos = j
                break

        # Adjust scroll offset
        if sel_pos < state["scroll_offset"]:
            state["scroll_offset"] = sel_pos
        elif sel_pos >= state["scroll_offset"] + max_visible:
            state["scroll_offset"] = sel_pos - max_visible + 1

        offset = state["scroll_offset"]
        visible = filtered[offset:offset + max_visible]

        fragments = []
        fragments.append(("class:title", "  Model Picker"))
        cur = current_model.replace(":cloud", "").replace(":latest", "")
        fragments.append(("class:dim", f"  (current: {cur})"))
        fragments.append(("", "\n"))

        if filter_text[0]:
            fragments.append(("class:dim", "  Filter: "))
            fragments.append(("class:filter", filter_text[0]))
            fragments.append(("", "\n"))

        fragments.append(("class:dim", "  " + "-" * 55 + "\n"))

        for j, (orig_i, (model, role, ctx)) in enumerate(visible):
            is_selected = (orig_i == state["idx"])
            is_current = (model == current_model)
            global_j = offset + j

            # Cursor
            if is_selected:
                fragments.append(("class:cursor", "  > "))
            else:
                fragments.append(("", "    "))

            # Model name
            model_display = model.replace(":cloud", "").replace(":latest", "")
            if len(model_display) > 30:
                model_display = model_display[:27] + "..."

            if is_selected and is_current:
                fragments.append(("class:selected-current", f"{model_display:<32s}"))
            elif is_selected:
                fragments.append(("class:selected", f"{model_display:<32s}"))
            elif is_current:
                fragments.append(("class:current", f"{model_display:<32s}"))
            else:
                fragments.append(("class:model", f"{model_display:<32s}"))

            # Role/tag
            if role:
                if role in ("fast", "reason", "code", "think", "vision", "longctx", "auto-route"):
                    fragments.append(("class:role", f" {role:<12s}"))
                elif role == "cloud":
                    fragments.append(("class:cloud", f" {'cloud':<12s}"))
                else:
                    fragments.append(("class:local", f" {'local':<12s}"))

            # Context size
            if ctx:
                fragments.append(("class:dim", f" {ctx}"))

            # Current marker
            if is_current:
                fragments.append(("class:current-marker", " <-"))

            fragments.append(("", "\n"))

        # Scroll indicator
        total = len(filtered)
        if total > max_visible:
            if offset > 0:
                fragments.append(("class:dim", "  ... more above\n"))
            if offset + max_visible < total:
                fragments.append(("class:dim", f"  ... {total - offset - max_visible} more below\n"))

        fragments.append(("class:dim", "  " + "-" * 55 + "\n"))
        fragments.append(("class:hint", "  Up/Down"))
        fragments.append(("class:dim", " navigate  "))
        fragments.append(("class:hint", "Enter"))
        fragments.append(("class:dim", " select  "))
        fragments.append(("class:hint", "Esc"))
        fragments.append(("class:dim", " cancel  "))
        fragments.append(("class:hint", "Type"))
        fragments.append(("class:dim", " to filter"))

        return fragments

    # Keybindings
    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        filtered = _get_filtered_items()
        if not filtered:
            return
        # Find current position in filtered list
        pos = 0
        for j, (orig_i, _) in enumerate(filtered):
            if orig_i == state["idx"]:
                pos = j
                break
        if pos > 0:
            state["idx"] = filtered[pos - 1][0]

    @kb.add("down")
    def _down(event):
        filtered = _get_filtered_items()
        if not filtered:
            return
        pos = 0
        for j, (orig_i, _) in enumerate(filtered):
            if orig_i == state["idx"]:
                pos = j
                break
        if pos < len(filtered) - 1:
            state["idx"] = filtered[pos + 1][0]

    @kb.add("enter")
    def _select(event):
        filtered = _get_filtered_items()
        for orig_i, (model, _, _) in filtered:
            if orig_i == state["idx"]:
                result[0] = model
                event.app.exit()
                return
        event.app.exit()

    @kb.add("escape")
    def _cancel(event):
        result[0] = None
        event.app.exit()

    @kb.add("c-c")
    def _ctrl_c(event):
        result[0] = None
        event.app.exit()

    @kb.add("backspace")
    def _backspace(event):
        if filter_text[0]:
            filter_text[0] = filter_text[0][:-1]
            # Reset selection to first filtered item
            filtered = _get_filtered_items()
            if filtered:
                state["idx"] = filtered[0][0]
                state["scroll_offset"] = 0

    # Type to filter — catch printable characters
    @kb.add("<any>")
    def _type_char(event):
        data = event.data
        if data and len(data) == 1 and data.isprintable():
            filter_text[0] += data
            # Reset selection to first filtered item
            filtered = _get_filtered_items()
            if filtered:
                state["idx"] = filtered[0][0]
                state["scroll_offset"] = 0

    from prompt_toolkit.styles import Style
    style = Style.from_dict({
        "title": "bold cyan",
        "dim": "#666666",
        "filter": "bold yellow",
        "cursor": "bold cyan",
        "selected": "bold white",
        "selected-current": "bold green",
        "current": "green",
        "current-marker": "bold green",
        "model": "#cccccc",
        "role": "bold yellow",
        "cloud": "bold cyan",
        "local": "#888888",
        "hint": "bold cyan",
    })

    control = FormattedTextControl(_get_display_text)
    window = Window(content=control, wrap_lines=False)

    layout = Layout(HSplit([window]))

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )

    app.run()
    return result[0]


def _pick_model_fallback(console, current_model: str) -> "str | None":
    """Simple fallback picker when prompt_toolkit can't create an Application."""
    items = _build_model_list(current_model)
    console.print("\n[bold cyan]  Model Picker[/bold cyan]")
    for i, (model, role, ctx) in enumerate(items):
        marker = " [green]<-[/green]" if model == current_model else ""
        model_short = model.replace(":cloud", "").replace(":latest", "")
        console.print(f"  [bold cyan]{i + 1:>2}[/bold cyan]. {model_short:<32s} [dim yellow]{role:<10s}[/dim yellow] [dim]{ctx}[/dim]{marker}")

    console.print()
    try:
        pick = input("  Pick # or name (q to cancel) > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not pick or pick.lower() in ("q", "esc"):
        return None

    try:
        idx = int(pick) - 1
        if 0 <= idx < len(items):
            return items[idx][0]
    except ValueError:
        pass

    for model, _, _ in items:
        if pick.lower() in model.lower():
            return model

    return pick


def update_model_roles_from_config():
    """Refresh MODEL_ROLES from Config at runtime."""
    global MODEL_ROLES
    try:
        from aura.config import Config
        MODEL_ROLES = [
            ("fast", Config.MODEL_FAST, "1M ctx"),
            ("reason", Config.MODEL_REASON, "256K ctx"),
            ("code", Config.MODEL_CODE, "196K ctx"),
            ("think", Config.MODEL_THINK, "256K ctx"),
            ("vision", Config.MODEL_VISION, "256K ctx"),
            ("longctx", Config.MODEL_LONGCTX, "1M ctx"),
        ]
    except Exception:
        pass
