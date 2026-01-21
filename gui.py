"""
Aura - Apprentice Agent GUI
Simple, readable chat interface.
"""

import gradio as gr
import time
from typing import Generator
from pathlib import Path

from apprentice_agent import ApprenticeAgent
from apprentice_agent.metacognition import MetacognitionLogger


# ============================================================================
# CSS WITH EXACT COLORS
# ============================================================================

CUSTOM_CSS = """
/* Page background */
.gradio-container {
    background: #0f172a !important;
}

/* Sidebar and chat area */
.block {
    background: #1e293b !important;
    border: none !important;
}

/* All text - almost white */
body, .gradio-container, .gradio-container *, p, span, div, label {
    color: #f1f5f9 !important;
}

/* Headers - pure white */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Labels */
label span {
    color: #94a3b8 !important;
}

/* Input box */
textarea, input[type="text"], input[type="number"] {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
}

textarea::placeholder, input::placeholder {
    color: #64748b !important;
}

/* Send button - blue */
button.primary, button[variant="primary"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
}

/* Other buttons */
button.secondary, button:not(.primary) {
    background: #334155 !important;
    color: #f1f5f9 !important;
    border: none !important;
}

/* Chatbot container */
.chatbot, .chatbot-container {
    background: #1e293b !important;
}

/* User message bubble - blue */
.message.user, .user-message, [data-testid="user"] {
    background: #3b82f6 !important;
    color: #ffffff !important;
}

/* Aura message bubble - dark gray */
.message.bot, .bot-message, [data-testid="bot"] {
    background: #334155 !important;
    color: #f1f5f9 !important;
}

/* Message text must be visible */
.message p, .message span, .message div {
    color: inherit !important;
}

/* Markdown inside messages */
.message .prose, .message .markdown {
    color: inherit !important;
}

.message .prose *, .message .markdown * {
    color: inherit !important;
}

/* Slider */
input[type="range"] {
    accent-color: #3b82f6 !important;
}

/* Accordion */
.accordion {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
}

/* Status dot */
.status-online {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 8px;
}

.status-offline {
    display: inline-block;
    width: 10px;
    height: 10px;
    background: #ef4444;
    border-radius: 50%;
    margin-right: 8px;
}
"""


# ============================================================================
# GUI CLASS
# ============================================================================

class AuraGUI:
    def __init__(self):
        self.max_iterations = 10

    def _check_fluxmind(self) -> dict:
        try:
            from apprentice_agent.tools import FluxMindTool, FLUXMIND_AVAILABLE
            if FLUXMIND_AVAILABLE:
                models_path = Path(__file__).parent / "models" / "fluxmind_v0751.pt"
                tool = FluxMindTool(str(models_path))
                if tool.is_available():
                    return {"available": True, "version": "0.75.1"}
            return {"available": False}
        except:
            return {"available": False}

    def get_status_html(self) -> str:
        status = self._check_fluxmind()
        if status["available"]:
            return f'''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-online"></span>
<strong style="color: #22c55e;">FluxMind Online</strong>
<div style="color: #94a3b8; font-size: 13px; margin-top: 4px;">v{status["version"]}</div>
</div>'''
        else:
            return '''<div style="background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0;">
<span class="status-offline"></span>
<strong style="color: #ef4444;">FluxMind Offline</strong>
</div>'''

    def run_agent(self, message: str, history: list) -> Generator:
        if not message.strip():
            yield history
            return

        # Add user message
        history = history + [{"role": "user", "content": message}]
        yield history

        # Run agent
        try:
            agent = ApprenticeAgent()
            agent.max_iterations = self.max_iterations
            result = agent.run(message)

            # Get response
            if result.get("fast_path"):
                response = result.get("response", "Done.")
            else:
                response = self._build_response(result)

            history = history + [{"role": "assistant", "content": response}]
            yield history

        except Exception as e:
            history = history + [{"role": "assistant", "content": f"Error: {e}"}]
            yield history

    def _build_response(self, result: dict) -> str:
        outputs = []

        for item in result.get("history", []):
            item_result = item.get("result", {})
            if not item_result:
                continue

            tool = item.get("action", {}).get("tool", "")

            if tool == "fluxmind":
                conf = item_result.get("confidence", 0)
                if isinstance(conf, (int, float)):
                    pct = conf * 100 if conf <= 1 else conf
                    if pct >= 80:
                        outputs.append(f"FluxMind: {pct:.1f}% confidence (high)")
                    else:
                        outputs.append(f"FluxMind: {pct:.1f}% confidence (uncertain)")

            elif tool == "web_search" and item_result.get("success"):
                for r in item_result.get("results", [])[:3]:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    if snippet:
                        outputs.append(f"{title}: {snippet}")

            elif tool == "code_executor" and item_result.get("success"):
                output = item_result.get("output", "")
                if output:
                    outputs.append(output[:500])

        if result.get("completed"):
            final_eval = result.get("final_evaluation", {})
            progress = final_eval.get("progress", "") if final_eval else ""
            if outputs:
                return "\n\n".join(outputs) + (f"\n\n{progress}" if progress else "")
            return progress or "Done."
        else:
            return f"Incomplete after {result.get('iterations', 0)} iterations."

    def get_stats(self) -> str:
        try:
            logger = MetacognitionLogger()
            stats = logger.get_stats()
            if "error" in stats:
                return stats['error']
            return f"Actions: {stats['total_actions']}, Success: {stats['success_rate']}%"
        except Exception as e:
            return str(e)


# ============================================================================
# CREATE APP
# ============================================================================

def create_app():
    gui = AuraGUI()

    with gr.Blocks(title="Aura", css=CUSTOM_CSS) as app:

        gr.HTML("""<div style="text-align: center; padding: 16px; border-bottom: 1px solid #334155;">
            <h1 style="color: #ffffff; margin: 0;">Aura</h1>
            <p style="color: #94a3b8; margin: 4px 0 0 0;">AI Assistant</p>
        </div>""")

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gr.HTML("""<div style="text-align: center; padding: 16px;">
                    <div style="font-size: 40px;">🤖</div>
                    <div style="color: #ffffff; font-weight: bold; font-size: 18px;">Aura</div>
                    <div style="color: #94a3b8; font-size: 13px;">Apprentice Agent</div>
                </div>""")

                fluxmind_html = gr.HTML(value=gui.get_status_html())
                refresh_btn = gr.Button("Refresh", size="sm")

                max_iter = gr.Slider(1, 20, value=10, step=1, label="Max Iterations")

                with gr.Accordion("Stats", open=False):
                    stats_md = gr.Markdown("Click Load")
                    stats_btn = gr.Button("Load", size="sm")

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, show_label=False)

                with gr.Row():
                    msg = gr.Textbox(placeholder="Type message...", show_label=False, scale=5)
                    send = gr.Button("Send", variant="primary", scale=1)

                clear = gr.Button("Clear", size="sm")

        # Events
        def on_send(message, history):
            for h in gui.run_agent(message, history):
                yield h

        send.click(on_send, [msg, chatbot], chatbot).then(lambda: "", outputs=msg)
        msg.submit(on_send, [msg, chatbot], chatbot).then(lambda: "", outputs=msg)
        clear.click(lambda: [], outputs=chatbot)
        refresh_btn.click(gui.get_status_html, outputs=fluxmind_html)
        stats_btn.click(gui.get_stats, outputs=stats_md)
        max_iter.change(lambda v: setattr(gui, 'max_iterations', int(v)), inputs=max_iter)

    return app


if __name__ == "__main__":
    print("Starting Aura GUI...")
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
