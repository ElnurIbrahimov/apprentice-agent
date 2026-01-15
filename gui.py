"""
Gradio GUI for Apprentice Agent
A modern chat interface with real-time thinking process visualization.
"""

import gradio as gr
import threading
import queue
import time
from datetime import datetime
from typing import Generator
from apprentice_agent import ApprenticeAgent


class AgentGUI:
    """GUI wrapper for ApprenticeAgent with real-time updates."""

    def __init__(self):
        self.agent = None
        self.update_queue = queue.Queue()
        self.is_running = False
        self.max_iterations = 10

    def _create_agent_with_hooks(self):
        """Create an agent with hooked methods for real-time updates."""
        agent = ApprenticeAgent()
        agent.max_iterations = self.max_iterations

        # Store original methods
        original_observe = agent._observe
        original_plan = agent._plan
        original_act = agent._act
        original_evaluate = agent._evaluate
        original_remember = agent._remember

        def hooked_observe(context):
            self.update_queue.put(("phase", "observe", "Analyzing current context..."))
            result = original_observe(context)
            self.update_queue.put(("observe", agent.state.observations, None))
            return result

        def hooked_plan():
            self.update_queue.put(("phase", "plan", "Creating action plan..."))
            result = original_plan()
            self.update_queue.put(("plan", agent.state.current_plan, None))
            return result

        def hooked_act():
            self.update_queue.put(("phase", "act", "Executing action..."))
            result = original_act()
            action = agent.state.last_action or {}
            tool = action.get("tool", "unknown")
            action_detail = action.get("action", "")
            self.update_queue.put(("tool", tool, action_detail))
            self.update_queue.put(("act", str(agent.state.last_result), None))
            return result

        def hooked_evaluate():
            self.update_queue.put(("phase", "evaluate", "Assessing results..."))
            result = original_evaluate()
            eval_data = agent.state.evaluation or {}
            self.update_queue.put(("evaluate", eval_data, None))
            return result

        def hooked_remember():
            self.update_queue.put(("phase", "remember", "Storing in memory..."))
            result = original_remember()
            self.update_queue.put(("remember", "Experience stored", None))
            return result

        # Apply hooks
        agent._observe = hooked_observe
        agent._plan = hooked_plan
        agent._act = hooked_act
        agent._evaluate = hooked_evaluate
        agent._remember = hooked_remember

        return agent

    def run_agent(self, goal: str, history: list) -> Generator:
        """Run the agent and yield updates for the UI."""
        if not goal.strip():
            yield history, "", "", "", "", ""
            return

        self.is_running = True
        self.agent = self._create_agent_with_hooks()

        # Add user message to history (Gradio 6.0+ format)
        history = history + [{"role": "user", "content": goal}]
        yield history, "", "", "", "", "Starting agent..."

        # Run agent in background thread
        result_container = {"result": None, "error": None}

        def run_thread():
            try:
                result_container["result"] = self.agent.run(goal)
            except Exception as e:
                result_container["error"] = str(e)
            finally:
                self.is_running = False
                self.update_queue.put(("done", None, None))

        thread = threading.Thread(target=run_thread)
        thread.start()

        # Process updates
        observe_text = ""
        plan_text = ""
        act_text = ""
        evaluate_text = ""
        tool_log = ""
        iteration = 0

        while True:
            try:
                update = self.update_queue.get(timeout=0.1)
                update_type, data, extra = update

                if update_type == "done":
                    break
                elif update_type == "phase":
                    phase_name = data
                    phase_msg = extra
                    if phase_name == "observe":
                        iteration += 1
                        observe_text = f"**Iteration {iteration}**\n{phase_msg}"
                        plan_text = ""
                        act_text = ""
                        evaluate_text = ""
                elif update_type == "observe":
                    observe_text = f"**Iteration {iteration}**\n{data}"
                elif update_type == "plan":
                    plan_text = data
                elif update_type == "tool":
                    tool_name = data
                    tool_action = extra
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    tool_log += f"[{timestamp}] **{tool_name}**\n{tool_action}\n\n"
                elif update_type == "act":
                    act_text = data[:500] + "..." if len(data) > 500 else data
                elif update_type == "evaluate":
                    if isinstance(data, dict):
                        success = "Yes" if data.get("success") else "No"
                        progress = data.get("progress", "")
                        next_step = data.get("next", "")
                        evaluate_text = f"**Success:** {success}\n**Progress:** {progress}\n**Next:** {next_step}"
                    else:
                        evaluate_text = str(data)
                elif update_type == "remember":
                    pass  # Memory stored silently

                yield history, observe_text, plan_text, act_text, evaluate_text, tool_log

            except queue.Empty:
                continue

        # Finalize response
        thread.join()

        if result_container["error"]:
            response = f"Error: {result_container['error']}"
        elif result_container["result"]:
            result = result_container["result"]

            # Extract actual outputs from history
            actual_outputs = []
            history_items = result.get("history", [])
            for item in history_items:
                item_result = item.get("result", {})
                if item_result and item_result.get("success"):
                    tool = item.get("action", {}).get("tool", "")
                    # Get output based on tool type
                    if tool == "code_executor":
                        output = item_result.get("output", "")
                        if output:
                            actual_outputs.append(f"Result: {output}")
                    elif tool == "web_search":
                        results = item_result.get("results", [])
                        if results:
                            search_output = "**Search Results:**\n"
                            for r in results[:3]:
                                title = r.get("title", "")
                                snippet = r.get("snippet", "")
                                if snippet:
                                    search_output += f"- **{title}**: {snippet}\n"
                                elif title:
                                    search_output += f"- {title}\n"
                            actual_outputs.append(search_output.strip())
                    elif tool == "filesystem":
                        if "content" in item_result:
                            actual_outputs.append(f"File content retrieved")
                        elif "entries" in item_result:
                            entries = item_result.get("entries", [])
                            actual_outputs.append(f"Listed {len(entries)} items")
                    elif tool == "summarize":
                        summary = item_result.get("summary", "")
                        if summary:
                            actual_outputs.append(summary[:200])

            # Build response with actual results
            if result.get("completed"):
                final_eval = result.get("final_evaluation", {})
                progress = final_eval.get("progress", "") if final_eval else ""

                if actual_outputs:
                    # Show actual output prominently
                    response = "\n".join(actual_outputs)
                    if progress:
                        response += f"\n\n{progress}"
                elif progress:
                    response = progress
                else:
                    response = "Task completed successfully."
            else:
                # Task didn't complete - show what happened
                iterations = result.get("iterations", 0)
                final_eval = result.get("final_evaluation", {})
                progress = final_eval.get("progress", "") if final_eval else ""
                response = f"Task incomplete after {iterations} iterations."
                if actual_outputs:
                    response += "\n" + "\n".join(actual_outputs)
                if progress:
                    response += f"\n{progress}"
        else:
            response = "Agent finished without a result."

        # Add assistant response (Gradio 6.0+ format)
        history = history + [{"role": "assistant", "content": response}]
        yield history, observe_text, plan_text, act_text, evaluate_text, tool_log

    def search_memory(self, query: str) -> str:
        """Search agent memory for relevant experiences."""
        if not query.strip():
            return "Enter a search query to find relevant memories."

        try:
            from apprentice_agent.memory import MemorySystem
            memory = MemorySystem()
            results = memory.recall(query, n_results=5)

            if not results:
                return "No memories found matching your query."

            output = ""
            for i, mem in enumerate(results, 1):
                content = mem.get("content", "")[:200]
                metadata = mem.get("metadata", {})
                score = mem.get("distance", 0)
                output += f"### Memory {i} (relevance: {score:.2f})\n"
                output += f"{content}...\n"
                if metadata:
                    output += f"*Goal: {metadata.get('goal', 'N/A')}*\n"
                output += "\n---\n\n"

            return output
        except Exception as e:
            return f"Error searching memory: {str(e)}"

    def get_memory_stats(self) -> str:
        """Get memory system statistics."""
        try:
            from apprentice_agent.memory import MemorySystem
            memory = MemorySystem()
            count = memory.count()
            recent = memory.get_recent(3)

            stats = f"**Total Memories:** {count}\n\n"
            if recent:
                stats += "**Recent Memories:**\n"
                for mem in recent:
                    content = mem.get("content", "")[:100]
                    stats += f"- {content}...\n"
            return stats
        except Exception as e:
            return f"Error: {str(e)}"

    def update_settings(self, max_iter: int) -> str:
        """Update agent settings."""
        self.max_iterations = int(max_iter)
        return f"Settings updated. Max iterations: {self.max_iterations}"

    def clear_thinking(self):
        """Clear all thinking panels."""
        return "", "", "", "", ""


def create_gui():
    """Create and return the Gradio interface."""
    gui = AgentGUI()

    custom_css = """
    .thinking-panel {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 15px;
        min-height: 150px;
    }
    .phase-header {
        color: #00d4ff;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .tool-log {
        font-family: 'Consolas', monospace;
        font-size: 12px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .gradio-container {
        max-width: 1400px !important;
    }
    """

    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(title="Apprentice Agent") as app:
        gr.Markdown(
            """
            # Apprentice Agent
            ### An intelligent AI assistant with memory and reasoning capabilities
            """
        )

        with gr.Row():
            # Left column - Chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Ask the agent to do something...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", size="sm")
                    stop_btn = gr.Button("Stop", variant="stop", size="sm")

            # Right column - Thinking Process & Settings
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Thinking Process"):
                        with gr.Row():
                            with gr.Column():
                                observe_box = gr.Markdown(
                                    label="Observe",
                                    value="*Waiting for input...*",
                                    elem_classes=["thinking-panel"]
                                )
                                gr.Markdown("**OBSERVE** - Analyzing context", elem_classes=["phase-header"])

                            with gr.Column():
                                plan_box = gr.Markdown(
                                    label="Plan",
                                    value="*Waiting...*",
                                    elem_classes=["thinking-panel"]
                                )
                                gr.Markdown("**PLAN** - Creating strategy", elem_classes=["phase-header"])

                        with gr.Row():
                            with gr.Column():
                                act_box = gr.Markdown(
                                    label="Act",
                                    value="*Waiting...*",
                                    elem_classes=["thinking-panel"]
                                )
                                gr.Markdown("**ACT** - Executing action", elem_classes=["phase-header"])

                            with gr.Column():
                                evaluate_box = gr.Markdown(
                                    label="Evaluate",
                                    value="*Waiting...*",
                                    elem_classes=["thinking-panel"]
                                )
                                gr.Markdown("**EVALUATE** - Assessing results", elem_classes=["phase-header"])

                    with gr.TabItem("Tool Usage"):
                        tool_log = gr.Markdown(
                            value="*No tools used yet...*",
                            elem_classes=["tool-log"],
                            label="Tool Log"
                        )
                        clear_tools_btn = gr.Button("Clear Log", size="sm")

                    with gr.TabItem("Memory"):
                        memory_stats = gr.Markdown(value="Click 'Refresh' to load memory stats")
                        refresh_memory_btn = gr.Button("Refresh Stats", size="sm")

                        gr.Markdown("---")
                        gr.Markdown("### Search Memory")

                        with gr.Row():
                            memory_query = gr.Textbox(
                                placeholder="Search past experiences...",
                                show_label=False,
                                scale=3
                            )
                            search_btn = gr.Button("Search", scale=1)

                        memory_results = gr.Markdown(value="*Enter a query to search memories*")

                    with gr.TabItem("Settings"):
                        gr.Markdown("### Agent Configuration")

                        max_iter_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Max Iterations",
                            info="Maximum thinking cycles before stopping"
                        )

                        settings_status = gr.Markdown("*Current: 10 iterations*")

                        apply_settings_btn = gr.Button("Apply Settings", variant="primary")

                        gr.Markdown("---")
                        gr.Markdown(
                            """
                            ### About

                            **Apprentice Agent** uses a 5-phase thinking process:

                            1. **Observe** - Gather context and recall relevant memories
                            2. **Plan** - Create a strategy to achieve the goal
                            3. **Act** - Execute one action from the plan
                            4. **Evaluate** - Assess if the action succeeded
                            5. **Remember** - Store the experience for future learning

                            The agent has access to:
                            - File system operations
                            - Web search
                            - Python code execution
                            """
                        )

        # Event handlers
        def submit_message(message, history):
            """Consume generator and return final values."""
            if not message.strip():
                return history, "", "", "", "", ""

            final_result = None
            for result in gui.run_agent(message, history):
                final_result = result

            if final_result:
                return final_result
            return history, "", "", "", "", ""

        def clear_chat():
            return [], "", "", "", "", ""

        # Wire up events
        send_btn.click(
            fn=submit_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, observe_box, plan_box, act_box, evaluate_box, tool_log],
        ).then(
            fn=lambda: "",
            outputs=msg_input
        )

        msg_input.submit(
            fn=submit_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, observe_box, plan_box, act_box, evaluate_box, tool_log],
        ).then(
            fn=lambda: "",
            outputs=msg_input
        )

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, observe_box, plan_box, act_box, evaluate_box, tool_log]
        )

        clear_tools_btn.click(
            fn=lambda: "*Log cleared*",
            outputs=tool_log
        )

        refresh_memory_btn.click(
            fn=gui.get_memory_stats,
            outputs=memory_stats
        )

        search_btn.click(
            fn=gui.search_memory,
            inputs=memory_query,
            outputs=memory_results
        )

        memory_query.submit(
            fn=gui.search_memory,
            inputs=memory_query,
            outputs=memory_results
        )

        apply_settings_btn.click(
            fn=gui.update_settings,
            inputs=max_iter_slider,
            outputs=settings_status
        )

    return app, theme, custom_css


if __name__ == "__main__":
    app, theme, css = create_gui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=css
    )
