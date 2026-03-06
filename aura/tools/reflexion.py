"""
Reflexion Engine for AURA - Learn From Mistakes

Implements a reflection loop that:
1. Checks past lessons before attempting a task
2. Attempts the task
3. If failed, reflects on why and stores the lesson
4. Retries with accumulated lessons

SECURITY: Thread-safe with file locking to prevent corruption.
"""

import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import requests

from ..config import Config

logger = logging.getLogger(__name__)

# Thread lock for in-process synchronization
_reflexion_lock = threading.RLock()


# ============================================================================
#                    CROSS-PLATFORM FILE LOCKING
# ============================================================================

@contextmanager
def file_lock(filepath, mode='r', encoding='utf-8', exclusive: bool = True):
    f = None
    try:
        f = open(filepath, mode, encoding=encoding)
        # Try to acquire OS-level lock
        try:
            if sys.platform == 'win32':
                import msvcrt
                file_size = max(1, os.path.getsize(filepath))
                lock_mode = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
                msvcrt.locking(f.fileno(), lock_mode, file_size)
            else:
                import fcntl
                lock_mode = fcntl.LOCK_EX | fcntl.LOCK_NB if exclusive else fcntl.LOCK_SH | fcntl.LOCK_NB
                fcntl.flock(f.fileno(), lock_mode)
        except (ImportError, OSError) as e:
            logger.warning(f"File locking unavailable: {e}")
            # f is already open, proceed without lock
        yield f
    finally:
        if f is not None:
            try:
                if sys.platform == 'win32':
                    import msvcrt
                    try:
                        file_size = max(1, os.path.getsize(filepath))
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, file_size)
                    except Exception:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
            finally:
                f.close()


@dataclass
class Reflection:
    """A single reflection/lesson from a past attempt."""
    task: str           # What was attempted
    attempt: str        # What AURA tried
    outcome: str        # "success" or "failure"
    feedback: str       # Why it failed
    reflection: str     # The lesson learned
    timestamp: str      # When it happened


@dataclass
class ReflexionResult:
    """Result of executing a task with reflexion."""
    task: str                           # Original task
    final_output: str                   # Final result
    success: bool                       # Did it work?
    attempts: int                       # How many tries
    reflections_used: List[str]         # Past lessons that helped
    new_reflection: Optional[str]       # New lesson if failed


class ReflexionEngine:
    """
    Engine that learns from mistakes using reflection.

    Flow:
        Task -> Check Past Lessons -> Attempt -> Failed? -> Reflect -> Store -> Retry
                                                    |
                                              Success? -> Done
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = None,
        memory_path: Optional[str] = None,
        max_attempts: int = 3
    ):
        """
        Initialize the reflexion engine.

        Args:
            ollama_url: URL for Ollama API
            model: Model to use for generation (default: Config.MODEL_CODE)
            memory_path: Path to memories.jsonl file
            max_attempts: Maximum retry attempts
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model or Config.MODEL_CODE
        self.max_attempts = max_attempts

        # Set up memory path
        if memory_path is None:
            # Go up from tools/ -> aura/ -> project root -> data/reflexion
            base_dir = Path(__file__).parent.parent.parent / "data" / "reflexion"
            base_dir.mkdir(parents=True, exist_ok=True)
            self.memory_path = base_dir / "memories.jsonl"
        else:
            self.memory_path = Path(memory_path)
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing reflections
        self.reflections: List[Reflection] = self._load_reflections()

        logger.info(f"ReflexionEngine initialized with {len(self.reflections)} stored lessons")

    def _load_reflections(self) -> List[Reflection]:
        """Load reflections from JSONL file with file locking."""
        reflections = []

        if not self.memory_path.exists():
            # Create empty file
            self.memory_path.touch()
            return reflections

        try:
            # Use shared lock for reading
            with file_lock(self.memory_path, 'r', exclusive=False) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        reflections.append(Reflection(**data))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"Skipping invalid line {line_num} in memories: {e}")
        except IOError as e:
            logger.error(f"Error reading memories file: {e}")

        return reflections

    def _save_reflection(self, reflection: Reflection) -> None:
        """Append a reflection to the JSONL file with file locking.

        SECURITY: Uses exclusive lock to prevent corruption from concurrent writes.
        """
        try:
            # Use exclusive lock for writing
            with file_lock(self.memory_path, 'a', exclusive=True) as f:
                f.write(json.dumps(asdict(reflection)) + "\n")
                f.flush()  # Ensure data is written before releasing lock
            self.reflections.append(reflection)
            logger.info(f"Stored new reflection: {reflection.reflection[:50]}...")
        except IOError as e:
            logger.error(f"Error saving reflection: {e}")

    def _find_relevant(self, task: str, k: int = 3) -> List[Reflection]:
        """
        Find past lessons relevant to the current task.

        Uses simple keyword overlap for matching.

        Args:
            task: The current task description
            k: Maximum number of lessons to return

        Returns:
            List of relevant Reflection objects, sorted by relevance
        """
        task_words = set(task.lower().split())

        # Remove common words that don't add meaning
        stop_words = {"a", "an", "the", "is", "are", "to", "for", "of", "in", "on", "with", "that", "this"}
        task_words -= stop_words

        scored = []
        for ref in self.reflections:
            ref_words = set(ref.task.lower().split()) - stop_words
            overlap = len(task_words & ref_words)
            if overlap > 0:
                scored.append((overlap, ref))

        # Sort by overlap score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        return [ref for _, ref in scored[:k]]

    def _call_llm(self, prompt: str, timeout: int = 60) -> str:
        """Call Ollama API to generate text."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024
                    }
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            logger.warning("LLM request timed out")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            return ""

    def _generate_attempt(self, task: str, past_reflections: List[Reflection]) -> str:
        """
        Generate an attempt at the task, using past lessons.

        Args:
            task: The task to complete
            past_reflections: Relevant past lessons to apply

        Returns:
            The generated attempt/response
        """
        # Build lessons section
        lessons_text = ""
        if past_reflections:
            lessons_text = "LESSONS FROM PAST ATTEMPTS:\n"
            for ref in past_reflections:
                lessons_text += f'- Task: "{ref.task}"\n'
                lessons_text += f'  Learning: "{ref.reflection}"\n\n'
            lessons_text += "Apply these lessons to avoid repeating mistakes.\n\n"

        prompt = f"""Complete this task:

TASK: {task}

{lessons_text}Provide a complete, correct response:"""

        return self._call_llm(prompt)

    def _generate_reflection(self, task: str, attempt: str, feedback: str) -> str:
        """
        Generate a reflection on why the attempt failed.

        Args:
            task: The original task
            attempt: What was tried
            feedback: Why it failed

        Returns:
            A brief actionable lesson
        """
        prompt = f"""You attempted a task and it failed. Reflect on what went wrong.

TASK: {task}

YOUR ATTEMPT:
{attempt[:1000]}

WHY IT FAILED:
{feedback}

Write a brief, actionable lesson (1-2 sentences):
1. What specifically went wrong
2. What to do differently next time

LESSON:"""

        reflection = self._call_llm(prompt)

        # Ensure we have something useful
        if not reflection:
            reflection = f"Task failed due to: {feedback}. Need to address this issue."

        return reflection

    def execute(
        self,
        task: str,
        evaluator: Callable[[str, str], Tuple[bool, str]]
    ) -> ReflexionResult:
        """
        Execute a task with learning from mistakes.

        Args:
            task: What to do
            evaluator: Function(task, output) -> (success: bool, feedback: str)

        Returns:
            ReflexionResult with final output and metadata
        """
        # 1. Find relevant past lessons
        past_lessons = self._find_relevant(task, k=3)
        reflections_used = [r.reflection for r in past_lessons]

        if past_lessons:
            logger.info(f"Found {len(past_lessons)} relevant past lessons")

        output = ""
        last_feedback = ""

        for attempt_num in range(self.max_attempts):
            logger.info(f"Attempt {attempt_num + 1}/{self.max_attempts} for task: {task[:50]}...")

            # 2. Generate attempt using past lessons
            output = self._generate_attempt(task, past_lessons)

            if not output:
                last_feedback = "LLM failed to generate a response"
                continue

            # 3. Evaluate result
            success, feedback = evaluator(task, output)
            last_feedback = feedback

            if success:
                logger.info(f"Task succeeded on attempt {attempt_num + 1}")

                # Optionally store successful patterns too
                if attempt_num > 0:
                    # We learned something if it took multiple tries
                    self._save_reflection(Reflection(
                        task=task,
                        attempt=output[:500],
                        outcome="success",
                        feedback=feedback,
                        reflection=f"After {attempt_num + 1} attempts, succeeded by applying lessons learned.",
                        timestamp=datetime.now().isoformat()
                    ))

                return ReflexionResult(
                    task=task,
                    final_output=output,
                    success=True,
                    attempts=attempt_num + 1,
                    reflections_used=reflections_used,
                    new_reflection=None
                )

            # 4. Failed - reflect on why
            logger.info(f"Attempt {attempt_num + 1} failed: {feedback}")
            reflection = self._generate_reflection(task, output, feedback)

            # 5. Store lesson for future
            new_ref = Reflection(
                task=task,
                attempt=output[:500],
                outcome="failure",
                feedback=feedback,
                reflection=reflection,
                timestamp=datetime.now().isoformat()
            )
            self._save_reflection(new_ref)

            # 6. Add to context for next attempt
            past_lessons.append(new_ref)

        # All attempts failed
        logger.warning(f"All {self.max_attempts} attempts failed for task: {task[:50]}...")

        return ReflexionResult(
            task=task,
            final_output=output,
            success=False,
            attempts=self.max_attempts,
            reflections_used=reflections_used,
            new_reflection=past_lessons[-1].reflection if past_lessons else None
        )

    def get_lessons_summary(self) -> str:
        """Get a summary of all stored lessons."""
        if not self.reflections:
            return "No lessons stored yet."

        summary = f"Total lessons: {len(self.reflections)}\n\n"

        # Group by outcome
        failures = [r for r in self.reflections if r.outcome == "failure"]
        successes = [r for r in self.reflections if r.outcome == "success"]

        summary += f"Failures learned from: {len(failures)}\n"
        summary += f"Successes recorded: {len(successes)}\n\n"

        # Show recent lessons
        summary += "Recent lessons:\n"
        for ref in self.reflections[-5:]:
            summary += f"- [{ref.outcome}] {ref.reflection[:100]}...\n"

        return summary

    def clear_memories(self) -> None:
        """Clear all stored memories (use with caution)."""
        self.reflections = []
        with open(self.memory_path, "w", encoding="utf-8") as f:
            f.write("")
        logger.info("All reflexion memories cleared")


# Example evaluators for common use cases

def code_syntax_evaluator(task: str, output: str) -> Tuple[bool, str]:
    """Check if output contains valid Python syntax."""
    # Extract code if wrapped in markdown
    code = output
    if "```python" in output:
        start = output.find("```python") + 9
        end = output.find("```", start)
        if end > start:
            code = output[start:end].strip()
    elif "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        if end > start:
            code = output[start:end].strip()

    try:
        compile(code, '<string>', 'exec')
        return True, "Valid Python syntax"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def function_evaluator(task: str, output: str) -> Tuple[bool, str]:
    """Check if output contains a function definition with return."""
    if "def " not in output:
        return False, "Missing function definition (no 'def' keyword found)"
    if "return" not in output:
        return False, "Function has no return statement"

    # Also check syntax
    return code_syntax_evaluator(task, output)


def answer_completeness_evaluator(task: str, output: str) -> Tuple[bool, str]:
    """Check if response is a complete answer."""
    if len(output) < 20:
        return False, "Response too short (less than 20 characters)"
    if output.endswith("?"):
        return False, "Response ends with a question instead of an answer"
    if output.lower().startswith("i don't know") or output.lower().startswith("i cannot"):
        return False, "Response indicates inability to answer"
    return True, "Response appears complete"


def json_evaluator(task: str, output: str) -> Tuple[bool, str]:
    """Check if output is valid JSON."""
    # Try to extract JSON if wrapped
    text = output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else ""

    try:
        json.loads(text)
        return True, "Valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e.msg} at position {e.pos}"


if __name__ == "__main__":
    # Test the reflexion engine
    print("=" * 60)
    print("Reflexion Engine Test")
    print("=" * 60)

    engine = ReflexionEngine()

    print(f"\nLoaded {len(engine.reflections)} existing lessons")

    # Test 1: Function writing task
    print("\n--- Test 1: Write a function ---")

    task = "Write a Python function to reverse a string"
    result = engine.execute(task, function_evaluator)

    print(f"Success: {result.success}")
    print(f"Attempts: {result.attempts}")
    print(f"Lessons used: {result.reflections_used}")
    if result.new_reflection:
        print(f"New lesson: {result.new_reflection}")
    print(f"\nOutput:\n{result.final_output[:500]}")

    # Test 2: JSON task
    print("\n--- Test 2: Generate JSON ---")

    task = "Generate a JSON object with name, age, and city fields"
    result = engine.execute(task, json_evaluator)

    print(f"Success: {result.success}")
    print(f"Attempts: {result.attempts}")
    print(f"\nOutput:\n{result.final_output[:300]}")

    # Show summary
    print("\n--- Lessons Summary ---")
    print(engine.get_lessons_summary())

    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Total stored lessons: {len(engine.reflections)}")
