"""Spaced Repetition tool with SM-2 algorithm for flashcard-based learning."""

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class FlashCard:
    """A flashcard with SM-2 scheduling metadata."""
    id: str
    front: str
    back: str
    deck: str = "default"
    tags: List[str] = field(default_factory=list)
    ease_factor: float = 2.5
    interval: int = 1           # days
    repetitions: int = 0
    next_review: str = ""       # ISO datetime
    last_reviewed: Optional[str] = None
    review_history: List[dict] = field(default_factory=list)
    created_at: str = ""
    source: str = "manual"      # manual, auto:neurodream, auto:conversation

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.next_review:
            self.next_review = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlashCard':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SpacedRepetitionTool:
    """Flashcard-based learning with SM-2 spaced repetition algorithm."""

    name = "spaced_repetition"
    description = "Flashcard-based learning with SM-2 spaced repetition"

    CARDS_FILE = Path(__file__).parent.parent.parent / "data" / "flashcards.json"

    def __init__(self):
        self._ensure_file()

    def _ensure_file(self):
        """Ensure the cards file and directory exist."""
        self.CARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not self.CARDS_FILE.exists():
            self._save_cards([])

    def _load_cards(self) -> List[Dict[str, Any]]:
        """Load cards from JSON file."""
        try:
            with open(self.CARDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_cards(self, cards: List[Dict[str, Any]]) -> bool:
        """Save cards to JSON file."""
        try:
            with open(self.CARDS_FILE, "w", encoding="utf-8") as f:
                json.dump(cards, f, indent=4)
            return True
        except IOError:
            return False

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _update_sm2(self, card: FlashCard, quality: int) -> FlashCard:
        """SuperMemo 2 algorithm.

        quality: 0=blackout, 1=wrong, 2=hard, 3=ok, 4=good, 5=perfect
        """
        quality = max(0, min(5, quality))

        if quality < 3:
            # Failed — reset repetitions, review again tomorrow
            card.repetitions = 0
            card.interval = 1
        else:
            if card.repetitions == 0:
                card.interval = 1
            elif card.repetitions == 1:
                card.interval = 6
            else:
                card.interval = int(card.interval * card.ease_factor)
            card.repetitions += 1

        # Update ease factor (minimum 1.3)
        card.ease_factor = max(1.3,
            card.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

        card.next_review = (datetime.now() + timedelta(days=card.interval)).isoformat()
        card.last_reviewed = datetime.now().isoformat()
        card.review_history.append({
            "date": datetime.now().isoformat(),
            "quality": quality,
            "interval": card.interval
        })

        return card

    def add_card(self, front: str, back: str, tags: List[str] = None,
                 deck: str = "default", source: str = "manual") -> dict:
        """Create a new flashcard."""
        if not front or not back:
            return {"success": False, "error": "Both front and back text are required"}

        card = FlashCard(
            id=self._generate_id(),
            front=front,
            back=back,
            deck=deck,
            tags=tags or [],
            source=source,
        )

        cards = self._load_cards()
        cards.append(card.to_dict())
        self._save_cards(cards)

        return {
            "success": True,
            "card_id": card.id,
            "deck": card.deck,
            "response": f"Flashcard added to '{deck}' deck: {front[:50]}..."
        }

    def add_cards_from_text(self, text: str, tags: List[str] = None, deck: str = "default") -> dict:
        """Auto-generate flashcards from text using proposition extraction."""
        if not text:
            return {"success": False, "error": "No text provided"}

        propositions = self._extract_propositions_simple(text)

        if not propositions:
            return {"success": False, "error": "Could not extract any facts from the text"}

        cards = self._load_cards()
        created = 0

        for prop in propositions:
            # Turn proposition into Q&A
            front, back = self._proposition_to_qa(prop)
            if front and back:
                card = FlashCard(
                    id=self._generate_id(),
                    front=front,
                    back=back,
                    deck=deck,
                    tags=tags or [],
                    source="auto:text",
                )
                cards.append(card.to_dict())
                created += 1

        self._save_cards(cards)
        return {
            "success": True,
            "created": created,
            "response": f"Auto-generated {created} flashcard(s) from text"
        }

    def _extract_propositions_simple(self, text: str) -> List[str]:
        """Extract atomic propositions from text (simplified regex extraction)."""
        propositions = []
        seen = set()

        def _add(prop: str):
            prop = prop.strip().rstrip(".")
            if prop and len(prop) > 8 and prop not in seen:
                seen.add(prop)
                propositions.append(prop)

        # Definitional: "X is Y"
        for match in re.finditer(
            r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z]?[a-zA-Z]+){0,3})\s+(?:is|are|was|were|means?)\s+(.{5,80}?)(?:[.!?,;]|$)',
            text
        ):
            _add(f"{match.group(1)} is {match.group(2).strip()}")

        # Verb-object: "X does Y"
        for match in re.finditer(
            r'(\b[A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+){0,2})\s+(uses?|creates?|provides?|contains?|requires?|supports?|enables?)\s+(.{3,60}?)(?:[.!?,;]|$)',
            text, re.IGNORECASE
        ):
            _add(f"{match.group(1)} {match.group(2)} {match.group(3).strip()}")

        # Causal: "X because Y"
        for match in re.finditer(
            r'(.{10,60}?)\s+(?:because|since|due\s+to)\s+(.{5,60}?)(?:[.!?,;]|$)',
            text, re.IGNORECASE
        ):
            _add(f"{match.group(1).strip()} because {match.group(2).strip()}")

        return propositions[:15]

    def _proposition_to_qa(self, proposition: str) -> tuple:
        """Convert a proposition into a question-answer pair."""
        # "X is Y" -> Q: "What is X?" A: "Y"
        is_match = re.match(r'(.+?)\s+(?:is|are)\s+(.+)', proposition, re.IGNORECASE)
        if is_match:
            subject = is_match.group(1).strip()
            predicate = is_match.group(2).strip()
            return f"What is {subject}?", predicate

        # "X because Y" -> Q: "Why X?" A: "Because Y"
        because_match = re.match(r'(.+?)\s+because\s+(.+)', proposition, re.IGNORECASE)
        if because_match:
            effect = because_match.group(1).strip()
            cause = because_match.group(2).strip()
            return f"Why does {effect}?", f"Because {cause}"

        # "X does Y" -> Q: "What does X do?" A: "Y"
        verb_match = re.match(r'(.+?)\s+(uses?|creates?|provides?|contains?|requires?|supports?|enables?)\s+(.+)', proposition, re.IGNORECASE)
        if verb_match:
            subject = verb_match.group(1).strip()
            verb = verb_match.group(2).strip()
            obj = verb_match.group(3).strip()
            return f"What does {subject} {verb}?", obj

        # Fallback: use proposition as both Q and A
        return f"What do you know about: {proposition[:40]}?", proposition

    def review(self) -> dict:
        """Get the next due card for review."""
        cards = self._load_cards()
        now = datetime.now()

        due_cards = []
        for c in cards:
            try:
                next_review = datetime.fromisoformat(c.get("next_review", ""))
                if next_review <= now:
                    due_cards.append(c)
            except (ValueError, TypeError):
                due_cards.append(c)  # If no review date, it's due

        if not due_cards:
            return {
                "success": True,
                "due": False,
                "total_cards": len(cards),
                "response": "No cards due for review right now!"
            }

        # Sort by next_review (oldest first) and pick the first
        due_cards.sort(key=lambda c: c.get("next_review", ""))
        card = due_cards[0]

        return {
            "success": True,
            "due": True,
            "card_id": card["id"],
            "front": card["front"],
            "deck": card.get("deck", "default"),
            "due_count": len(due_cards),
            "response": f"[{card['deck']}] {card['front']}\n\n(Rate 0-5 after revealing answer)"
        }

    def answer(self, card_id: str, quality: int) -> dict:
        """Record answer quality and update scheduling."""
        if not card_id:
            return {"success": False, "error": "No card ID provided"}

        try:
            quality = int(quality)
        except (ValueError, TypeError):
            return {"success": False, "error": f"Invalid quality: {quality}. Must be 0-5"}

        if not 0 <= quality <= 5:
            return {"success": False, "error": "Quality must be 0-5"}

        cards = self._load_cards()
        card_data = None
        card_index = None

        for i, c in enumerate(cards):
            if c.get("id") == card_id:
                card_data = c
                card_index = i
                break

        if card_data is None:
            return {"success": False, "error": f"Card not found: {card_id}"}

        card = FlashCard.from_dict(card_data)
        card = self._update_sm2(card, quality)
        cards[card_index] = card.to_dict()
        self._save_cards(cards)

        quality_labels = {0: "Blackout", 1: "Wrong", 2: "Hard", 3: "OK", 4: "Good", 5: "Perfect"}
        next_dt = datetime.fromisoformat(card.next_review)

        return {
            "success": True,
            "card_id": card.id,
            "quality": quality,
            "quality_label": quality_labels.get(quality, "Unknown"),
            "new_interval": card.interval,
            "new_ease": round(card.ease_factor, 2),
            "next_review": card.next_review,
            "back": card.back,
            "response": f"Rated {quality_labels.get(quality, quality)}. "
                        f"Next review in {card.interval} day(s) ({next_dt.strftime('%Y-%m-%d')})"
        }

    def list_decks(self) -> dict:
        """List all decks with summary stats."""
        cards = self._load_cards()
        now = datetime.now()
        decks = {}

        for c in cards:
            deck = c.get("deck", "default")
            if deck not in decks:
                decks[deck] = {"total": 0, "due": 0, "new": 0, "learning": 0, "mastered": 0}
            decks[deck]["total"] += 1

            reps = c.get("repetitions", 0)
            try:
                next_review = datetime.fromisoformat(c.get("next_review", ""))
                is_due = next_review <= now
            except (ValueError, TypeError):
                is_due = True

            if reps == 0:
                decks[deck]["new"] += 1
            elif reps < 3:
                decks[deck]["learning"] += 1
            else:
                decks[deck]["mastered"] += 1

            if is_due:
                decks[deck]["due"] += 1

        formatted = []
        for name, stats in decks.items():
            formatted.append(
                f"[{name}] {stats['total']} cards "
                f"({stats['due']} due, {stats['new']} new, {stats['learning']} learning, {stats['mastered']} mastered)"
            )

        return {
            "success": True,
            "decks": decks,
            "count": len(decks),
            "formatted": "\n".join(formatted) if formatted else "No decks found",
            "response": f"Found {len(decks)} deck(s)\n" + "\n".join(formatted)
        }

    def deck_stats(self, deck: str = "default") -> dict:
        """Get detailed stats for a specific deck."""
        cards = self._load_cards()
        deck_cards = [c for c in cards if c.get("deck", "default") == deck]

        if not deck_cards:
            return {"success": True, "deck": deck, "total": 0, "response": f"Deck '{deck}' is empty or not found"}

        now = datetime.now()
        due = 0
        total_reviews = 0
        ease_sum = 0.0

        for c in deck_cards:
            try:
                if datetime.fromisoformat(c.get("next_review", "")) <= now:
                    due += 1
            except (ValueError, TypeError):
                due += 1
            total_reviews += len(c.get("review_history", []))
            ease_sum += c.get("ease_factor", 2.5)

        avg_ease = ease_sum / len(deck_cards)

        return {
            "success": True,
            "deck": deck,
            "total": len(deck_cards),
            "due": due,
            "total_reviews": total_reviews,
            "average_ease": round(avg_ease, 2),
            "response": f"Deck '{deck}': {len(deck_cards)} cards, {due} due, avg ease {avg_ease:.2f}"
        }

    def search_cards(self, query: str) -> dict:
        """Search card content."""
        if not query:
            return {"success": False, "error": "No search query provided"}

        cards = self._load_cards()
        query_lower = query.lower()
        matching = [
            c for c in cards
            if query_lower in c.get("front", "").lower()
            or query_lower in c.get("back", "").lower()
            or any(query_lower in t.lower() for t in c.get("tags", []))
        ]

        formatted = []
        for c in matching:
            formatted.append(f"[{c['id']}] Q: {c['front'][:60]} | A: {c['back'][:60]}")

        return {
            "success": True,
            "count": len(matching),
            "cards": matching,
            "formatted": "\n".join(formatted) if formatted else "No cards found",
            "response": f"Found {len(matching)} card(s) matching '{query}'"
        }

    def delete_card(self, card_id: str) -> dict:
        """Remove a flashcard."""
        if not card_id:
            return {"success": False, "error": "No card ID provided"}

        cards = self._load_cards()
        original_count = len(cards)
        cards = [c for c in cards if c.get("id") != card_id]

        if len(cards) == original_count:
            return {"success": False, "error": f"Card not found: {card_id}"}

        self._save_cards(cards)
        return {
            "success": True,
            "removed_id": card_id,
            "response": f"Deleted card {card_id}"
        }

    def due_count(self) -> dict:
        """Get count of cards due for review."""
        cards = self._load_cards()
        now = datetime.now()
        due = 0

        for c in cards:
            try:
                if datetime.fromisoformat(c.get("next_review", "")) <= now:
                    due += 1
            except (ValueError, TypeError):
                due += 1

        return {
            "success": True,
            "due": due,
            "total": len(cards),
            "response": f"{due} card(s) due for review out of {len(cards)} total"
        }

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a spaced repetition action."""
        action_lower = action.lower().strip()

        # Review
        if action_lower in ("review", "next", "study"):
            return self.review()

        # Due count
        if action_lower in ("due", "due_count", "how many due"):
            return self.due_count()

        # Stats / decks
        if action_lower in ("stats", "statistics", "decks", "list_decks"):
            return self.list_decks()

        # Deck-specific stats
        if action_lower.startswith("stats ") or action_lower.startswith("deck "):
            deck_name = action.split(None, 1)[-1].strip()
            return self.deck_stats(deck=deck_name)

        # Answer
        if action_lower.startswith("answer") or action_lower.startswith("rate"):
            card_id = kwargs.get("card_id")
            quality = kwargs.get("quality")
            if not card_id or quality is None:
                # Try to parse from action: "answer <id> <quality>"
                parts = action.split()
                if len(parts) >= 3:
                    card_id = card_id or parts[1]
                    quality = quality if quality is not None else parts[2]
            if card_id and quality is not None:
                return self.answer(card_id, int(quality))
            return {"success": False, "error": "Usage: answer <card_id> <quality 0-5>"}

        # Delete
        if action_lower.startswith("delete") or action_lower.startswith("remove"):
            card_id = kwargs.get("card_id")
            if not card_id:
                id_match = re.search(r'\b([a-f0-9]{8})\b', action)
                card_id = id_match.group(1) if id_match else None
            if card_id:
                return self.delete_card(card_id)
            return {"success": False, "error": "No card ID specified"}

        # Search
        if action_lower.startswith("search") or action_lower.startswith("find"):
            query = kwargs.get("query") or (action.split(None, 1)[-1] if len(action.split()) > 1 else "")
            return self.search_cards(query)

        # Auto-generate from text
        if action_lower.startswith("auto") or action_lower.startswith("generate"):
            text = kwargs.get("text") or (action.split(None, 1)[-1] if len(action.split()) > 1 else "")
            tags = kwargs.get("tags", [])
            deck = kwargs.get("deck", "default")
            return self.add_cards_from_text(text, tags=tags, deck=deck)

        # Add card
        if action_lower.startswith("add") or "front:" in action_lower:
            front = kwargs.get("front")
            back = kwargs.get("back")
            tags = kwargs.get("tags", [])
            deck = kwargs.get("deck", "default")

            if not front or not back:
                # Parse "add front:<q> back:<a>"
                front_match = re.search(r'front:\s*(.+?)(?:\s+back:|\s*$)', action, re.IGNORECASE)
                back_match = re.search(r'back:\s*(.+?)(?:\s+tags:|\s+deck:|\s*$)', action, re.IGNORECASE)
                front = front or (front_match.group(1).strip() if front_match else None)
                back = back or (back_match.group(1).strip() if back_match else None)

            if front and back:
                return self.add_card(front=front, back=back, tags=tags, deck=deck)
            return {"success": False, "error": "Usage: add front:<question> back:<answer>"}

        return {
            "success": False,
            "error": f"Unknown action: {action}. "
                     "Try: 'review', 'add front:<q> back:<a>', 'answer <id> <quality>', 'due', 'stats', 'search <query>'"
        }


# Singleton
spaced_repetition_tool = SpacedRepetitionTool()
