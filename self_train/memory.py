"""
Nelson AI — Conversation Memory
Persists all conversations to disk for self-evolution training.
Also tracks Nelson's long-term knowledge and learned facts.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional


MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)

CONVERSATIONS_FILE = MEMORY_DIR / "conversations.jsonl"
FACTS_FILE         = MEMORY_DIR / "learned_facts.json"
STATS_FILE         = MEMORY_DIR / "stats.json"


# ─────────────────────────────────────────────────────────────────
# Conversation Logger
# ─────────────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Logs every conversation turn to disk.
    These logs are later used for self-evolution fine-tuning.
    """

    def __init__(self):
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.session_start = datetime.now().isoformat()
        self.turns = []
        self._load_stats()

    def _load_stats(self):
        if STATS_FILE.exists():
            with open(STATS_FILE, "r") as f:
                self.stats = json.load(f)
        else:
            self.stats = {
                "total_conversations": 0,
                "total_turns": 0,
                "total_tokens_generated": 0,
                "evolution_count": 0,
                "last_evolution": None,
                "created_at": datetime.now().isoformat(),
            }

    def _save_stats(self):
        with open(STATS_FILE, "w") as f:
            json.dump(self.stats, f, indent=2)

    def log_turn(self, role: str, content: str, tool_calls: list = None, metadata: dict = None):
        """Log a single conversation turn."""
        turn = {
            "role":       role,
            "content":    content,
            "timestamp":  datetime.now().isoformat(),
            "session_id": self.session_id,
        }
        if tool_calls:
            turn["tool_calls"] = tool_calls
        if metadata:
            turn["metadata"] = metadata

        self.turns.append(turn)
        self.stats["total_turns"] += 1
        self.stats["total_tokens_generated"] += len(content.split())

    def save_session(self):
        """Flush entire session to disk as one JSONL entry."""
        if not self.turns:
            return

        session = {
            "session_id":    self.session_id,
            "started_at":    self.session_start,
            "ended_at":      datetime.now().isoformat(),
            "turn_count":    len(self.turns),
            "turns":         self.turns,
        }

        with open(CONVERSATIONS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(session, ensure_ascii=False) + "\n")

        self.stats["total_conversations"] += 1
        self._save_stats()
        print(f"\n  💾 Conversation saved (session {self.session_id})")

    def get_training_examples(self, min_turns: int = 2) -> list[dict]:
        """
        Load all saved conversations and format them as training examples.
        Used by the self-evolution engine.
        """
        examples = []
        if not CONVERSATIONS_FILE.exists():
            return examples

        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    session = json.loads(line.strip())
                    if session.get("turn_count", 0) >= min_turns:
                        # Format into training text
                        text = self._format_as_training_text(session["turns"])
                        if text:
                            examples.append({"text": text, "session_id": session["session_id"]})
                except json.JSONDecodeError:
                    continue

        return examples

    def _format_as_training_text(self, turns: list) -> str:
        """Convert conversation turns to training format."""
        parts = []
        for turn in turns:
            role    = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if not content:
                continue
            if role == "user":
                parts.append(f"<|user|> {content} <|eos|>")
            elif role == "nelson":
                parts.append(f"<|nelson|> {content} <|eos|>")
        return " ".join(parts)

    def get_stats_display(self) -> str:
        s = self.stats
        lines = [
            f"  📊 Nelson Memory Stats",
            f"  {'─' * 35}",
            f"  Conversations   : {s.get('total_conversations', 0):,}",
            f"  Total turns     : {s.get('total_turns', 0):,}",
            f"  Words generated : {s.get('total_tokens_generated', 0):,}",
            f"  Evolutions done : {s.get('evolution_count', 0)}",
            f"  Last evolution  : {s.get('last_evolution', 'Never')}",
            f"  Session         : {self.session_id}",
        ]
        return "\n".join(lines)

    def mark_evolution(self):
        self.stats["evolution_count"] += 1
        self.stats["last_evolution"] = datetime.now().isoformat()
        self._save_stats()


# ─────────────────────────────────────────────────────────────────
# Facts Store — Nelson remembers key things it has learned
# ─────────────────────────────────────────────────────────────────

class FactsStore:
    """
    Key-value store for things Nelson has explicitly learned.
    Persisted to disk. Injected into context as background knowledge.
    """

    def __init__(self):
        self.facts = self._load()

    def _load(self) -> dict:
        if FACTS_FILE.exists():
            with open(FACTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"facts": [], "sources": []}

    def save(self):
        with open(FACTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)

    def add_fact(self, fact: str, source: str = "conversation"):
        """Add a learned fact."""
        entry = {
            "fact":      fact,
            "source":    source,
            "learned_at": datetime.now().isoformat(),
        }
        if fact not in [f["fact"] for f in self.facts.get("facts", [])]:
            self.facts.setdefault("facts", []).append(entry)
            self.save()

    def get_context_snippet(self, max_facts: int = 5) -> str:
        """Return recent facts as a context injection."""
        recent = self.facts.get("facts", [])[-max_facts:]
        if not recent:
            return ""
        lines = ["Nelson yize ibi byinshi (Nelson has learned):"]
        for f in recent:
            lines.append(f"  - {f['fact']}")
        return "\n".join(lines)

    def count(self) -> int:
        return len(self.facts.get("facts", []))
