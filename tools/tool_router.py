"""
Nelson AI — Tool Call Parser
Detects and handles tool call tokens emitted by Nelson during generation.

Nelson can emit special tool call syntax mid-generation:
    <|search|>u Rwanda amateka<|/search|>
    <|fetch|>https://example.com<|/fetch|>
    <|wiki|>Kigali<|/wiki|>

The chat loop detects these, runs the real tool, injects the result
back into context, and continues generation.
"""

import re
from typing import Optional, Tuple
from tools.web_tools import run_tool


# ── Token patterns Nelson learns to emit ─────────────────────────
TOOL_PATTERNS = {
    "search":    re.compile(r"<\|search\|>(.*?)<\|/search\|>",    re.DOTALL),
    "fetch":     re.compile(r"<\|fetch\|>(.*?)<\|/fetch\|>",      re.DOTALL),
    "wikipedia": re.compile(r"<\|wiki\|>(.*?)<\|/wiki\|>",        re.DOTALL),
}

# Partial-match detector: Nelson started a tool call but hasn't closed it
PARTIAL_PATTERNS = {
    "search":    re.compile(r"<\|search\|>(.*)$",    re.DOTALL),
    "fetch":     re.compile(r"<\|fetch\|>(.*)$",     re.DOTALL),
    "wikipedia": re.compile(r"<\|wiki\|>(.*)$",      re.DOTALL),
}

# Tool result wrapper (injected back into context)
RESULT_TEMPLATE = "\n<|result|>\n{result}\n<|/result|>\n"

# Max times a tool can be called in one response (prevents loops)
MAX_TOOL_CALLS = 3


def extract_tool_call(text: str) -> Optional[Tuple[str, str]]:
    """
    Scan generated text for a complete tool call.
    Returns (tool_name, query) or None if no tool call found.
    """
    for tool_name, pattern in TOOL_PATTERNS.items():
        match = pattern.search(text)
        if match:
            query = match.group(1).strip()
            return tool_name, query
    return None


def has_partial_tool_call(text: str) -> bool:
    """
    Check if Nelson has started a tool call but not finished it.
    Used to decide whether to keep generating vs. stop + execute tool.
    """
    for pattern in PARTIAL_PATTERNS.values():
        if pattern.search(text):
            return True
    return False


def strip_tool_calls(text: str) -> str:
    """Remove tool call syntax from final output shown to user."""
    result = text
    # Remove complete tool calls
    for pattern in TOOL_PATTERNS.values():
        result = pattern.sub("", result)
    # Remove result blocks (already shown inline)
    result = re.sub(r"<\|result\|>.*?<\|/result\|>", "", result, flags=re.DOTALL)
    # Remove any partial/orphaned tags
    result = re.sub(r"<\|/?(search|fetch|wiki|result)\|>", "", result)
    return result.strip()


def format_tool_result(tool_name: str, query: str, result: str) -> str:
    """Format a tool result for injection into the model's context."""
    return RESULT_TEMPLATE.format(result=result[:2000])  # Cap to avoid context overflow


class ToolRouter:
    """
    Handles tool call detection and execution during generation.
    Maintains per-response call count to prevent infinite loops.
    """

    def __init__(self):
        self.call_count = 0
        self.call_log = []

    def reset(self):
        """Call at start of each new response."""
        self.call_count = 0
        self.call_log = []

    def can_call(self) -> bool:
        return self.call_count < MAX_TOOL_CALLS

    def process(self, generated_text: str) -> Optional[str]:
        """
        If generated_text contains a tool call, execute it and return
        a formatted result string to append to context.
        Returns None if no tool call was found or limit reached.
        """
        if not self.can_call():
            return None

        call = extract_tool_call(generated_text)
        if call is None:
            return None

        tool_name, query = call
        if not query:
            return None

        self.call_count += 1
        self.call_log.append({"tool": tool_name, "query": query})

        # Execute the tool
        result = run_tool(tool_name, query)

        return format_tool_result(tool_name, query, result)

    def get_system_prompt_addon(self) -> str:
        """
        Returns the tool-use instructions to inject into system prompt.
        Nelson learns this format during fine-tuning.
        """
        return (
            "Ufite ubushobozi bwo gushakisha kuri internet. "  # You have internet access.
            "Iyo ukeneye amakuru, koresha ubu buryo:\n"        # When you need info, use:
            "  <|search|>ibibazo byawe<|/search|>    - Gushakisha kuri internet\n"
            "  <|wiki|>ikibazo<|/wiki|>              - Gushakisha kuri Wikipedia\n"
            "  <|fetch|>https://...<|/fetch|>        - Gusoma urubuga\n"
            "Subiza GUSA mu Kinyarwanda."  # Answer ONLY in Kinyarwanda.
        )


# Singleton router instance
router = ToolRouter()
