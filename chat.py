"""
Nelson AI — Terminal Chat Interface (Full Version)
Features:
  • Internet access (search, Wikipedia, page fetching)
  • Tool-calling: Nelson can emit <|search|>...<|/search|> tokens
  • Streaming token-by-token output
  • Conversation memory (saved to disk)
  • Auto self-evolution after configurable number of conversations
  • Conversation history with rolling context window

Usage:
    python chat.py                          # Load latest checkpoint
    python chat.py --checkpoint path/to.pt  # Load specific checkpoint
    python chat.py --temperature 0.9        # Adjust creativity
    python chat.py --no-internet            # Disable web access
    python chat.py --no-evolve              # Disable auto-evolution

Commands during chat:
    /search <query>  — Manually force a web search
    /wiki <topic>    — Search Wikipedia in Kinyarwanda
    /evolve          — Manually trigger evolution now
    /stats           — Show memory and evolution stats
    /clear           — Clear conversation history
    /settings        — Show generation settings
    /temp /topp /topk /tokens — Adjust generation params
    /help            — Show all commands
    /quit or /q      — Exit (saves conversation first)
"""

import os
import sys
import re
import json
import argparse
import threading
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))

from model.config import NelsonConfig, NELSON_MINI
from model.architecture import Nelson
from tools.tool_router import ToolRouter, extract_tool_call, strip_tool_calls, has_partial_tool_call
from tools.web_tools import run_tool
from tools.lang_detect import detect_language, get_lang_label
from self_train.memory import ConversationMemory, FactsStore


# ── ANSI Colors ────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    WHITE   = "\033[97m"
    PURPLE  = "\033[95m"
    ORANGE  = "\033[38;5;208m"


# ── Auto-evolution trigger ────────────────────────────────────────
EVOLVE_EVERY_N_CONVERSATIONS = 5   # Run evolution after every 5 chats


def print_banner():
    print(f"""
{C.CYAN}{C.BOLD}
  ███╗   ██╗███████╗██╗     ███████╗ ██████╗ ███╗   ██╗
  ████╗  ██║██╔════╝██║     ██╔════╝██╔═══██╗████╗  ██║
  ██╔██╗ ██║█████╗  ██║     ███████╗██║   ██║██╔██╗ ██║
  ██║╚██╗██║██╔══╝  ██║     ╚════██║██║   ██║██║╚██╗██║
  ██║ ╚████║███████╗███████╗███████║╚██████╔╝██║ ╚████║
  ╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
{C.RESET}
  {C.WHITE}Kinyarwanda AI — Trained from Scratch{C.RESET}
  {C.ORANGE}⚡ Self-Evolving  •  🌐 Internet Access{C.RESET}
  {C.DIM}Type /help for commands{C.RESET}
""")


def print_help():
    print(f"""
{C.YELLOW}  ── Web & Knowledge ─────────────────────────────────{C.RESET}
    /search <query>   Force a web search
    /wiki <topic>     Search Kinyarwanda Wikipedia
    /fetch <url>      Fetch and read a webpage

{C.YELLOW}  ── Evolution ────────────────────────────────────────{C.RESET}
    /evolve           Run self-evolution now (trains on your chats)
    /stats            Show memory & evolution statistics

{C.YELLOW}  ── Conversation ─────────────────────────────────────{C.RESET}
    /clear            Clear conversation history
    /save             Save current conversation
    /history          Show recent conversation turns

{C.YELLOW}  ── Generation Settings ──────────────────────────────{C.RESET}
    /settings         Show all settings
    /temp <n>         Temperature (0.1=focused, 1.5=creative)
    /topp <n>         Top-p nucleus sampling (default 0.9)
    /topk <n>         Top-k sampling (default 50)
    /tokens <n>       Max new tokens per response

{C.YELLOW}  ── System ───────────────────────────────────────────{C.RESET}
    /help             Show this help
    /quit /q          Exit (conversation is auto-saved)
""")


# ─────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────

def find_best_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    """Find the best checkpoint: prefer evolved > latest step."""
    ckpt_dir = Path(checkpoint_dir)
    evolved = sorted(ckpt_dir.glob("nelson_evolved_*.pt"))
    if evolved:
        return str(evolved[-1])
    step_ckpts = sorted(ckpt_dir.glob("nelson_step_*.pt"))
    if step_ckpts:
        return str(step_ckpts[-1])
    raise FileNotFoundError(
        "No checkpoint found. Run: python training/trainer.py"
    )


def load_model_and_tokenizer(checkpoint_path: str):
    print(f"\n  {C.DIM}Loading model ...{C.RESET}", end="", flush=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = NelsonConfig(**ckpt["config"]) if "config" in ckpt else NELSON_MINI
    model = Nelson(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    step = ckpt.get("step", "?")
    evol = ckpt.get("evolution_num", None)
    label = f"evolution #{evol}" if evol else f"step {step}"
    print(f" {C.GREEN}✓{C.RESET} ({model.count_params()/1e6:.1f}M params, {label})")

    print(f"  {C.DIM}Loading tokenizer ...{C.RESET}", end="", flush=True)
    tok_path = Path("tokenizer/nelson_tokenizer/tokenizer.json")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    tokenizer.no_padding()
    print(f" {C.GREEN}✓{C.RESET} (vocab={tokenizer.get_vocab_size():,})")

    return model, config, tokenizer


# ─────────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────────

def build_system_prompt(user_lang: str = "rw") -> str:
    """
    Build a language-aware system prompt.
    Nelson understands rw/en/fr but adapts response language to the user.
    """
    base = (
        "<|system|> "
        "Witwa Nelson. Uri AI yigiye Ikinyarwanda, Icyongereza, na Igifaransa. "
        "Subiza mu rurimi umuntu akoresha: Ikinyarwanda niba abajije mu Kinyarwanda, "
        "Icyongereza niba abajije mu Cyongereza, Igifaransa niba abajije mu Gifaransa. "
        "Ufite uburenganzira bwo gushakisha kuri internet. "
        "Iyo ukeneye amakuru, koresha: <|search|>ikibazo<|/search|> "
        "cyangwa <|wiki|>ikibazo<|/wiki|>. "
        "Subiza neza, igitabo cyane. "
    )
    lang_hints = {
        "en": "The user is writing in English. Reply in English.",
        "fr": "L'utilisateur écrit en français. Réponds en français.",
        "rw": "Umutumiwe akoresha Ikinyarwanda. Subiza mu Kinyarwanda.",
    }
    hint = lang_hints.get(user_lang, lang_hints["rw"])
    return base + hint + " <|eos|> "


# Keep a fixed base for backwards compat
SYSTEM_PROMPT = build_system_prompt("rw")


def build_prompt(
    conversation: list[dict],
    facts_snippet: str = "",
    max_turns: int = 10,
    user_lang: str = "rw",
) -> str:
    """Build full input prompt string from conversation history."""
    parts = [build_system_prompt(user_lang)]

    if facts_snippet:
        parts.append(f"<|system|> {facts_snippet} <|eos|> ")

    # Rolling window of last N turns
    turns = conversation[-max_turns * 2:]
    for turn in turns:
        if turn["role"] == "user":
            parts.append(f"<|user|> {turn['content']} <|eos|> ")
        elif turn["role"] == "nelson":
            parts.append(f"<|nelson|> {turn['content']} <|eos|> ")
        elif turn["role"] == "result":
            parts.append(f"<|result|> {turn['content']} <|/result|> ")

    parts.append("<|nelson|>")
    return "".join(parts)


def encode_prompt(text: str, tokenizer: Tokenizer) -> torch.Tensor:
    enc = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor([enc.ids], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────
# Streaming Generation with Tool-Calling
# ─────────────────────────────────────────────────────────────────

def generate_with_tools(
    model: Nelson,
    tokenizer: Tokenizer,
    conversation: list[dict],
    facts_store: FactsStore,
    device: torch.device,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.2,
    internet_enabled: bool = True,
    user_lang: str = "rw",
) -> str:
    """
    Generate Nelson's response with:
    - Streaming output token-by-token
    - Automatic tool-call detection and execution
    - Context injection of tool results
    """
    router = ToolRouter()
    router.reset()

    eos_id   = tokenizer.token_to_id("<|eos|>")
    user_id  = tokenizer.token_to_id("<|user|>")

    facts_snippet = facts_store.get_context_snippet()
    full_response_text = ""

    print(f"\n  {C.GREEN}{C.BOLD}Nelson:{C.RESET} ", end="", flush=True)

    # We may loop if Nelson uses tools
    max_loops = 4
    for loop in range(max_loops):
        prompt_text = build_prompt(conversation, facts_snippet, user_lang=user_lang)
        input_ids = encode_prompt(prompt_text, tokenizer).to(device)

        model.eval()
        loop_ids = []
        loop_text = ""

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # Crop context
                ctx = input_ids[:, -model.config.context_length:]
                logits, _ = model(ctx)
                logits = logits[:, -1, :] / temperature

                # Repetition Penalty
                if repetition_penalty > 1.0:
                    score = torch.gather(logits, 1, input_ids)
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits.scatter_(1, input_ids, score)

                # Top-K
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-P
                if top_p < 1.0:
                    sl, si = torch.sort(logits, descending=True)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    remove = cp - F.softmax(sl, dim=-1) > top_p
                    sl[remove] = float("-inf")
                    logits = torch.zeros_like(logits).scatter_(1, si, sl)

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                nid = next_id.item()

                # Stop tokens
                if nid in (eos_id, user_id):
                    break

                loop_ids.append(nid)
                input_ids = torch.cat([input_ids, next_id], dim=1)

                token_text = tokenizer.decode([nid])
                loop_text += token_text

                # Hide tool call tags from display but don't stop yet
                display_token = token_text
                if "<|" not in loop_text[-20:]:  # Only print clean tokens
                    print(display_token, end="", flush=True)

                # Check if Nelson completed a tool call
                if internet_enabled and router.can_call():
                    result_text = router.process(loop_text)
                    if result_text:
                        # Show tool execution indicator
                        tool_call = router.call_log[-1]
                        print(f"\n\n  {C.ORANGE}🔍 Searching: \"{tool_call['query']}\" ...{C.RESET}")
                        # Already executed — result_text has the result
                        actual_result = run_tool(tool_call["tool"], tool_call["query"])
                        print(f"  {C.DIM}[Result injected into context]{C.RESET}\n")
                        print(f"  {C.GREEN}{C.BOLD}Nelson:{C.RESET} ", end="", flush=True)

                        # Add result to conversation context and re-generate
                        conversation.append({
                            "role": "result",
                            "content": actual_result[:1500],
                        })
                        break  # Restart generation loop with updated context

        full_response_text += strip_tool_calls(loop_text)

        # If no tool was called in this loop, we're done
        if len(router.call_log) <= loop:
            break

    print()  # Newline after response
    return full_response_text.strip()


# ─────────────────────────────────────────────────────────────────
# Background Evolution Thread
# ─────────────────────────────────────────────────────────────────

def trigger_evolution_async():
    """Run evolution in background so chat stays responsive."""
    def _run():
        try:
            from self_train.evolve import evolve
            print(f"\n  {C.ORANGE}⚡ Background evolution starting ...{C.RESET}")
            evolve(n_steps=200, collect_web=True)
            print(f"\n  {C.GREEN}✅ Evolution complete! Nelson has improved.{C.RESET}")
            print(f"  {C.DIM}Restart chat.py to load the evolved model.{C.RESET}\n")
        except Exception as e:
            print(f"\n  {C.RED}Evolution error: {e}{C.RESET}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


# ─────────────────────────────────────────────────────────────────
# Main Chat Loop
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chat with Nelson AI")
    parser.add_argument("--checkpoint",  type=str,   default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p",       type=float, default=0.9)
    parser.add_argument("--top-k",       type=int,   default=50)
    parser.add_argument("--max-tokens",  type=int,   default=256)
    parser.add_argument("--no-internet", action="store_true", help="Disable internet access")
    parser.add_argument("--no-evolve",   action="store_true", help="Disable auto-evolution")
    args = parser.parse_args()

    # Mutable settings
    settings = {
        "temperature":    args.temperature,
        "top_p":          args.top_p,
        "top_k":          args.top_k,
        "max_tokens":     args.max_tokens,
        "internet":       not args.no_internet,
        "auto_evolve":    not args.no_evolve,
    }

    # ── Set Windows Taskbar Icon ─────────────────────────────────────
    try:
        import ctypes
        # Separate this app from standard Python/CMD windows in the taskbar
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("nelson.ai.chat.1")
        
        # Change the actual console window icon
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            icon_path = str(Path(__file__).parent / "nelson_icon.ico")
            if os.path.exists(icon_path):
                LR_LOADFROMFILE = 0x0010
                IMAGE_ICON = 1
                WM_SETICON = 0x0080
                hicon = ctypes.windll.user32.LoadImageW(0, icon_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE)
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 0, hicon) # Small icon
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, 1, hicon) # Big icon
    except Exception:
        pass # Silently fail if not on Windows or missing icon

    print_banner()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  {C.DIM}Device: {device} ({gpu_info}){C.RESET}")
    print(f"  {C.DIM}Internet: {'✓ enabled' if settings['internet'] else '✗ disabled'}{C.RESET}")
    print(f"  {C.DIM}Auto-evolve: {'✓ every ' + str(EVOLVE_EVERY_N_CONVERSATIONS) + ' chats' if settings['auto_evolve'] else '✗ disabled'}{C.RESET}")

    # Load model + tokenizer
    try:
        ckpt_path = args.checkpoint or find_best_checkpoint()
        model, config, tokenizer = load_model_and_tokenizer(ckpt_path)
        model = model.to(device)
    except FileNotFoundError as e:
        print(f"\n  {C.RED}✗ {e}{C.RESET}\n")
        sys.exit(1)

    # Memory systems
    memory     = ConversationMemory()
    facts      = FactsStore()

    # State
    conversation       = []
    session_turn_count = 0
    conversation_count = memory.stats.get("total_conversations", 0)

    print(f"\n  {C.DIM}{'─' * 60}{C.RESET}")
    print(f"  {C.CYAN}Muraho! / Hello! / Bonjour! — Ndi Nelson.{C.RESET}")
    print(f"  {C.DIM}Ndashimishwa no gutumanahana nawe mu Kinyarwanda, Icyongereza, cyangwa Igifaransa.{C.RESET}")
    print(f"  {C.DIM}I speak Kinyarwanda, English, and French. Je parle kinyarwanda, anglais et français.{C.RESET}")
    if settings["internet"]:
        print(f"  {C.ORANGE}🌐 Internet access enabled — I can search the web for you.{C.RESET}")
    print(f"  {C.DIM}{'─' * 60}{C.RESET}\n")

    while True:
        # ── User input ────────────────────────────────────────────
        try:
            user_input = input(f"  {C.BLUE}{C.BOLD}Wowe:{C.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {C.DIM}Murabeho! Saving conversation ...{C.RESET}")
            memory.save_session()
            break

        if not user_input:
            continue

        # ── Commands ─────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd   = parts[0].lower()
            arg   = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("quit", "q", "exit"):
                print(f"\n  {C.DIM}Murabeho! Saving conversation ...{C.RESET}")
                memory.save_session()
                print(f"  {C.DIM}Bye!{C.RESET}\n")
                break

            elif cmd == "help":
                print_help()

            elif cmd == "clear":
                conversation.clear()
                print(f"  {C.DIM}✓ Conversation history cleared.{C.RESET}")

            elif cmd == "save":
                memory.save_session()

            elif cmd == "history":
                if not conversation:
                    print(f"  {C.DIM}No history yet.{C.RESET}")
                else:
                    print()
                    for t in conversation[-10:]:
                        role_label = f"{C.BLUE}Wowe{C.RESET}" if t["role"] == "user" else f"{C.GREEN}Nelson{C.RESET}"
                        print(f"  {role_label}: {t['content'][:120]}")
                    print()

            elif cmd == "stats":
                print(f"\n{memory.get_stats_display()}")
                print(f"  Learned facts : {facts.count()}")
                print()

            elif cmd == "settings":
                print(f"""
  {C.YELLOW}  Settings:{C.RESET}
      temperature : {settings['temperature']}
      top_p       : {settings['top_p']}
      top_k       : {settings['top_k']}
      max_tokens  : {settings['max_tokens']}
      internet    : {settings['internet']}
      auto_evolve : {settings['auto_evolve']}
      model       : {ckpt_path}
      device      : {device}
""")
            elif cmd == "temp" and arg:
                settings["temperature"] = float(arg)
                print(f"  {C.DIM}Temperature → {settings['temperature']}{C.RESET}")

            elif cmd == "topp" and arg:
                settings["top_p"] = float(arg)
                print(f"  {C.DIM}top_p → {settings['top_p']}{C.RESET}")

            elif cmd == "topk" and arg:
                settings["top_k"] = int(arg)
                print(f"  {C.DIM}top_k → {settings['top_k']}{C.RESET}")

            elif cmd == "tokens" and arg:
                settings["max_tokens"] = int(arg)
                print(f"  {C.DIM}max_tokens → {settings['max_tokens']}{C.RESET}")

            elif cmd == "search" and arg:
                print(f"\n  {C.ORANGE}🔍 Searching: {arg}{C.RESET}")
                result = run_tool("search", arg)
                print(f"\n{result}\n")

            elif cmd == "wiki" and arg:
                print(f"\n  {C.ORANGE}📖 Wikipedia: {arg}{C.RESET}")
                result = run_tool("wikipedia", arg)
                print(f"\n{result}\n")

            elif cmd == "fetch" and arg:
                print(f"\n  {C.ORANGE}🌐 Fetching: {arg}{C.RESET}")
                result = run_tool("fetch", arg)
                print(f"\n{result[:2000]}\n")

            elif cmd == "evolve":
                print(f"\n  {C.ORANGE}⚡ Starting evolution in background ...{C.RESET}")
                print(f"  {C.DIM}(Chat stays available. Restart after completion to load evolved model.){C.RESET}\n")
                memory.save_session()
                trigger_evolution_async()

            else:
                print(f"  {C.RED}Unknown command. Type /help.{C.RESET}")

            continue

        # ── Normal conversation turn ──────────────────────────────
        session_turn_count += 1
        memory.log_turn("user", user_input)
        conversation.append({"role": "user", "content": user_input})

        # Detect user's language for this message
        user_lang = detect_language(user_input)
        lang_label = get_lang_label(user_lang)
        if user_lang != "rw":
            print(f"  {C.DIM}[Detected: {lang_label}]{C.RESET}")

        try:
            response = generate_with_tools(
                model, tokenizer, conversation, facts,
                device=device,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                max_new_tokens=settings["max_tokens"],
                internet_enabled=settings["internet"],
                user_lang=user_lang,
            )

            conversation.append({"role": "nelson", "content": response})
            memory.log_turn("nelson", response, tool_calls=None)

        except torch.cuda.OutOfMemoryError:
            print(f"\n  {C.RED}✗ GPU out of memory. Clearing context ...{C.RESET}")
            conversation = conversation[-4:]  # Keep last 2 turns
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"\n  {C.RED}✗ Error: {e}{C.RESET}")
            continue

        # Keep conversation window manageable
        if len(conversation) > 30:
            conversation = conversation[-30:]

        print()

        # ── Auto-evolution trigger ────────────────────────────────
        if settings["auto_evolve"] and session_turn_count % (EVOLVE_EVERY_N_CONVERSATIONS * 2) == 0:
            print(f"  {C.DIM}[Auto-evolution scheduled in background...]{C.RESET}")
            memory.save_session()
            trigger_evolution_async()


if __name__ == "__main__":
    main()
