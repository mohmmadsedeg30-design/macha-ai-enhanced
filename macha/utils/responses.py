"""
🍵 Macha Smart Response System v3.0
- Math & Logic Engine
- Pattern Recognition
- Open-Source Model Integration (no API keys)
- Supports: English + Arabic
"""

import re
import os
import json
import random
from typing import Optional
from datetime import datetime


# ─────────────────────────────────────────────
# OPTIONAL: Load free open-source models
# Install with: pip install transformers torch
# ─────────────────────────────────────────────
try:
    from transformers import pipeline
    _QA_PIPE = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",   # ~300MB, free, no API key
        max_new_tokens=100
    )
    _HAS_TRANSFORMERS = True
    print("[OK] flan-t5-small loaded for smart responses")
except Exception as _e:
    _QA_PIPE = None
    _HAS_TRANSFORMERS = False
    print(f"[INFO] Transformers not available ({_e}). Using built-in engine.")


# ─────────────────────────────────────────────
# MATH & LOGIC ENGINE
# ─────────────────────────────────────────────

def _solve_math(text: str) -> Optional[str]:
    """Solve math expressions hidden inside any text."""

    # --- Algebra: solve for x ---
    algebra = re.search(
        r'([-\d.]+)\s*x\s*([+-]\s*[-\d.]+)?\s*=\s*([-\d.]+)', text, re.I
    )
    if algebra:
        try:
            a = float(algebra.group(1))
            b = float(algebra.group(2).replace(' ', '')) if algebra.group(2) else 0.0
            c = float(algebra.group(3))
            if a != 0:
                x = (c - b) / a
                x = int(x) if x == int(x) else round(x, 4)
                return f"x = {x}"
        except Exception:
            pass

    # --- Extract and evaluate pure math expression ---
    expr = re.sub(r'[^0-9+\-*/().%\s]', ' ', text)
    expr = re.sub(r'\s+', ' ', expr).strip()

    # Handle ^ as power
    raw = re.sub(r'(\d+)\s*\^\s*(\d+)', r'(\1**\2)', text)
    expr2 = re.sub(r'[^0-9+\-*/().%*\s]', ' ', raw).strip()

    for candidate in [expr2, expr]:
        candidate = candidate.strip()
        if len(candidate) < 1:
            continue
        if not any(op in candidate for op in ['+', '-', '*', '/', '%']):
            continue
        try:
            result = eval(candidate, {"__builtins__": {}})
            result = int(result) if isinstance(result, float) and result == int(result) else round(result, 6)
            return f"{result}"
        except Exception:
            pass

    return None


def _solve_logic(text: str) -> Optional[str]:
    """Evaluate boolean logic expressions."""
    t = text.upper()

    # NOT X
    m = re.search(r'NOT\s+(TRUE|FALSE)', t)
    if m:
        val = m.group(1) == 'TRUE'
        return "TRUE" if not val else "FALSE"

    # X AND Y
    m = re.search(r'(TRUE|FALSE)\s+AND\s+(TRUE|FALSE)', t)
    if m:
        a = m.group(1) == 'TRUE'
        b = m.group(2) == 'TRUE'
        return "TRUE" if (a and b) else "FALSE"

    # X OR Y
    m = re.search(r'(TRUE|FALSE)\s+OR\s+(TRUE|FALSE)', t)
    if m:
        a = m.group(1) == 'TRUE'
        b = m.group(2) == 'TRUE'
        return "TRUE" if (a or b) else "FALSE"

    # X XOR Y
    m = re.search(r'(TRUE|FALSE)\s+XOR\s+(TRUE|FALSE)', t)
    if m:
        a = m.group(1) == 'TRUE'
        b = m.group(2) == 'TRUE'
        return "TRUE" if (a ^ b) else "FALSE"

    # Comparison: 9 > 11 ?
    m = re.search(r'([-\d.]+)\s*(>=|<=|!=|>|<|==)\s*([-\d.]+)', text)
    if m:
        try:
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            ops = {'>=': a >= b, '<=': a <= b, '!=': a != b,
                   '>': a > b, '<': a < b, '==': a == b}
            return "TRUE" if ops[op] else "FALSE"
        except Exception:
            pass

    return None


def _solve_pattern(text: str) -> Optional[str]:
    """Detect numeric sequences and predict next value."""
    nums = re.findall(r'-?\d+\.?\d*', text)
    if len(nums) < 3:
        return None

    seq = [float(n) for n in nums]

    # Arithmetic sequence
    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    if len(set(round(d, 6) for d in diffs)) == 1:
        nxt = seq[-1] + diffs[0]
        nxt = int(nxt) if nxt == int(nxt) else round(nxt, 4)
        return str(nxt)

    # Geometric sequence
    if all(seq[i] != 0 for i in range(len(seq)-1)):
        ratios = [round(seq[i+1] / seq[i], 6) for i in range(len(seq)-1)]
        if len(set(ratios)) == 1:
            nxt = seq[-1] * ratios[0]
            nxt = int(nxt) if nxt == int(nxt) else round(nxt, 4)
            return str(nxt)

    # Fibonacci-like
    fib_check = all(
        abs(seq[i] - (seq[i-1] + seq[i-2])) < 0.01
        for i in range(2, len(seq))
    )
    if fib_check:
        nxt = int(seq[-1] + seq[-2])
        return str(nxt)

    return None


def _classify_question(text: str) -> Optional[str]:
    """Detect question intent for conditional / word-to-math problems."""
    t = text.lower()

    # IF x = N
    m = re.search(r'if\s+x\s*=\s*([-\d.]+)', t)
    if m:
        x = float(m.group(1))
        if x > 0:
            return "positive"
        elif x < 0:
            return "negative"
        else:
            return "zero"

    # "double" X
    m = re.search(r'double.*?(\d+)', t)
    if m:
        return str(int(m.group(1)) * 2)

    # "half of" X
    m = re.search(r'half\s+of\s+(\d+)', t)
    if m:
        val = float(m.group(1)) / 2
        return str(int(val) if val == int(val) else val)

    # "split X between N"
    m = re.search(r'split\s+(\d+)\s+between\s+(\d+)', t)
    if m:
        total, people = float(m.group(1)), float(m.group(2))
        each = total / people
        each = int(each) if each == int(each) else round(each, 2)
        return str(each)

    # "none" = 0
    if re.search(r'\bi have none\b|\bnone\b', t):
        return "0"

    # IF A = B AND B = N → A = N
    m = re.search(r'if\s+\w\s*=\s*\w\s+and\s+\w\s*=\s*([\d.]+)', t)
    if m:
        return m.group(1)

    # Percentage: 0.75 same as 75%?
    m = re.search(r'([\d.]+)\s+(?:same as|equal to|=)\s+([\d.]+)%', t)
    if m:
        val = float(m.group(1))
        pct = float(m.group(2)) / 100
        return "TRUE" if abs(val - pct) < 0.0001 else "FALSE"

    return None


# ─────────────────────────────────────────────
# OPEN-SOURCE MODEL QUERY
# ─────────────────────────────────────────────

def _query_transformers(text: str) -> Optional[str]:
    """Use flan-t5-small for general questions (free, offline after download)."""
    if not _HAS_TRANSFORMERS or _QA_PIPE is None:
        return None
    try:
        result = _QA_PIPE(text, max_new_tokens=80)
        answer = result[0]['generated_text'].strip()
        if answer and len(answer) > 1:
            return answer
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# KEYWORD FALLBACK BANK
# ─────────────────────────────────────────────

_FALLBACK = {
    "greet":     (["hello","hi","hey","هلا","مرحبا","سلام"],
                  ["Hello! 🍵 How can I help?", "Hey! What's on your mind?"]),
    "how_are":   (["how are you","شلونك","كيف حالك","how r u"],
                  ["I'm Macha — always learning! What do you need?"]),
    "who":       (["who are you","what are you","من أنت","ما اسمك"],
                  ["I'm Macha 🍵 — a small AI that learns from you!"]),
    "thanks":    (["thank","شكرا","مشكور","تسلم"],
                  ["You're welcome! 🍵", "No problem at all!"]),
    "bye":       (["bye","goodbye","مع السلامة","باي"],
                  ["Goodbye! 🍵 Come back anytime."]),
    "help":      (["help","assist","ساعدني","مساعدة"],
                  ["Of course! What do you need help with? 💪"]),
    "code":      (["code","programming","python","bug","برمجة","كود"],
                  ["I can help with code! What language or problem? 💻"]),
    "joke":      (["joke","نكتة","funny"],
                  ["Why don't programmers trust nature? Too many bugs! 🐛😂"]),
    "weather":   (["weather","طقس","جو"],
                  ["No weather sensors here 😅 — check a weather app!"]),
    "time":      (["what time","الوقت","الساعة"],
                  [f"Current time: {datetime.now().strftime('%I:%M %p')} ⏰"]),
}


def _keyword_response(text: str) -> Optional[str]:
    t = text.lower()
    for _, (patterns, responses) in _FALLBACK.items():
        if any(p in t for p in patterns):
            return random.choice(responses)
    return None


# ─────────────────────────────────────────────
# MAIN SMART RESPONSE SYSTEM
# ─────────────────────────────────────────────

class SmartResponseSystem:
    """
    Priority pipeline:
    1. Math solver
    2. Logic evaluator
    3. Pattern detector
    4. Word-to-math classifier
    5. flan-t5-small (open-source, offline)
    6. Keyword fallback
    7. Generic fallback
    """

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.knowledge_base = {}   # kept for compatibility with app.py stats
        self._load_learned()

    # ── Persistence ──────────────────────────

    def _learned_path(self):
        return os.path.join(self.data_dir, 'learned.json')

    def _load_learned(self):
        p = self._learned_path()
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)

    def _save_learned(self):
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self._learned_path(), 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)

    # ── Learning ─────────────────────────────

    def learn_response(self, input_text: str, correct_response: str):
        key = input_text.lower().strip()
        self.knowledge_base[key] = correct_response
        self._save_learned()
        print(f"[LEARN] '{input_text[:40]}' → '{correct_response[:40]}'")

    # ── Main Entry ───────────────────────────

    def get_response(self, user_input: str, context: str = "") -> str:
        text = user_input.strip()

        # 0. Check learned responses first
        learned = self.knowledge_base.get(text.lower())
        if learned:
            return learned

        # 1. Math solver
        math_ans = _solve_math(text)
        if math_ans:
            return math_ans

        # 2. Logic evaluator
        logic_ans = _solve_logic(text)
        if logic_ans:
            return logic_ans

        # 3. Pattern detector
        pattern_ans = _solve_pattern(text)
        if pattern_ans:
            return f"Next: {pattern_ans}"

        # 4. Word-to-math / conditional
        classified = _classify_question(text)
        if classified:
            return classified

        # 5. Open-source model (flan-t5-small)
        ai_ans = _query_transformers(text)
        if ai_ans:
            return ai_ans

        # 6. Keyword fallback
        kw = _keyword_response(text)
        if kw:
            return kw

        # 7. Generic
        return random.choice([
            "I'm still learning — could you rephrase that?",
            "Interesting! Tell me more. 📚",
            "I don't know yet, but I'll learn if you teach me! 🍵"
        ])

    # ── Compatibility helper ─────────────────

    def get_context(self, last_n: int = 5) -> str:
        return ""
  
