"""Readable layout of a question: sub-parts on their own lines and code in indented code blocks.

Questions come from scanned PDFs (one long paragraph) or from the AI (sometimes still one paragraph),
so every screen and the PDF export run the text through this module:

    restore_parts(text)   puts "i. / ii. / (a) / (b) / 1. / 2." sub-parts on their own lines, so the
                          instruction sentence is followed by a line break and every part starts a new line
    layout_blocks(text)   [("text", "..."), ("code", "...")]: code (also code squashed into a sentence)
                          is found, split into lines and indented by its braces
    beautify_code(code)   one statement per line, indented by brace depth (4 spaces)
    blocks_html(text)     the same as safe HTML for the Streamlit pages

Plain rules, no AI. No word of the question is ever changed or dropped, only line breaks and the
indentation of code are added. Everything is idempotent.
"""
from __future__ import annotations

import html
import re

# ----------------------------------------------------------------------------- sub-parts
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
          "xi": 11, "xii": 12}
_ROMAN_TOK = r"(?:xii|xi|x|ix|viii|vii|vi|iv|v|iii|ii|i)"
_ROMAN_RX = re.compile(rf"(?<![\w.])(?:\((?P<a>{_ROMAN_TOK})\)|(?P<b>{_ROMAN_TOK})[.)])(?!\w|\.\w)(?=\s|$)", re.I)
_LETTER_RX = re.compile(r"(?<![\w.])(?:\((?P<a>[A-Za-z])\)|(?P<b>[A-Za-z])[.)])(?=\s+\S)")
_NUMBER_RX = re.compile(r"(?<![\w.])(?:\((?P<a>\d{1,2})\)|(?P<b>\d{1,2})[.)])(?=\s+[A-Za-z(\"'])")
_SENTENCE_END = ".?!:;)]}"


def _roman(tok: str):
    return _ROMAN.get(tok.lower())


def _letter(tok: str):
    return ord(tok.lower()) - 96 if len(tok) == 1 and tok.isalpha() else None


def _number(tok: str):
    return int(tok) if tok.isdigit() else None


def _chain(text: str, rx: re.Pattern, value) -> str:
    """Break the line before every label of ONE consecutive series (i, ii, iii / a, b, c / 1, 2, 3).
    A series must start at 1, count up by one and have at least two labels, so "i.e." or "Plan A. Plan B."
    are left alone."""
    hits, expected, case = [], 1, None
    for m in rx.finditer(text):
        tok = m.group("a") or m.group("b")
        if value(tok) != expected:
            continue
        if case is not None and tok.isupper() != case:
            continue
        if not hits:                                  # the first label must start a sentence (or the text)
            before = text[:m.start()].rstrip()
            if before and before[-1] not in _SENTENCE_END:
                continue
        if case is None:
            case = tok.isupper()
        hits.append(m)
        expected += 1
    if len(hits) < 2:
        return text
    out, pos = [], 0
    for m in hits:
        out.append(text[pos:m.start()].rstrip(" \t"))
        pos = m.start()
    out.append(text[pos:])
    return "\n".join(part for k, part in enumerate(out) if k or part.strip())


def restore_parts(text: str) -> str:
    """Instruction sentence, then every sub-part (i, ii, a, b, 1, 2 ...) on its own line."""
    lines = []
    for line in str(text or "").split("\n"):
        for rx, value in ((_ROMAN_RX, _roman), (_LETTER_RX, _letter), (_NUMBER_RX, _number)):
            line = _chain(line, rx, value)
        lines.append(line)
    return "\n".join(lines)


# ----------------------------------------------------------------------------- code detection
_TYPES = (r"(?:(?:unsigned|const|static|virtual|inline|friend|explicit)\s+)*"
          r"(?:int|void|char|float|double|bool|long|short|string|auto|size_t)")
_ANCHORS = re.compile("|".join([
    r"#\s*include\b", r"#\s*define\b", r"\busing\s+namespace\b",
    r"\b(?:class|struct|union)\s+\w+\s*(?::\s*(?:public|private|protected)?\s*\w+(?:\s*,\s*(?:public|private|protected)?\s*\w+)*)?\s*\{",
    r"\btemplate\s*<[^>]*>",
    r"\b(?:public|private|protected)\s*:(?!:)",
    rf"\b{_TYPES}\s*[*&]*\s*[\w:~]+\s*\([^;{{}}]*\)\s*(?:const\s*)?\{{",
    r"\b(?:int|void)\s+main\s*\(",
    r"\b(?:cout|cin|cerr)\s*(?:<<|>>)",
    r"\b(?:printf|scanf)\s*\(",
    rf"\b{_TYPES}\s+[*&]*\s*\w+\s*(?:=[^=;]+|(?:\[[^\]]*\])+)?;",
]))
_CONTINUES = re.compile(r"""^(?:
    [}{;] | // | /\* | \#\s*(?:include|define)
  | (?:using|namespace|template|typedef|class|struct|union|enum|public|private|protected|virtual|static|const
       |inline|friend|return|if|else|for|while|do|switch|case|break|continue|delete|new|cout|cin|cerr|printf
       |scanf|int|void|char|float|double|bool|long|short|string|auto|unsigned)\b
  | [A-Za-z_]\w*(?:::\w+)*(?:<[^>\n]*>)?[*&]*(?:\(|\[|\.|->|\+\+|--|<<|>>)
  | [A-Za-z_]\w*\s*(?:=(?!=)|\+=|-=)
  | [A-Za-z_]\w*[*&]*\s+[*&]*[a-z_]\w*\s*(?:=(?!=)|;|\([^()\n]{0,30}\)\s*;|\[)
  )""", re.X)
_PREFIX_CODE = re.compile(r"[{};=]|<<|>>|->|::|\w\(|^\s*(?:for|while|if|else|switch|do|return)\b|\(\s*$")
_PROSE_LINE = re.compile(r"[{}]|<<|>>|->|::|\+\+|==")


def _scan_end(s: str, start: int) -> int:
    """End of the code that starts at s[start]: the end of the last statement or block."""
    n, i, depth, par, quote = len(s), start, 0, 0, None
    while i < n:
        c = s[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote or c == "\n":
                quote = None
            i += 1
            continue
        if s.startswith("//", i):
            j = s.find("\n", i)
            i = n if j < 0 else j
            continue
        if c in "\"'":
            quote = c
        elif c == "(":
            par += 1
        elif c == ")":
            par = max(0, par - 1)
        elif c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        i += 1
        if c in "};" and depth == 0 and par == 0:
            rest = s[i:].lstrip(" \t")
            if not rest or rest[0] == "\n" or not _CONTINUES.match(rest):
                return i
    return n


def _split_line(line: str) -> list[tuple[str, str]]:
    """One line -> [("text", ...), ("code", ...)] (code that is squashed into a sentence is cut out)."""
    parts, rest = [], line
    while rest.strip():
        m = _ANCHORS.search(rest)
        if not m:
            parts.append(("text", rest))
            break
        prefix = rest[:m.start()]
        start = 0 if (not prefix.strip() or _PREFIX_CODE.search(prefix)) else m.start()
        if start:
            parts.append(("text", prefix.rstrip()))
        end = _scan_end(rest, start)
        parts.append(("code", rest[start:end].strip()))
        rest = rest[end:]
    return parts or [("text", line)]


def _weak(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s in ("{", "}", "};") or s.startswith("//"):
        return True
    if s[-1] not in ";{}":
        return False
    words = re.findall(r"[A-Za-z]{2,}", s)
    return not (len(words) >= 7 and s[0].isupper() and not _PROSE_LINE.search(s))


def _braces(s: str) -> int:
    s = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|//.*", "", s)
    return s.count("{") - s.count("}")


def layout_blocks(text: str) -> list[tuple[str, str]]:
    """[("text", ...), ("code", ...)] in reading order. Code is already beautified."""
    t = str(text or "").replace("\r", "\n").replace("\u00a0", " ")
    segs: list[list] = []
    for line in t.split("\n"):
        for kind, s in _split_line(line):
            segs.append([kind, s])
    # weak lines (end with ; { }) join code that is next to them, or inside an open brace
    depth = 0
    for seg in segs:
        if seg[0] == "code":
            depth = max(0, depth + _braces(seg[1]))
        elif depth > 0 and seg[1].strip():
            seg[0] = "code"
            depth = max(0, depth + _braces(seg[1]))
    i = 0
    while i < len(segs):                                 # runs of weak text lines
        if segs[i][0] == "text" and _weak(segs[i][1]):
            j = i
            while j < len(segs) and segs[j][0] == "text" and (_weak(segs[j][1]) or not segs[j][1].strip()):
                j += 1
            run = [k for k in range(i, j) if segs[k][1].strip()]
            near = (i > 0 and segs[i - 1][0] == "code") or (j < len(segs) and segs[j][0] == "code")
            if near or len(run) >= 2:
                for k in range(i, j):
                    segs[k][0] = "code"
            i = j
        else:
            i += 1
    for k, seg in enumerate(segs):                       # a blank line between two code lines stays inside
        if seg[0] == "text" and not seg[1].strip() and 0 < k < len(segs) - 1:
            if segs[k - 1][0] == "code" and any(s[0] == "code" for s in segs[k + 1:k + 2]):
                seg[0] = "code"
    blocks: list[tuple[str, list[str]]] = []
    for kind, s in segs:
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(s)
        else:
            blocks.append((kind, [s]))
    out: list[tuple[str, str]] = []
    for kind, lines in blocks:
        body = "\n".join(lines).strip("\n")
        if not body.strip():
            continue
        if kind == "code":
            out.append(("code", beautify_code(body)))
        else:
            out.append(("text", re.sub(r"\n{3,}", "\n\n", body).strip()))
    return out


# ----------------------------------------------------------------------------- code beautifier
def beautify_code(code: str) -> str:
    """One statement per line, indented by brace depth. Spaces inside a line are not changed."""
    code = str(code or "").replace("\t", "    ")
    code = re.sub(r"(#\s*include\s*(?:<[^>\n]*>|\"[^\"\n]*\"))[ \t]*", r"\1\n", code)
    lines: list[str] = []
    buf: list[str] = []
    st = {"depth": 0, "indent": None, "init": 0}
    par, quote, i, n = 0, None, 0, len(code)

    def put(ch: str) -> None:
        if st["indent"] is None:
            if ch in " \t":
                return
            st["indent"] = st["depth"]
        buf.append(ch)

    def flush(level: int | None = None) -> None:
        text = "".join(buf).strip()
        if text:
            lvl = st["indent"] if level is None else level
            lines.append("    " * max(lvl or 0, 0) + text)
        buf.clear()
        st["indent"] = None

    def last() -> str:
        text = "".join(buf).rstrip()
        return text[-1] if text else ""

    while i < n:
        c = code[i]
        if quote:
            if c == "\n":
                quote = None
                flush()
            else:
                put(c)
                if c == "\\" and i + 1 < n:
                    put(code[i + 1])
                    i += 1
                elif c == quote:
                    quote = None
            i += 1
            continue
        if c == "\n":
            if par > 0:
                put(" ")
            else:
                flush()
            i += 1
            continue
        if code.startswith("//", i):
            j = code.find("\n", i)
            j = n if j < 0 else j
            if buf:
                put(" ")
            for ch in code[i:j].rstrip():
                put(ch)
            flush()
            i = j
            continue
        if c in "\"'":
            quote = c
            put(c)
            i += 1
            continue
        if c in "()":
            par = max(0, par + (1 if c == "(" else -1))
            put(c)
            i += 1
            continue
        if par > 0:
            put(c)
            i += 1
            continue
        if c == "{":
            if st["init"] or last() in ("=", ",", "{") or (last() == "]" and "=" in "".join(buf)):
                st["init"] += 1                      # int a[3] = {1, 2, 3};
                put(c)
            else:
                while buf and buf[-1].isspace():
                    buf.pop()
                if buf:
                    put(" ")
                put("{")
                flush()
                st["depth"] += 1
            i += 1
            continue
        if c == "}":
            if st["init"]:
                st["init"] -= 1
                put(c)
                i += 1
                continue
            flush()
            st["depth"] = max(0, st["depth"] - 1)
            put("}")
            if not re.match(r"\s*(?:;|else\b|catch\b|while\b|[A-Za-z_]\w*\s*(?:\[[^\]]*\])?\s*[;,])", code[i + 1:]):
                flush()
            i += 1
            continue
        if c == ";" and not st["init"]:
            put(";")
            if re.match(r"[ \t]*//", code[i + 1:]):
                i += 1
                continue
            flush()
            i += 1
            continue
        if c == ":" and not code.startswith("::", i) and (i == 0 or code[i - 1] != ":"):
            if "".join(buf).strip() in ("public", "private", "protected"):
                put(":")
                flush(max(st["depth"] - 1, 0))
                i += 1
                continue
        put(c)
        i += 1
    flush()
    return "\n".join(lines)


# ----------------------------------------------------------------------------- html
_PART_LINE = re.compile(r"^(?:\(\s*(?:[A-Za-z]|\d{1,2}|" + _ROMAN_TOK + r")\s*\)|(?:[A-Za-z]|\d{1,2}|"
                        + _ROMAN_TOK + r")[.)]|[\u2022-])\s+\S", re.I)


PART_LINE = _PART_LINE


def blocks_html(text: str) -> str:
    """Safe HTML (no line breaks in the markup, so Markdown cannot split it) for a question."""
    out = []
    for kind, body in layout_blocks(restore_parts(text)):
        if kind == "code":
            code = html.escape(body).replace("\n", "&#10;")
            out.append(f'<pre class="codebox"><code>{code}</code></pre>')
            continue
        for line in body.split("\n"):
            if not line.strip():
                continue
            cls = "ql qpart" if _PART_LINE.match(line.strip()) else "ql"
            out.append(f'<div class="{cls}">{html.escape(line.strip())}</div>')
    return "".join(out)
