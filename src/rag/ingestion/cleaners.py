"""Markup cleaners: render or strip the templating each documentation site uses.

Documentation sources are "Markdown" in name only. Each site layers its own
templating on top, which would otherwise end up in the index as noise:

==========================  =================================================
Source                      Markup handled here
==========================  =================================================
Kubernetes (Hugo)           ``{{< glossary_tooltip text="Pod" >}}``, callouts
MDN (KumaScript)            ``{{jsxref("Promise")}}``, ``{{Compat}}``
React / React Native (MDX)  ``import ...``, ``<Pitfall>``, ``{/*anchor*/}``
web.dev (Nunjucks)          ``{% Aside %}``, ``{% Img ... %}``
Go website                  ``<!--{ JSON front matter }-->``, ``{{code ...}}``
==========================  =================================================

Every cleaner runs only *outside* fenced code blocks and inline code spans, so
code (JSX ``style={{...}}``, ``#include <Python.h>``, Go templates) is never
altered.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import yaml

_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def outside_fences(markdown: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to every run of text that is not inside a code fence."""
    out: list[str] = []
    prose: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            if not in_fence and prose:
                out.append(transform("\n".join(prose)))
                prose = []
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            prose.append(line)
    if prose:
        out.append(transform("\n".join(prose)))
    return "\n".join(out)


# Inline code spans: `x` or ``x `y` z``.
_INLINE_CODE_RE = re.compile(r"(``[^\n]*?``|`[^`\n]+`)")


def outside_inline_code(text: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to text between inline code spans only."""
    parts = _INLINE_CODE_RE.split(text)
    return "".join(p if i % 2 else transform(p) for i, p in enumerate(parts))


# --- Front matter -------------------------------------------------------------
_YAML_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_JSON_FRONT_MATTER_RE = re.compile(r"\A\s*<!--\s*(\{.*?\})\s*-->", re.DOTALL)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(metadata, body)``. Keys are lower-cased.

    Supports YAML front matter (``---``) and the Go website's JSON-in-comment
    header (``<!--{ "Title": ... }-->``).
    """
    meta: Any = {}
    body = text
    if match := _YAML_FRONT_MATTER_RE.match(text):
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        body = text[match.end() :]
    elif match := _JSON_FRONT_MATTER_RE.match(text):
        try:
            meta = json.loads(match.group(1))
        except json.JSONDecodeError:
            meta = {}
        body = text[match.end() :]
    if not isinstance(meta, dict):
        return {}, body
    return {str(k).lower(): v for k, v in meta.items()}, body


# --- Hugo shortcodes (Kubernetes) ---------------------------------------------
_SHORTCODE_RE = re.compile(r"\{\{[<%]\s*(/?)\s*([\w-]+)(.*?)\s*[>%]\}\}", re.DOTALL)
_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')
_STRING_ARG_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
_HUGO_HEADINGS = {
    "whatsnext": "What's next",
    "prerequisites": "Before you begin",
    "objectives": "Objectives",
    "cleanup": "Cleaning up",
    "synopsis": "Synopsis",
    "options": "Options",
    "seealso": "See also",
}
_CALLOUTS = {"note": "Note", "caution": "Caution", "warning": "Warning"}


def _string_args(args: str) -> list[str]:
    return [a or b for a, b in _STRING_ARG_RE.findall(args)]


def _render_shortcode(match: re.Match[str]) -> str:
    closing, name, args = match.group(1), match.group(2), match.group(3)
    attrs = dict(_ATTR_RE.findall(args))
    if closing:
        return ""
    if name == "glossary_tooltip":
        return attrs.get("text") or attrs.get("term_id", "")
    if name == "heading":
        key = next(iter(_string_args(args)), "")
        return _HUGO_HEADINGS.get(key, key.capitalize())
    if name in _CALLOUTS:
        return f"**{_CALLOUTS[name]}:** "
    if name == "feature-state":
        version, state = attrs.get("for_k8s_version", ""), attrs.get("state", "")
        return f"*Feature state: Kubernetes {version} [{state}]*" if state else ""
    if name == "mermaid":
        return "\n```mermaid\n"  # the closing tag is handled in clean_hugo
    if name in {"code_sample", "codenew"}:
        file = attrs.get("file", "")
        return f"(Example manifest: `{file}`)" if file else ""
    if name == "tab":
        return f"\n**{attrs['name']}**\n" if attrs.get("name") else ""
    return ""  # tabs, table, figure, skew, param, thirdparty-content, ...


def clean_hugo(markdown: str) -> str:
    """Render Hugo shortcodes as plain Markdown (Mermaid diagrams are kept)."""
    markdown = re.sub(r"\{\{[<%]\s*/\s*mermaid\s*[>%]\}\}", "\n```\n", markdown)
    markdown = re.sub(
        r"\{\{[<%]\s*comment\s*[>%]\}\}.*?\{\{[<%]\s*/\s*comment\s*[>%]\}\}",
        "",
        markdown,
        flags=re.DOTALL,
    )
    return _SHORTCODE_RE.sub(_render_shortcode, markdown)


# --- KumaScript macros (MDN) ----------------------------------------------------
_MACRO_RE = re.compile(r"\{\{\s*([A-Za-z][\w-]*)\s*(?:\((.*?)\))?\s*\}\}")
# Macros that render widgets, sidebars or banners rather than inline text.
_DROP_MACROS = {
    "compat",
    "specifications",
    "embedinteractiveexample",
    "embedlivesample",
    "embedghlivesample",
    "previous",
    "next",
    "previousnext",
    "previousmenunext",
    "listsubpages",
    "listsubpagesgrouped",
    "quicklinkswithsubpages",
}
_MDN_ALERT_RE = re.compile(
    r"^>\s*\[!(NOTE|WARNING|CALLOUT|TIP|CAUTION)\]\s*$", re.MULTILINE
)
_MDN_DEFINITION_RE = re.compile(r"^(\s*)- : ", re.MULTILINE)


def _render_macro(match: re.Match[str]) -> str:
    name, args = match.group(1).lower(), match.group(2)
    if args is None or name in _DROP_MACROS:
        return ""  # {{jsSidebar}}, {{Deprecated_Inline}}, {{Compat}}, ...
    strings = _string_args(args)
    if not strings:
        return ""
    if len(strings) >= 2 and strings[1]:
        return strings[1]  # {{jsxref("Promise/then", "then()")}} -> then()
    return (
        strings[0].rstrip("/").split("/")[-1]
    )  # {{jsxref("Global_Objects/Array")}} -> Array


def clean_mdn(markdown: str) -> str:
    markdown = _MACRO_RE.sub(_render_macro, markdown)
    markdown = _MDN_ALERT_RE.sub(
        lambda m: f"> **{m.group(1).capitalize()}:**", markdown
    )
    return _MDN_DEFINITION_RE.sub(r"\1", markdown)


# --- MDX (React, React Native) ---------------------------------------------------
# MDX module lines: `import X from './x.md';`, `import './x.css';`, `export const ...`.
_MDX_IMPORT_RE = re.compile(
    r"^(?:import\s[^\n]*?\bfrom\s+['\"][^\n]*|import\s+['\"][^\n]*|export\s+(?:const|default|function)\b[^\n]*)$",
    re.MULTILINE,
)
_MDX_COMMENT_RE = re.compile(r"[ \t]*\{/\*.*?\*/\}", re.DOTALL)
_HEADING_ANCHOR_RE = re.compile(r"^(#{1,6}\s.*?)\s*\{#[^}]*\}\s*$", re.MULTILINE)
# A capitalised JSX tag; props may hold expressions nested two levels deep,
# e.g. <Tabs values={[{label: 'iOS'}]}>.
_JSX_TAG_RE = re.compile(
    r"</?([A-Z][A-Za-z0-9.]*)\b(?:[^<>{}]|\{(?:[^{}]|\{[^{}]*\})*\})*?/?>"
)
# Components used inline in react.dev prose; any other capitalised tag is only
# treated as JSX when it starts a line, so text such as "<Enter>" or "<T>" in
# prose is left alone.
_INLINE_JSX = {
    "CodeStep",
    "Trans",
    "Math",
    "MathI",
    "Hint",
    "Diagram",
    "Illustration",
    "ExperimentalBadge",
    "CanaryBadge",
    "NextMajorBadge",
    "RSCBadge",
    "BoxLink",
    "ColorDefaults",
    "TableRow",
}
_JSX_LABELS = {
    "Pitfall": "**Pitfall:** ",
    "Note": "**Note:** ",
    "DeepDive": "**Deep dive:** ",
    "Recap": "**Recap:** ",
    "YouWillLearn": "**You will learn:** ",
    "Wip": "**Note:** ",
    "Canary": "**Canary:** ",
}


def _render_jsx(match: re.Match[str]) -> str:
    line_start = match.string.rfind("\n", 0, match.start()) + 1
    starts_line = not match.string[line_start : match.start()].strip()
    if not starts_line and match.group(1) not in _INLINE_JSX:
        return match.group(0)
    if match.group(0).startswith("</"):
        return ""
    return _JSX_LABELS.get(match.group(1), "")


def clean_mdx_lines(markdown: str) -> str:
    """Line-level MDX syntax: module imports, comments and heading ids."""
    markdown = _MDX_IMPORT_RE.sub("", markdown)
    markdown = _MDX_COMMENT_RE.sub("", markdown)  # react.dev heading ids: {/*title*/}
    return _HEADING_ANCHOR_RE.sub(r"\1", markdown)  # Go/Hugo heading ids: {#title}


def clean_jsx(markdown: str) -> str:
    """JSX components: block tags removed (callouts kept as labels)."""
    return _JSX_TAG_RE.sub(_render_jsx, markdown)


# --- Nunjucks (web.dev) -----------------------------------------------------------
_NUNJUCKS_RE = re.compile(r"\{%-?\s*(\w+)(.*?)-?%\}", re.DOTALL)
_NUNJUCKS_LABELS = {"Aside": "**Note:** ", "Banner": "**Note:** "}


def _render_nunjucks(match: re.Match[str]) -> str:
    name, args = match.group(1), match.group(2)
    if name.startswith("end"):
        return ""
    if name == "Compare":
        kind = next(iter(_string_args(args)), "")
        return {"better": "**Do:** ", "worse": "**Don't:** "}.get(kind, "")
    return _NUNJUCKS_LABELS.get(name, "")  # Img, Video, BrowserCompat, Codepen, ...


# --- Go website templates -----------------------------------------------------------
# {{raw `...`}} wraps a literal (usually code) block in the Go website sources.
_GO_RAW_RE = re.compile(r"\{\{-?\s*raw\s+`(.*?)`\s*-?\}\}", re.DOTALL)
_GO_TEMPLATE_RE = re.compile(
    r"\{\{-?\s*(?:code|raw|template|if|else|end|range|with|breadcrumbs|\.)[^{}\n]*-?\}\}"
)


def clean_markup(markdown: str) -> str:
    """Apply every cleaner, outside code fences. Safe for all supported sources:
    each pattern only matches its own site's syntax."""
    markdown = clean_hugo(markdown)  # first: it emits mermaid fences of its own
    markdown = _GO_RAW_RE.sub(
        lambda m: f"\n```\n{m.group(1).strip(chr(10))}\n```\n", markdown
    )

    def prose(text: str) -> str:
        text = clean_mdn(text)
        text = clean_jsx(text)
        return _NUNJUCKS_RE.sub(_render_nunjucks, text)

    def block(text: str) -> str:
        # Line-level syntax first: Go templates may contain backticks
        # ({{code "x.go" `/^func/`}}) and headings may contain inline code, so
        # both are handled before the text is split around inline code.
        text = _GO_TEMPLATE_RE.sub("", text)
        text = clean_mdx_lines(text)
        # Neither inline code (`<Python.h>`, `<IPv4 CIDR>`) nor fences are touched.
        return outside_inline_code(text, prose)

    return outside_fences(markdown, block)
