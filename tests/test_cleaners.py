"""Markup cleaners for each documentation site's templating."""

from rag.config import UrlRule
from rag.ingestion.cleaners import clean_markup, outside_fences, split_front_matter


def test_mdn_macros_render_as_text() -> None:
    text = (
        '{{JSRef}}\nUse {{jsxref("Promise/then", "then()")}} on a {{jsxref("Promise")}}.\n'
        'A {{Glossary("Closure")}} captures scope.\n{{EmbedInteractiveExample("pages/js/x.html")}}\n'
        "> [!NOTE]\n> Careful.\n- `x`\n  - : the value\n{{Compat}}"
    )
    out = clean_markup(text)
    assert "Use then() on a Promise." in out
    assert "A Closure captures scope." in out
    assert "> **Note:**" in out and "  the value" in out
    assert "{{" not in out


def test_mdx_components_imports_and_anchors() -> None:
    text = (
        "import Tabs from '@theme/Tabs';\nimport TabItem from '@theme/TabItem';\n"
        "## Choosing state {/*choosing-state*/}\n\n"
        "<Pitfall>\nDon't mutate state.\n</Pitfall>\n\n"
        "<Tabs groupId=\"lang\" values={[{label: 'iOS', value: 'ios'}]}>\ntext\n</Tabs>"
    )
    out = clean_markup(text)
    assert "import" not in out
    assert "## Choosing state\n" in out
    assert "**Pitfall:** \nDon't mutate state." in out
    assert "<Tabs" not in out and "</Tabs>" not in out


def test_code_fences_are_never_modified() -> None:
    text = "Prose {{jsxref(\"Array\")}}\n\n```jsx\n<div style={{color: 'red'}}>{% raw %}</div>\nimport x from 'y';\n```"
    out = clean_markup(text)
    assert out.startswith("Prose Array")
    assert "<div style={{color: 'red'}}>{% raw %}</div>\nimport x from 'y';" in out


def test_nunjucks_and_go_templates() -> None:
    text = (
        '{% Aside %}\nCache first.\n{% endAside %}\n{% Img src="image/x.png", alt="x" %}\n'
        "{% Compare 'better' %}\nok\n{% endCompare %}\n"
        '## Origins {#Origins}\n{{code "/doc/progs/x.go" `/^func/`}}\n{{raw `\n\tpackage io\n`}}'
    )
    out = clean_markup(text)
    assert "**Note:** \nCache first." in out
    assert "**Do:** \nok" in out
    assert "## Origins\n" in out
    assert "{{" not in out and "{%" not in out
    assert "```\n\tpackage io\n```" in out


def test_outside_fences_handles_unclosed_fence() -> None:
    assert outside_fences("a\n```\nb", str.upper) == "A\n```\nb"


def test_go_json_front_matter() -> None:
    meta, body = split_front_matter(
        '<!--{\n\t"Title": "Effective Go",\n\t"Template": true\n}-->\n<p>Hi</p>'
    )
    assert meta["title"] == "Effective Go"
    assert body.strip() == "<p>Hi</p>"


def test_url_from_front_matter_key() -> None:
    rule = UrlRule(
        base="https://developer.mozilla.org/en-US/docs/", front_matter_key="slug"
    )
    url = rule.build("files/en-us/web/x/index.md", {"slug": "Web/JavaScript/Closures"})
    assert url == "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Closures"
    # Falls back to the path rule when the field is missing.
    assert rule.build("x.md", {}) == "https://developer.mozilla.org/en-US/docs/x.md"


def test_inline_code_and_prose_placeholders_are_preserved() -> None:
    text = (
        "Add `#include <Python.h>` and pass `--range=<IPv4 CIDR>`.\n"
        'Press <Enter> to continue; a `{{jsxref("x")}}` literal stays.\n'
        "Call <CodeStep step={1}>useState</CodeStep> first."
    )
    out = clean_markup(text)
    assert "`#include <Python.h>`" in out and "`--range=<IPv4 CIDR>`" in out
    assert "Press <Enter> to continue" in out
    assert '`{{jsxref("x")}}`' in out
    assert "Call useState first." in out
