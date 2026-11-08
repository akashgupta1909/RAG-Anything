from pathlib import Path

from rag.config import SourceConfig, UrlRule
from rag.ingestion.cleaners import clean_hugo, split_front_matter
from rag.ingestion.loaders import DocumentLoader, html_to_markdown
from rag.ingestion.sources import _matches


def test_hugo_shortcodes_are_rendered_as_markdown() -> None:
    text = (
        'A {{< glossary_tooltip text="Pod" term_id="pod" >}} runs.\n'
        "{{< note >}}\nBe careful.\n{{< /note >}}\n"
        '{{< feature-state for_k8s_version="v1.30" state="stable" >}}\n'
        '## {{% heading "whatsnext" %}}\n'
        "{{< mermaid >}}\ngraph LR; A-->B\n{{< /mermaid >}}\n"
    )
    out = clean_hugo(text)
    assert "A Pod runs." in out
    assert "**Note:** \nBe careful." in out
    assert "Kubernetes v1.30 [stable]" in out
    assert "## What's next" in out
    assert "```mermaid\n\ngraph LR; A-->B\n\n```" in out
    assert "{{" not in out


def test_front_matter_is_parsed() -> None:
    meta, body = split_front_matter("---\ntitle: Pods\nweight: 10\n---\nBody text")
    assert meta["title"] == "Pods"
    assert body == "Body text"


def test_sphinx_html_main_content_to_markdown() -> None:
    html = """<html><body><div class="related">nav</div>
    <div class="body" role="main"><h1>asyncio<a class="headerlink" href="#x">¶</a></h1>
    <p>Run <code>asyncio.run()</code>.</p>
    <div class="highlight-python3"><div class="highlight"><pre>import asyncio
asyncio.run(main())</pre></div></div></div></body></html>"""
    title, md = html_to_markdown(html)
    assert title == "asyncio"
    assert "¶" not in md and "nav" not in md
    assert "```python\nimport asyncio\nasyncio.run(main())\n```" in md


def test_url_rule() -> None:
    rule = UrlRule(
        base="https://kubernetes.io/docs/",
        strip_prefix="content/en/docs/",
        replace={"_index.md": "", ".md": "/"},
    )
    assert (
        rule.build("content/en/docs/concepts/workloads/pods/_index.md")
        == "https://kubernetes.io/docs/concepts/workloads/pods/"
    )
    assert (
        rule.build("content/en/docs/concepts/overview/components.md")
        == "https://kubernetes.io/docs/concepts/overview/components/"
    )


def test_glob_matching() -> None:
    assert _matches(
        "content/en/docs/concepts/a/b.md", ["content/en/docs/concepts/**/*.md"]
    )
    assert _matches(
        "content/en/docs/concepts/b.md", ["content/en/docs/concepts/**/*.md"]
    )
    assert not _matches("README-ja.md", ["README.md"])


def test_loader_extracts_local_images(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "lb.png").write_bytes(b"\x89PNG fake bytes")
    (tmp_path / "doc.md").write_text(
        "# Design\n\n## Load balancer\n\n![Imgur](http://i.imgur.com/lb.png)\n\n"
        '<p align="center"><img src="images/lb.png"></p>\n\nText about [balancing](#x) traffic.\n',
        encoding="utf-8",
    )
    source = SourceConfig(
        name="sdp",
        kind="local",
        location=str(tmp_path),
        format="markdown",
        images=True,
        asset_base="https://raw.example.com/",
    )
    doc = DocumentLoader("system_design", source, tmp_path).load(
        tmp_path / "doc.md", "doc.md"
    )
    assert doc is not None
    assert len(doc.images) == 1  # same file referenced twice
    image = doc.images[0]
    assert image.caption == "Load balancer"  # "Imgur" alt text is ignored
    assert image.url == "https://raw.example.com/images/lb.png"
    assert doc.text.count(f"[[image:{image.image_id}]]") == 2
    assert "<p" not in doc.text
    assert "Text about balancing traffic." in doc.text  # link target removed
