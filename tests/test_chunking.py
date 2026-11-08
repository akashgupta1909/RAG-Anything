from rag.chunking import MarkdownChunker
from rag.models import Document, ImageRef


def _doc(text: str, images: list[ImageRef] | None = None) -> Document:
    return Document(
        doc_id="python/docs/a.md",
        domain="python",
        source="docs",
        title="Guide",
        url="https://example.com/a",
        text=text,
        images=images or [],
    )


def test_chunks_follow_headings_and_keep_heading_path() -> None:
    text = (
        "# Guide\n\nIntro paragraph. " * 3
        + "\n\n## Install\n\n"
        + "Run pip install. " * 30
    )
    chunks = MarkdownChunker(max_chars=400, min_chars=50).chunk(_doc(text))
    assert chunks[0].heading_path == []  # the title H1 is dropped from the path
    assert all(c.heading_path == ["Install"] for c in chunks[1:])
    assert all(len(c.text) <= 400 for c in chunks)
    assert chunks[1].embedding_text.startswith("Guide > Install")


def test_heading_path_handles_missing_h1_and_sibling_sections() -> None:
    # Kubernetes pages start at H2: siblings must replace each other in the path.
    text = "## What is a Pod?\n\n" + "a " * 200 + "\n\n## Using Pods\n\n" + "b " * 200
    pairs = MarkdownChunker(max_chars=500, min_chars=10).split_text(text)
    assert [path for path, _ in pairs] == [["What is a Pod?"], ["Using Pods"]]


def test_code_blocks_are_not_split_when_they_fit() -> None:
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(20)) + "\n```"
    text = "## Example\n\n" + "Text before. " * 20 + "\n\n" + code
    chunks = MarkdownChunker(max_chars=400, min_chars=10).split_text(text)
    code_chunks = [t for _, t in chunks if "```python" in t]
    assert len(code_chunks) == 1
    assert code_chunks[0].count("```") == 2


def test_oversized_code_block_is_split_into_valid_fences() -> None:
    code = (
        "```python\n"
        + "\n".join(f"value_{i} = {i} * 1000" for i in range(200))
        + "\n```"
    )
    pieces = MarkdownChunker(max_chars=500, min_chars=10).split_text(code)
    assert len(pieces) > 1
    for _, piece in pieces:
        assert len(piece) <= 500
        assert piece.startswith("```python") and piece.endswith("```")


def test_small_sections_are_merged() -> None:
    text = "## A\n\nshort.\n\n## B\n\nalso short.\n\n## C\n\n" + "long text. " * 60
    pairs = MarkdownChunker(max_chars=1000, min_chars=100).split_text(text)
    assert "## A" in pairs[0][1] and "## B" in pairs[0][1]


def test_image_tokens_become_captions_and_ids() -> None:
    image = ImageRef(
        image_id="0123456789abcdef", local_path="/x.png", url="u", caption="Cache layer"
    )
    chunks = MarkdownChunker().chunk(
        _doc("## Caching\n\n[[image:0123456789abcdef]]\n\nUse a cache." * 2, [image])
    )
    assert chunks[0].image_ids == ["0123456789abcdef"]
    assert "[Diagram: Cache layer]" in chunks[0].text
    assert "[[image:" not in chunks[0].text


def test_chunk_ids_are_stable() -> None:
    doc = _doc("## A\n\n" + "words " * 400)
    first = [c.chunk_id for c in MarkdownChunker().chunk(doc)]
    assert first == [c.chunk_id for c in MarkdownChunker().chunk(doc)]
    assert len(set(first)) == len(first)
