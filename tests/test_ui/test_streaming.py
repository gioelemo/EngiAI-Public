"""Tests for streaming text functionality."""

from src.ui.streaming import stream_text


class TestStreamText:
    """Tests for stream_text generator function."""

    def test_stream_text_basic(self):
        """Test basic text streaming."""
        text = "Hello world test"
        chunks = list(stream_text(text, chunk_size=1))

        # Should yield word by word
        assert len(chunks) > 0
        # Reconstruct text
        reconstructed = "".join(chunks)
        assert "Hello" in reconstructed
        assert "world" in reconstructed
        assert "test" in reconstructed

    def test_stream_text_with_chunk_size(self):
        """Test streaming with different chunk sizes."""
        text = "One two three four five"

        # Chunk size 2
        chunks = list(stream_text(text, chunk_size=2))
        assert len(chunks) > 0

        # Reconstruct should match original + newline
        reconstructed = "".join(chunks).strip()
        assert reconstructed == text

    def test_stream_text_with_markdown_heading(self):
        """Test streaming with markdown headings."""
        text = "# Main Title\nSome text"
        chunks = list(stream_text(text, chunk_size=1))

        # Heading should be yielded whole
        assert any("# Main Title" in chunk for chunk in chunks)

        # Reconstruct should preserve formatting
        reconstructed = "".join(chunks)
        assert "# Main Title" in reconstructed

    def test_stream_text_with_bullet_list(self):
        """Test streaming with bullet lists."""
        text = "- Item one\n- Item two\nRegular text"
        chunks = list(stream_text(text, chunk_size=1))

        # List items should be yielded whole
        assert any("- Item one" in chunk for chunk in chunks)
        assert any("- Item two" in chunk for chunk in chunks)

    def test_stream_text_with_numbered_list(self):
        """Test streaming with numbered lists."""
        text = "1. First item\n2. Second item\nRegular text"
        chunks = list(stream_text(text, chunk_size=1))

        # Numbered items should be yielded whole
        assert any("1. First item" in chunk for chunk in chunks)
        assert any("2. Second item" in chunk for chunk in chunks)

    def test_stream_text_with_code_block(self):
        """Test streaming with code blocks."""
        text = "Some text\n```python\ndef test():\n    pass\n```\nMore text"
        chunks = list(stream_text(text, chunk_size=1))

        # Code block markers should be preserved
        assert any("```python" in chunk for chunk in chunks)
        assert any("def test():" in chunk for chunk in chunks)

        # Reconstruct should preserve code block
        reconstructed = "".join(chunks)
        assert "```python" in reconstructed
        assert "def test():" in reconstructed

    def test_stream_text_with_blockquote(self):
        """Test streaming with blockquotes."""
        text = "> Quote text\nRegular text"
        chunks = list(stream_text(text, chunk_size=1))

        # Blockquote should be yielded whole
        assert any("> Quote text" in chunk for chunk in chunks)

    def test_stream_text_with_empty_lines(self):
        """Test streaming with empty lines."""
        text = "Line one\n\nLine two"
        chunks = list(stream_text(text, chunk_size=1))

        # Should handle empty lines
        reconstructed = "".join(chunks)
        assert "Line one" in reconstructed
        assert "Line two" in reconstructed

    def test_stream_text_with_asterisk_list(self):
        """Test streaming with asterisk-style lists."""
        text = "* Item A\n* Item B"
        chunks = list(stream_text(text, chunk_size=1))

        # Asterisk list items should be yielded whole
        assert any("* Item A" in chunk for chunk in chunks)
        assert any("* Item B" in chunk for chunk in chunks)

    def test_stream_text_with_plus_list(self):
        """Test streaming with plus-style lists."""
        text = "+ Item A\n+ Item B"
        chunks = list(stream_text(text, chunk_size=1))

        # Plus list items should be yielded whole
        assert any("+ Item A" in chunk for chunk in chunks)
        assert any("+ Item B" in chunk for chunk in chunks)

    def test_stream_text_with_table(self):
        """Test streaming with markdown tables."""
        text = "| Col1 | Col2 |\n| ---- | ---- |"
        chunks = list(stream_text(text, chunk_size=1))

        # Table rows should be yielded whole
        assert any("| Col1 | Col2 |" in chunk for chunk in chunks)
        assert any("| ---- | ---- |" in chunk for chunk in chunks)

    def test_stream_text_empty_string(self):
        """Test streaming empty string."""
        text = ""
        chunks = list(stream_text(text, chunk_size=1))

        # Empty text produces a newline for empty lines
        assert len(chunks) <= 1

    def test_stream_text_single_word(self):
        """Test streaming single word."""
        text = "Hello"
        chunks = list(stream_text(text, chunk_size=1))

        # Should yield the word
        assert len(chunks) > 0
        reconstructed = "".join(chunks).strip()
        assert "Hello" in reconstructed

    def test_stream_text_multiline_code_block(self):
        """Test streaming with multi-line code blocks."""
        text = """Regular text
```python
def hello():
    print("world")
    return True
```
More regular text"""

        chunks = list(stream_text(text, chunk_size=1))

        # All code lines should be present
        reconstructed = "".join(chunks)
        assert "def hello():" in reconstructed
        assert 'print("world")' in reconstructed
        assert "return True" in reconstructed

    def test_stream_text_mixed_markdown(self):
        """Test streaming with mixed markdown elements."""
        text = """# Title
Some text here

- Bullet one
- Bullet two

```python
code()
```

More text"""

        chunks = list(stream_text(text, chunk_size=1))

        # Should preserve all elements
        reconstructed = "".join(chunks)
        assert "# Title" in reconstructed
        assert "- Bullet one" in reconstructed
        assert "```python" in reconstructed
        assert "code()" in reconstructed

    def test_stream_text_generator_behavior(self):
        """Test that stream_text is a proper generator."""
        text = "Hello world"
        gen = stream_text(text)

        # Should be a generator
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")

        # Should be able to iterate
        first_chunk = next(gen)
        assert isinstance(first_chunk, str)
