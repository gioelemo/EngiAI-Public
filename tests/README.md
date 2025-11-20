# Test Suite for Engineer Assistant

Simple, focused tests for the LLM application using LangChain's `GenericFakeChatModel`.

## Quick Start

```bash
# Run all tests
pytest

# Run only fast tests
pytest -m "not slow"

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py              # Fixtures using GenericFakeChatModel
├── test_example.py          # Example tests (keep for reference)
├── test_agents/
│   └── test_supervisor.py   # Supervisor routing tests (5 tests)
└── test_tools/
    └── test_engibench.py    # Tool tests (2 tests)
```

## Writing New Tests

### 1. Create mock LLMs for testing

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

@pytest.mark.unit
def test_something():
    """Test using GenericFakeChatModel."""
    fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="test response")]))

    # Patch where init_chat_model is IMPORTED, not where it's defined
    with patch("src.agents.your_agent.init_chat_model", return_value=fake_llm):
        # Your test here
        pass
```

**Important:** Patch `init_chat_model` where it's imported in your module!

### 2. Mark slow tests

```python
@pytest.mark.slow
def test_optimization():
    """Slow test - skipped in CI."""
    pass
```

## Key Features

✅ Uses LangChain's `GenericFakeChatModel` (no API calls)
✅ Automatic environment variable mocking
✅ Fast unit tests (milliseconds)
✅ CI-ready (no API keys needed)

## More Information

See [TESTING.md](../TESTING.md) for:
- Detailed testing guide
- Best practices
- Examples
- Troubleshooting
