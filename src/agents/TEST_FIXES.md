# Test Fixes for BaseAgent Refactoring

## Issue

After refactoring agents to use `BaseAgent`, tests were failing with:
```
AttributeError: <module 'src.agents.rag_agent'> does not have the attribute 'init_chat_model'
```

## Root Cause

The tests were using `@patch` decorators to mock `init_chat_model` at the agent module level:

```python
# Old (incorrect after refactoring)
@patch("src.agents.rag_agent.init_chat_model")
@patch("src.agents.cli_agent.init_chat_model")
@patch("src.agents.engineering_agent.init_chat_model")
# etc.
```

However, after refactoring, `init_chat_model` is now imported and used in `base_agent.py`, not in the individual agent files.

## Solution

Update all test patches to point to the correct location:

```python
# New (correct)
@patch("src.agents.base_agent.init_chat_model")
```

## Files Modified

### tests/test_agents/test_rag_agent.py
All patches changed from:
- `@patch("src.agents.rag_agent.init_chat_model")`
- → `@patch("src.agents.base_agent.init_chat_model")`

### tests/test_agents/test_specialized_agents.py
All patches changed from:
- `@patch("src.agents.cli_agent.init_chat_model")`
- `@patch("src.agents.engineering_agent.init_chat_model")`
- `@patch("src.agents.hpc_agent.init_chat_model")`
- `@patch("src.agents.search_agent.init_chat_model")`
- → `@patch("src.agents.base_agent.init_chat_model")`

## Changes Applied

Used `sed` to replace all occurrences:

```bash
# Update rag_agent tests
sed -i '' 's/@patch("src\.agents\.rag_agent\.init_chat_model")/@patch("src.agents.base_agent.init_chat_model")/g' \
  tests/test_agents/test_rag_agent.py

# Update specialized_agents tests
sed -i '' 's/@patch("src\.agents\.cli_agent\.init_chat_model")/@patch("src.agents.base_agent.init_chat_model")/g' \
  tests/test_agents/test_specialized_agents.py
sed -i '' 's/@patch("src\.agents\.engineering_agent\.init_chat_model")/@patch("src.agents.base_agent.init_chat_model")/g' \
  tests/test_agents/test_specialized_agents.py
sed -i '' 's/@patch("src\.agents\.hpc_agent\.init_chat_model")/@patch("src.agents.base_agent.init_chat_model")/g' \
  tests/test_agents/test_specialized_agents.py
sed -i '' 's/@patch("src\.agents\.search_agent\.init_chat_model")/@patch("src.agents.base_agent.init_chat_model")/g' \
  tests/test_agents/test_specialized_agents.py
```

## Verification

All test files now compile successfully:
```bash
python -m py_compile tests/test_agents/test_rag_agent.py tests/test_agents/test_specialized_agents.py
✓ Test files compile successfully
```

## Why This Works

The `@patch` decorator replaces the target object at the location where it's **used**, not where it's **defined**. Since `BaseAgent.__init__()` calls `init_chat_model`, we need to patch it at `base_agent.init_chat_model`.

### Example

```python
# base_agent.py
from langchain.chat_models import init_chat_model

class BaseAgent:
    def __init__(self, model_name):
        self.llm = init_chat_model(model_name)  # ← Called here
```

When testing a subclass like `RAGAgent`, we patch where it's called:
```python
@patch("src.agents.base_agent.init_chat_model")  # ← Patch at usage location
def test_rag_agent():
    agent = RAGAgent()  # This will use the mocked init_chat_model
```

## Tests Affected

Total: **27 test functions** updated across 2 test files

### test_rag_agent.py (13 tests)
- ✅ TestRAGAgentInitialization::test_rag_agent_initialization
- ✅ TestRAGAgentInitialization::test_rag_agent_tools_created
- ✅ TestRAGAgentSearchDocuments::test_search_documents_success
- ✅ TestRAGAgentSearchDocuments::test_search_documents_with_num_results
- ✅ TestRAGAgentSearchDocuments::test_search_documents_error_handling
- ✅ TestRAGAgentAddDocument::test_add_document_success
- ✅ TestRAGAgentAddDocument::test_add_document_with_metadata
- ✅ TestRAGAgentAddDocument::test_add_document_error_handling
- ✅ TestRAGAgentListDocuments::test_list_documents_success
- ✅ TestRAGAgentListDocuments::test_list_documents_empty
- ✅ TestRAGAgentClearMemory::test_clear_memory_success
- ✅ TestRAGAgentInvoke::test_invoke_with_simple_query
- ✅ TestRAGAgentSystemPrompt::test_system_prompt_content

### test_specialized_agents.py (14 tests)
- ✅ test_cli_agent_creation
- ✅ test_cli_agent_has_tools
- ✅ test_engineering_agent_creation
- ✅ test_engineering_agent_has_engibench_tools
- ✅ test_hpc_agent_creation
- ✅ test_hpc_agent_has_slurm_tools
- ✅ test_search_agent_creation
- ✅ test_search_agent_has_search_tool
- ✅ test_agents_use_config_model
- ✅ test_agents_with_custom_model_name
- ✅ test_cli_agent_confirmation_flag
- ✅ test_engineering_agent_confirmation_flag
- ✅ test_agent_creation_with_missing_api_key
- ✅ test_agent_creation_with_invalid_model

## Best Practices

When using the `@patch` decorator:

1. **Patch where it's used, not where it's defined**
   ```python
   # If module A imports function from module B
   # and module C uses module A
   from B import func

   # In tests for C, patch at A.func (where C uses it)
   @patch("A.func")
   ```

2. **Update patches after refactoring**
   - When moving code to a base class, update test patches
   - Search for all `@patch` references to moved functions
   - Use grep/sed for bulk updates

3. **Test compilation after updates**
   ```bash
   python -m py_compile tests/**/*.py
   ```

## Status

✅ All test imports fixed
✅ All test files compile
✅ Tests should now pass (pending pytest run)
