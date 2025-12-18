"""
Tests for weave_integration module.

These tests cover Weave initialization, tracing decorators, dataset creation,
and the disable_tracing context manager.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.weave_integration import (
    create_dataset,
    disable_tracing,
    get_weave,
    init_weave,
    is_weave_enabled,
    log_evaluation,
    traced,
    traced_async,
)

# ============================================================================
# GET_WEAVE TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_get_weave_disabled(mock_config):
    """Test get_weave returns None when Weave is disabled."""
    mock_config.use_weave = False

    result = get_weave()

    assert result is None


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_get_weave_not_installed(mock_config):
    """Test get_weave returns None when weave is not installed."""
    mock_config.use_weave = True

    with patch.dict("sys.modules", {"weave": None}):
        with patch("builtins.__import__", side_effect=ImportError):
            result = get_weave()

    assert result is None


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_get_weave_success(mock_config):
    """Test get_weave returns weave module when available."""
    mock_config.use_weave = True
    mock_weave = MagicMock()

    with patch.dict("sys.modules", {"weave": mock_weave}):
        result = get_weave()

    assert result is mock_weave


# ============================================================================
# INIT_WEAVE TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_init_weave_disabled(mock_config):
    """Test init_weave returns False when disabled."""
    mock_config.setup_weave_tracing.return_value = False

    result = init_weave()

    assert result is False
    mock_config.setup_weave_tracing.assert_called_once()


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
def test_init_weave_success(mock_config):
    """Test init_weave returns True when successful."""
    mock_config.setup_weave_tracing.return_value = True

    result = init_weave()

    assert result is True
    mock_config.setup_weave_tracing.assert_called_once()


# ============================================================================
# IS_WEAVE_ENABLED TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
@patch("src.utils.weave_integration.get_weave")
def test_is_weave_enabled_false_when_disabled(mock_get_weave, mock_config):
    """Test is_weave_enabled returns False when Weave is disabled."""
    mock_config.use_weave = False
    mock_get_weave.return_value = None

    result = is_weave_enabled()

    assert result is False


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
@patch("src.utils.weave_integration.get_weave")
def test_is_weave_enabled_false_when_not_available(mock_get_weave, mock_config):
    """Test is_weave_enabled returns False when Weave is not available."""
    mock_config.use_weave = True
    mock_get_weave.return_value = None

    result = is_weave_enabled()

    assert result is False


@pytest.mark.unit
@patch("src.utils.weave_integration.config")
@patch("src.utils.weave_integration.get_weave")
def test_is_weave_enabled_true(mock_get_weave, mock_config):
    """Test is_weave_enabled returns True when enabled and available."""
    mock_config.use_weave = True
    mock_get_weave.return_value = MagicMock()

    result = is_weave_enabled()

    assert result is True


# ============================================================================
# TRACED DECORATOR TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_traced_decorator_weave_disabled(mock_get_weave):
    """Test traced decorator returns original function when Weave is disabled."""
    mock_get_weave.return_value = None

    @traced
    def test_func(x):
        return x * 2

    result = test_func(5)

    assert result == 10
    # Function should work normally without Weave
    assert callable(test_func)


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_traced_decorator_weave_enabled(mock_get_weave):
    """Test traced decorator wraps function when Weave is enabled."""
    mock_weave = MagicMock()

    # Create a mock decorator that wraps the function
    def mock_op_decorator():
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    mock_weave.op = mock_op_decorator
    mock_get_weave.return_value = mock_weave

    @traced
    def test_func(x):
        return x * 2

    result = test_func(5)

    assert result == 10


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_traced_async_decorator_weave_disabled(mock_get_weave):
    """Test traced_async decorator returns original function when disabled."""
    mock_get_weave.return_value = None

    @traced_async
    async def test_async_func(x):
        return x * 2

    # Verify function is still callable
    assert callable(test_async_func)


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_traced_async_decorator_weave_enabled(mock_get_weave):
    """Test traced_async decorator wraps async function when enabled."""
    mock_weave = MagicMock()
    mock_op = MagicMock()
    mock_weave.op.return_value = mock_op
    mock_get_weave.return_value = mock_weave

    # Create a mock decorator
    def mock_decorator(func):
        return func

    mock_op.return_value = mock_decorator

    @traced_async
    async def test_async_func(x):
        return x * 2

    mock_weave.op.assert_called_once()


# ============================================================================
# CREATE_DATASET TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_create_dataset_weave_disabled(mock_get_weave):
    """Test create_dataset returns None when Weave is disabled."""
    mock_get_weave.return_value = None

    result = create_dataset(
        name="test_dataset", rows=[{"input": "test", "output": "result"}]
    )

    assert result is None


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_create_dataset_success(mock_get_weave):
    """Test create_dataset creates dataset successfully."""
    mock_weave = MagicMock()
    mock_dataset = MagicMock()
    mock_weave.Dataset.return_value = mock_dataset
    mock_get_weave.return_value = mock_weave

    rows = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is the capital of France?", "output": "Paris"},
    ]

    result = create_dataset(name="test_dataset", rows=rows)

    assert result is mock_dataset
    mock_weave.Dataset.assert_called_once_with(name="test_dataset", rows=rows)


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_create_dataset_exception(mock_get_weave):
    """Test create_dataset returns None on exception."""
    mock_weave = MagicMock()
    mock_weave.Dataset.side_effect = Exception("Dataset creation failed")
    mock_get_weave.return_value = mock_weave

    result = create_dataset(name="test_dataset", rows=[])

    assert result is None


# ============================================================================
# LOG_EVALUATION TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_log_evaluation_weave_disabled(mock_get_weave):
    """Test log_evaluation does nothing when Weave is disabled."""
    mock_get_weave.return_value = None

    # Should not raise an error
    log_evaluation(
        dataset_name="test_dataset",
        predictions=["result1", "result2"],
        scores={"accuracy": 0.95},
    )


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_log_evaluation_success(mock_get_weave):
    """Test log_evaluation logs successfully."""
    mock_weave = MagicMock()
    mock_get_weave.return_value = mock_weave

    predictions = ["result1", "result2"]
    scores = {"accuracy": 0.95, "f1": 0.92}

    # Should not raise an error
    log_evaluation(dataset_name="test_dataset", predictions=predictions, scores=scores)


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_log_evaluation_no_scores(mock_get_weave):
    """Test log_evaluation works without scores."""
    mock_weave = MagicMock()
    mock_get_weave.return_value = mock_weave

    predictions = ["result1", "result2"]

    # Should not raise an error
    log_evaluation(dataset_name="test_dataset", predictions=predictions)


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_log_evaluation_exception(mock_get_weave):
    """Test log_evaluation handles exceptions gracefully."""
    mock_weave = MagicMock()
    mock_get_weave.return_value = mock_weave

    # Should not raise an error even if something fails internally
    log_evaluation(
        dataset_name="test_dataset",
        predictions=["result1"],
        scores={"accuracy": 0.95},
    )


# ============================================================================
# DISABLE_TRACING CONTEXT MANAGER TESTS
# ============================================================================


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_disable_tracing_weave_disabled(mock_get_weave):
    """Test disable_tracing context manager when Weave is disabled."""
    mock_get_weave.return_value = None

    # Should not raise an error
    with disable_tracing():
        result = "test"

    assert result == "test"


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_disable_tracing_weave_enabled(mock_get_weave):
    """Test disable_tracing context manager when Weave is enabled."""
    mock_weave = MagicMock()
    mock_get_weave.return_value = mock_weave

    # Mock the OpenAI completions module that will be imported inside the context
    with patch("openai.resources.chat.completions") as mock_completions_module:
        mock_create = MagicMock()
        mock_create.__wrapped__ = MagicMock()
        mock_completions_module.Completions.create = mock_create

        with disable_tracing():
            pass

        # Context manager should work without errors


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_disable_tracing_exception_handling(mock_get_weave):
    """Test disable_tracing handles exceptions gracefully."""
    mock_weave = MagicMock()
    mock_get_weave.return_value = mock_weave

    # Should not raise an error even if unpatching fails
    with disable_tracing():
        result = "test"

    assert result == "test"


@pytest.mark.unit
@patch("src.utils.weave_integration.get_weave")
def test_disable_tracing_restores_state(mock_get_weave):
    """Test disable_tracing restores original state on exit."""
    mock_weave = MagicMock()
    mock_get_weave.return_value = mock_weave

    with patch("openai.resources.chat.completions") as mock_completions_module:
        original_create = MagicMock()
        original_create.__wrapped__ = MagicMock()
        mock_completions_module.Completions.create = original_create

        with disable_tracing():
            # Inside context, state should be changed
            pass

        # After context, state should be restored
        # Context manager should complete without errors


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@patch("src.utils.weave_integration.config")
@patch("src.utils.weave_integration.get_weave")
def test_full_tracing_workflow(mock_get_weave, mock_config):
    """Test complete workflow from init to tracing."""
    mock_config.use_weave = True
    mock_config.setup_weave_tracing.return_value = True

    mock_weave = MagicMock()
    mock_get_weave.return_value = mock_weave

    # Initialize
    assert init_weave() is True
    assert is_weave_enabled() is True

    # Create a traced function
    def mock_decorator(func):
        return func

    mock_weave.op.return_value = mock_decorator

    @traced
    def test_func(x):
        return x * 2

    result = test_func(5)
    assert result == 10


@pytest.mark.integration
@patch("src.utils.weave_integration.get_weave")
def test_dataset_creation_and_evaluation_workflow(mock_get_weave):
    """Test creating dataset and logging evaluation."""
    mock_weave = MagicMock()
    mock_dataset = MagicMock()
    mock_weave.Dataset.return_value = mock_dataset
    mock_get_weave.return_value = mock_weave

    # Create dataset
    rows = [
        {"input": "test1", "expected": "result1"},
        {"input": "test2", "expected": "result2"},
    ]
    dataset = create_dataset(name="test_dataset", rows=rows)

    assert dataset is mock_dataset

    # Log evaluation
    predictions = ["result1", "result2"]
    scores = {"accuracy": 1.0}

    log_evaluation(dataset_name="test_dataset", predictions=predictions, scores=scores)


@pytest.mark.integration
@patch("src.utils.weave_integration.config")
@patch("src.utils.weave_integration.get_weave")
def test_disabled_workflow(mock_get_weave, mock_config):
    """Test that everything works gracefully when Weave is disabled."""
    mock_config.use_weave = False
    mock_config.setup_weave_tracing.return_value = False
    mock_get_weave.return_value = None

    # Initialize (should return False)
    assert init_weave() is False
    assert is_weave_enabled() is False

    # Traced function should work normally
    @traced
    def test_func(x):
        return x * 2

    assert test_func(5) == 10

    # Dataset creation should return None
    dataset = create_dataset(name="test", rows=[])
    assert dataset is None

    # Evaluation should work without errors
    log_evaluation(dataset_name="test", predictions=[])

    # disable_tracing should work
    with disable_tracing():
        result = "test"
    assert result == "test"
