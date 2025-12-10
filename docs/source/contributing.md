# Contributing

Thank you for your interest in contributing to Engineer Assistant! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/engineer-assistant.git
   cd engineer-assistant
   ```
3. **Create a virtual environment**:
   ```bash
   conda env create -f environment.yml
   conda activate engineer-assistant
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,test]"
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `test/` for test improvements

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add docstrings to new functions and classes
- Update documentation as needed

### 3. Write Tests

Add tests for your changes:

```bash
# Create test file
touch tests/test_your_feature.py

# Run tests
make test

# Run specific test
pytest tests/test_your_feature.py
```

### 4. Check Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check
```

### 5. Commit Your Changes

Write clear commit messages:

```bash
git add .
git commit -m "feat: add new ArXiv search feature"
```

Commit message format:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `test:` tests
- `refactor:` code refactoring
- `style:` formatting
- `chore:` maintenance

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots if applicable

## Adding New Engineering Problems

The system uses a **unified problem registry** that makes adding new EngiBench problems simple:

### Step-by-Step Guide

1. **Import the Problem Class** (in `src/tools/problems.py`):
   ```python
   from engibench import YourNewProblem
   ```

2. **Add to PROBLEM_CLASSES Registry**:
   ```python
   PROBLEM_CLASSES: dict[ProblemId, type[Problem]] = {
       "beams2d": Beams2D,
       "thermoelastic2d": ThermoElastic2D,
       "photonics2d": Photonics2D,
       "yournewproblem": YourNewProblem,  # Add here
   }
   ```

3. **Update ProblemId Type**:
   ```python
   ProblemId = Literal["beams2d", "thermoelastic2d", "photonics2d", "yournewproblem"]
   ```

4. **That's It!** The system automatically:
   - ✅ Enables the problem in all tools
   - ✅ Updates documentation in prompts
   - ✅ Extracts objectives dynamically
   - ✅ Handles rendering and constraints
   - ✅ Works with training and sampling

### What NOT to Modify

You **don't need** to edit:
- `src/tools/engibench.py` - Already problem-agnostic
- `src/tools/engiopt.py` - Uses registry automatically
- `src/utils/prompts.py` - Auto-generates documentation
- Agent files - They adapt automatically

### Testing Your New Problem

```python
@pytest.mark.parametrize("problem_id", ["beams2d", "yournewproblem"])
def test_new_problem(problem_id):
    from src.tools.problems import PROBLEM_CLASSES
    problem_class = PROBLEM_CLASSES[problem_id]
    problem = problem_class()
    assert problem is not None
    assert hasattr(problem, 'objectives')
```

## Code Style

### Python

Follow PEP 8 and use type hints:

```python
from typing import List, Optional

def process_data(
    data: List[float],
    threshold: float = 0.5,
    normalize: bool = True
) -> Optional[List[float]]:
    """Process input data with optional normalization.

    Args:
        data: Input data points
        threshold: Minimum value threshold
        normalize: Whether to normalize results

    Returns:
        Processed data or None if invalid
    """
    # Implementation
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided
    """
    pass
```

## Testing

### Writing Tests

Use pytest fixtures and parametrization:

```python
import pytest
from src.agents import EngineeringAgent

@pytest.fixture
def agent():
    return EngineeringAgent()

def test_optimization(agent):
    # Works with any problem type from the registry
    result = agent.optimize(problem="beams2d")
    assert result is not None

@pytest.mark.parametrize("volfrac", [0.3, 0.4, 0.5])
def test_volume_fractions(agent, volfrac):
    result = agent.optimize(config={"volfrac": volfrac})
    assert result["volfrac"] == volfrac

@pytest.mark.parametrize("problem_type", ["beams2d", "thermoelastic2d", "photonics2d"])
def test_all_problem_types(agent, problem_type):
    # System automatically adapts to any registered problem
    result = agent.optimize(problem=problem_type)
    assert result is not None
```

### Running Tests

```bash
# All tests
make test

# Specific file
pytest tests/test_agents/test_arxiv_agent.py

# With coverage
make test-coverage

# Verbose output
pytest -v
```

## Documentation

### Building Docs

```bash
# Build documentation
make docs

# Live preview
make docs-watch
```

### Writing Documentation

- Use Markdown for content pages
- Add examples and code snippets
- Include links to API reference
- Update the index when adding new pages

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code is formatted (`make format`)
- [ ] No linting errors (`make lint`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)

### PR Checklist

Your PR should:
- Have a clear title and description
- Link to related issues (e.g., "Fixes #123")
- Include tests for new features
- Update documentation
- Follow the code style guide
- Have passing CI checks

## Code Review Process

1. **Maintainers review** your PR
2. **Address feedback** by pushing new commits
3. **Once approved**, your PR will be merged
4. **Your contribution** will be credited in releases

## Questions?

- Open an [issue](https://github.com/gioelemo/engineer-assistant/issues) for bugs
- Start a [discussion](https://github.com/gioelemo/engineer-assistant/discussions) for questions
- Join our community chat (if available)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🎉
