# Development Guide

Guidelines for contributing to BOTCLAVE.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Style](#code-style)
3. [Testing](#testing)
4. [Contributing](#contributing)
5. [Pull Request Process](#pull-request-process)
6. [Release Process](#release-process)

## Development Setup

### Prerequisites

- Python 3.9+
- Poetry or pip
- Git
- Code editor (VSCode, PyCharm recommended)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/botclave.git
cd botclave

# Install development dependencies
poetry install --with dev
# or
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
black = "^23.0.0"
ruff = "^0.1.0"
mypy = "^1.5.0"
pre-commit = "^3.4.0"
```

## Code Style

### Formatting with Black

BOTCLAVE uses Black for code formatting with 88-character line length.

```bash
# Format all code
black src/ tests/ scripts/

# Check formatting without changes
black --check src/ tests/

# Format specific file
black src/botclave/engine/strategy.py
```

### Linting with Ruff

Ruff is used for fast Python linting.

```bash
# Lint all code
ruff check src/ tests/ scripts/

# Auto-fix issues
ruff check src/ tests/ --fix

# Lint specific file
ruff check src/botclave/engine/strategy.py
```

### Type Checking with MyPy

All code should include type hints.

```bash
# Type check entire project
mypy src/

# Type check specific module
mypy src/botclave/engine/
```

### Code Style Guidelines

1. **Line Length**: Maximum 88 characters
2. **Imports**: 
   - Standard library first
   - Third-party second
   - Local imports last
   - Alphabetically sorted within groups

```python
# Standard library
from datetime import datetime
from typing import Dict, List, Optional

# Third-party
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Local
from botclave.engine.depth import DepthAnalyzer
from botclave.engine.footprint import FootprintChart
```

3. **Naming Conventions**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private members: `_leading_underscore`

4. **Docstrings**: Use Google-style docstrings

```python
def calculate_signal(
    self,
    price: float,
    volume: float,
    timestamp: int,
) -> Optional[Signal]:
    """
    Calculate trading signal from market data.

    Args:
        price: Current market price
        volume: Trading volume
        timestamp: Data timestamp

    Returns:
        Signal object if conditions met, None otherwise

    Raises:
        ValueError: If price or volume is negative
    """
    pass
```

5. **Type Hints**: Always include type hints

```python
from typing import Dict, List, Optional, Tuple

def process_data(
    df: pd.DataFrame,
    window: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """Process DataFrame and return results."""
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_strategy.py -v

# Run specific test
pytest tests/test_strategy.py::TestOrderFlowStrategy::test_initialization -v

# Run with coverage
pytest tests/ --cov=botclave --cov-report=html --cov-report=term

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v
```

### Writing Tests

1. **Test Structure**: Follow AAA pattern (Arrange, Act, Assert)

```python
def test_calculate_delta(self):
    """Test delta calculation."""
    # Arrange
    chart = FootprintChart()
    bar = FootprintBar(
        timestamp=1000,
        buy_volume=60.0,
        sell_volume=40.0,
    )
    
    # Act
    delta = chart.calculate_delta(bar)
    
    # Assert
    assert delta == 20.0
```

2. **Test Coverage**: Aim for >80% coverage

```bash
# Generate coverage report
pytest tests/ --cov=botclave --cov-report=html

# Open report
open htmlcov/index.html
```

3. **Test Naming**: Use descriptive names

```python
def test_strategy_generates_long_signal_when_cvd_positive():
    """Test that strategy generates long signal with positive CVD."""
    pass
```

4. **Fixtures**: Use pytest fixtures for common setup

```python
import pytest
import pandas as pd

@pytest.fixture
def sample_ohlcv_data():
    """Fixture providing sample OHLCV data."""
    return pd.DataFrame({
        'open': [50000, 50100, 50050],
        'high': [50100, 50200, 50150],
        'low': [49900, 50000, 49950],
        'close': [50050, 50150, 50100],
        'volume': [100, 150, 120],
    })

def test_strategy_with_sample_data(sample_ohlcv_data):
    """Test strategy with sample data."""
    strategy = OrderFlowStrategy()
    result = strategy.analyze(sample_ohlcv_data)
    assert result is not None
```

5. **Mocking**: Mock external dependencies

```python
from unittest.mock import Mock, patch

def test_exchange_connection():
    """Test exchange connection handling."""
    with patch('ccxt.binance') as mock_exchange:
        mock_exchange.return_value.fetch_ticker.return_value = {
            'last': 50000.0
        }
        
        connector = BinanceConnector()
        ticker = connector.fetch_ticker('BTC/USDT')
        
        assert ticker['last'] == 50000.0
```

## Contributing

### Getting Started

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Make your changes**
4. **Write tests**
5. **Ensure tests pass**
6. **Commit your changes**
7. **Push to your fork**
8. **Create a Pull Request**

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Build process or tooling changes

**Examples**:

```
feat(engine): add liquidity void detection

Implements new algorithm for detecting liquidity voids
in order flow analysis.

Closes #123
```

```
fix(backtest): correct position sizing calculation

Position size was using wrong base value.
Now correctly calculates from available capital.

Fixes #456
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring
- `test/description` - Test additions

## Pull Request Process

### Before Submitting

1. **Update tests**: Add/update tests for changes
2. **Run test suite**: Ensure all tests pass
3. **Update documentation**: Keep docs in sync
4. **Format code**: Run black and ruff
5. **Type check**: Run mypy
6. **Update CHANGELOG**: Add entry for changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added
- [ ] CHANGELOG updated
```

### Review Process

1. **Automated Checks**: CI runs tests and linting
2. **Code Review**: Maintainers review code
3. **Feedback**: Address reviewer comments
4. **Approval**: Requires 1+ approvals
5. **Merge**: Squash and merge to main

## Release Process

### Versioning

Follow Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

### Release Steps

1. **Update Version**
   ```bash
   # In pyproject.toml
   version = "0.2.0"
   ```

2. **Update CHANGELOG.md**
   ```markdown
   ## [0.2.0] - 2024-01-15
   
   ### Added
   - New feature X
   - New feature Y
   
   ### Changed
   - Updated component Z
   
   ### Fixed
   - Bug fix for issue #123
   ```

3. **Create Release Branch**
   ```bash
   git checkout -b release/v0.2.0
   ```

4. **Tag Release**
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

5. **Build and Publish**
   ```bash
   poetry build
   poetry publish
   ```

## Development Workflow

### Daily Workflow

```bash
# 1. Update main branch
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and test
# ... edit files ...
pytest tests/ -v

# 4. Format and lint
black src/ tests/
ruff check src/ tests/ --fix

# 5. Commit changes
git add .
git commit -m "feat: add new feature"

# 6. Push and create PR
git push origin feature/my-feature
# Create PR on GitHub
```

### Best Practices

1. **Small, focused PRs**: One feature/fix per PR
2. **Test coverage**: Add tests for new code
3. **Documentation**: Update docs with changes
4. **Code review**: Review others' PRs
5. **Stay updated**: Pull from main regularly
6. **Communication**: Discuss major changes first

## IDE Configuration

### VSCode

`.vscode/settings.json`:
```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "editor.rulers": [88]
}
```

### PyCharm

1. **Black Integration**:
   - External Tools → Add Black
   - Configure to run on save

2. **Import Optimization**:
   - Settings → Editor → Code Style → Python
   - Enable "Optimize imports on the fly"

## Resources

- **Python Style Guide**: [PEP 8](https://pep8.org/)
- **Type Hints**: [PEP 484](https://www.python.org/dev/peps/pep-0484/)
- **Docstrings**: [Google Style Guide](https://google.github.io/styleguide/pyguide.html)
- **Testing**: [pytest documentation](https://docs.pytest.org/)

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions
- **Discord**: Join our community
- **Email**: dev@botclave.io

---

**Happy coding! 🚀**

*Last Updated: 2024-01-15*
