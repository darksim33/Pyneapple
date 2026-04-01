---
description: Run tests with pytest
agent: build
---

Run the test suite using `uv run pytest $ARGUMENTS`. 

Execute:
```bash
uv run pytest $ARGUMENTS
```

If tests fail, focus on the failing tests and suggest fixes based on the [TestingGuidelines](docs/TestingGuidelines.md).

If all tests pass, report success.
