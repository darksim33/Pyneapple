# Documentation Style Guide

> **TL;DR** — Writing conventions for all Pyneapple documentation. Each file has a single abstract TL;DR placed directly below the H1 title; sections use American English, present tense, active voice, and inline code for anything typed. Covers principles, file structure and template, language rules, code formatting, example conventions, file naming, cross-references, and documentation scope.

---

## Principles

1. **Brief over complete.** A reader who needs every detail will read the source. A reader who needs to get started needs the essentials fast.
2. **Show, don't just tell.** Every concept should be illustrated with at least one concrete example.
3. **One idea per section.** If a section tries to explain two things, split it.
4. **One TL;DR per file, as an abstract.** Placed directly below the H1 title, before any prose. It covers the whole file — name each major section briefly so a reader can decide where to focus.

---

## File structure

Every documentation file follows this template:

```markdown
# Title

> **TL;DR** — Two to four sentences summarising the whole file. State what each major section covers so a reader can decide where to focus. Think paper abstract, not chapter summary.

---

## Section

Body text ...
```

- The TL;DR is a `> blockquote` immediately below the H1 — never a paragraph.
- It covers the whole file, not individual sections. Name each major section in 1–2 words.
- The `---` horizontal rule separates the TL;DR from the first section.
- Sections use `##`; subsections use `###`. Never go deeper than `####`.

---

## Language

| Prefer | Avoid |
|---|---|
| "Run `uv sync`" | "You should run `uv sync`" |
| "The solver minimises the residual." | "The solver will minimize the residual." |
| "Pass `seg=None` to fit all voxels." | "It is possible to pass `seg=None` in order to fit all voxels." |
| "Returns `self` for chaining." | "This method returns `self`, which allows chaining." |

- **American English** spelling (`minimize`, `color`, `behavior`).
- Avoid filler words: *simply*, *just*, *obviously*, *note that*, *please*.
- Capitalise proper names: Python, NumPy, NIfTI, NNLS, TOML.

---

## Formatting

### Inline code

Use backticks for file names, paths, function/class/parameter names, CLI flags, TOML keys, and literal values:

```
`pyneapple-pixelwise`, `--config`, `reg_order`, `PixelWiseFitter`
```

### Code blocks

Always specify the language tag:

````markdown
```bash
uv run pytest
```

```python
from pyneapple import MonoExpModel
```

```toml
[Fitting.model]
type = "monoexp"
```
````

- Shell commands → `bash`
- Python → `python`
- Config files → `toml`

### Tables

Use tables for: argument references, parameter lists, model comparisons.
Keep them scannable — move long prose into the section body.

### Callouts (GitHub Markdown)

Use sparingly — one per page at most:

```markdown
> [!NOTE]
> Extra context that does not fit the main flow.

> [!WARNING]
> Something that will cause a hard-to-debug problem if ignored.
```

Do **not** use `[!TIP]` or `[!IMPORTANT]` — they add noise.

---

## Examples

- **Runnable snippets** — include all imports and show expected output in a comment:

  ```python
  from pyneapple import MonoExpModel
  import numpy as np

  b = np.array([0, 100, 500, 1000], dtype=float)
  print(MonoExpModel().forward(b, 1000.0, 0.001))
  # [1000.   905.   606.   368.]
  ```

- **Fragments** — mark continuation with `# ...`; never present partial code as complete.

- **TOML snippets** — always include the section header (`[Fitting.model]`), not bare keys.

---

## File naming and location

| Content type | Location | File name pattern |
|---|---|---|
| User guides (how-to) | `docs/guide/` | `kebab-case.md` |
| Conceptual reference | `docs/concepts/` | `kebab-case.md` |
| Developer / contributor docs | `docs/dev/` | `PascalCase.md` |
| Shared assets | `docs/assets/` | `kebab-case` |

Existing PascalCase files (`StyleGuide.md`, `TestingGuidelines.md`) keep their names for backwards compatibility.

---

## Cross-references

- Link to other doc files with relative paths: `[Configuration](guide/configuration.md)`.
- Link to source only when a specific line is essential; prefer prose descriptions otherwise.
- Never re-explain concepts that a standard library's own docs cover well.

---

## What to document — and what not to

**Document:**
- How to install, configure, and run the tool.
- The meaning of every user-facing config key and CLI flag.
- Expected input/output shapes and units for model parameters.
- Non-obvious design decisions — record the *why*, not just the *what*.

**Do not document:**
- Internal implementation details fully captured by docstrings.
- Things obvious from reading the code.
- Planned features that do not exist yet.
