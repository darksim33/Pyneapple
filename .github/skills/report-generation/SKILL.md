---
name: report-generation
description: Generate a styled HTML project report summarizing work accomplished in a session
---

# Report Generation

## When to use me

Use this skill when:
- The user asks to generate a report or summary of work done
- The user asks for a `report.html` file
- The user asks to document what was accomplished in the session

## Output

Write a single self-contained HTML file to the project root. The file must include all CSS inline (no external dependencies). The report is viewable by opening it in any browser.

**File naming:** `<issue-or-description>-report.html` — derive the name from the GitHub issue number or a short kebab-case description of the work. Examples: `203-scikit-api-report.html`, `result-integration-report.html`, `fix-hdf5-export-report.html`. Never use the generic name `report.html`.

## Required Sections (in order)

Every report must contain these sections in this exact order. Omit a section only if it genuinely does not apply to the session.

### 1. Header

- `<h1>` with project name and a short dash-separated descriptor
- `<p class="subtitle">` with a one-line summary and the current date

### 2. At a Glance

A row of stat cards showing key metrics. Pick 3-5 of the most relevant from:

- Files Created
- Files Modified
- New Tests
- Total Tests Pass
- Bugs Fixed
- Lines Added / Removed

Use the `.stats` flex layout with `.stat` cards (`.stat-number` + `.stat-label`).

### 3. Goals

Numbered list of the session's objectives. Each item: bold title, em-dash, one-sentence description.

### 4. Key Discoveries (if any)

Use a `.callout` block for each important finding that changed the approach. Skip this section if the work was straightforward with no surprises.

### 5. Technical Design (if applicable)

For implementation work, include:
- **Architecture** — ASCII diagram or class hierarchy in a `<pre>` block
- **How It Works** — table with Aspect / Detail columns
- **Validation Rules** — bulleted list of constraints and error conditions
- **Supported Features** — bulleted list

Skip this section for non-implementation work (e.g., pure config or docs changes).

### 6. Files Changed (summary)

Table with columns: File, Action, Purpose.

- Use `<span class="badge badge-new">Created</span>` for new files
- Use `<span class="badge badge-mod">Modified</span>` for modified files
- Use `<span class="badge badge-done">Updated</span>` for updated files

Group related files under `<h3>` subheadings if there are distinct categories (e.g., "Source code", "Configuration", "Tests").

This is a compact summary table — the detail lives in section 7.

### 7. Detailed Changes

A deep-dive into every file created or modified. For each file, use an `<h3>` with the file path and a badge, then describe what was added or changed. Use `<h4>` for individual functions/methods/sections within a file.

Guidelines:
- **Created files**: describe every class, method, and key logic block. Reference line numbers where helpful (`lines 52–118`). Use bulleted lists for method-level details.
- **Modified files**: show the exact lines added or changed in `<pre>` code blocks. Explain the purpose of each change.
- **Test files**: list every test class in a table (Class, Tests count, Coverage description). List all fixtures at the end.
- **Config/docs files**: summarize what was added or updated in bulleted lists.
- Separate each file with an `<hr>` divider.
- Use `<h4>` for method-level or section-level breakdowns within a file (styled at `0.9rem`, `font-weight: 600`).

### 8. Test Coverage (if tests were written)

- Summary sentence: count of new tests, total passing, regression count
- Table with columns: Category, Count, What's Tested

Skip this section if no tests were written or modified.

### 9. Design Decisions (if applicable)

Numbered list. Each item: bold short title, em-dash, one-sentence rationale.

### 10. Configuration Example (if applicable)

A `<pre>` code block showing how to use the new feature in config (TOML, YAML, etc.). Skip if no user-facing configuration was added.

## HTML Template

Use this exact CSS and structure. Replace placeholder content with actual session data.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pyneapple — REPORT_TITLE</title>
  <style>
    :root {
      --bg: #f8f9fa;
      --card: #ffffff;
      --accent: #2563eb;
      --accent-light: #dbeafe;
      --text: #1e293b;
      --muted: #64748b;
      --border: #e2e8f0;
      --green: #16a34a;
      --green-light: #dcfce7;
      --yellow: #ca8a04;
      --yellow-light: #fef9c3;
      --red: #dc2626;
      --red-light: #fee2e2;
      --code-bg: #1e293b;
      --code-fg: #e2e8f0;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 2rem 1rem;
    }
    .container { max-width: 960px; margin: 0 auto; }
    h1 { font-size: 2rem; margin-bottom: 0.25rem; }
    .subtitle { color: var(--muted); font-size: 0.95rem; margin-bottom: 2rem; }

    section {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.5rem;
      margin-bottom: 1.25rem;
    }
    section h2 {
      font-size: 1.25rem;
      margin-bottom: 0.75rem;
      border-bottom: 2px solid var(--accent-light);
      padding-bottom: 0.4rem;
    }
    section h3 { font-size: 1rem; margin: 1rem 0 0.5rem; color: var(--accent); }
    section h4 { font-size: 0.9rem; margin: 0.75rem 0 0.4rem; color: var(--text); font-weight: 600; }

    .badge {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.15rem 0.55rem;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .badge-done { background: var(--green-light); color: var(--green); }
    .badge-warn { background: var(--yellow-light); color: var(--yellow); }
    .badge-new  { background: var(--accent-light); color: var(--accent); }
    .badge-mod  { background: #f3e8ff; color: #7c3aed; }

    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.5rem; }
    th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }
    th { background: var(--bg); font-weight: 600; }

    ul, ol { padding-left: 1.4rem; }
    li { margin-bottom: 0.3rem; }
    li code, p code, td code {
      background: var(--bg);
      padding: 0.1rem 0.35rem;
      border-radius: 4px;
      font-size: 0.85em;
    }

    pre {
      background: var(--code-bg);
      color: var(--code-fg);
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.85rem;
      line-height: 1.5;
      margin: 0.75rem 0;
    }
    pre code { background: none; padding: 0; }

    .callout {
      border-left: 4px solid var(--accent);
      background: var(--accent-light);
      padding: 0.75rem 1rem;
      border-radius: 0 6px 6px 0;
      margin: 0.75rem 0;
      font-size: 0.9rem;
    }

    .stats { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }
    .stat {
      flex: 1;
      min-width: 140px;
      text-align: center;
      padding: 1rem;
      background: var(--bg);
      border-radius: 6px;
    }
    .stat-number { font-size: 1.75rem; font-weight: 700; color: var(--accent); }
    .stat-label {
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
  </style>
</head>
<body>
<div class="container">

  <h1>Pyneapple &mdash; REPORT_TITLE</h1>
  <p class="subtitle">ONE_LINE_SUMMARY &bull; DATE</p>

  <!-- At a Glance -->
  <section>
    <h2>At a Glance</h2>
    <div class="stats">
      <div class="stat">
        <div class="stat-number">N</div>
        <div class="stat-label">Metric Label</div>
      </div>
      <!-- repeat for each stat -->
    </div>
  </section>

  <!-- Goals -->
  <section>
    <h2>1. Goals</h2>
    <ol>
      <li><strong>Goal title</strong> &mdash; Description.</li>
    </ol>
  </section>

  <!-- Key Discoveries (optional) -->
  <section>
    <h2>2. Key Discoveries</h2>
    <div class="callout">
      <strong>Finding title.</strong> Explanation of the discovery and how it changed the approach.
    </div>
  </section>

  <!-- Technical Design (optional) -->
  <section>
    <h2>3. Technical Design</h2>
    <h3>Architecture</h3>
    <pre><code>Diagram here</code></pre>
    <h3>How It Works</h3>
    <table>
      <tr><th>Aspect</th><th>Detail</th></tr>
      <tr><td>Key</td><td>Value</td></tr>
    </table>
    <h3>Validation Rules</h3>
    <ul><li>Rule</li></ul>
    <h3>Supported Features</h3>
    <ul><li>Feature</li></ul>
  </section>

  <!-- Files Changed (summary) -->
  <section>
    <h2>4. Files Changed</h2>
    <table>
      <tr><th>File</th><th>Action</th><th>Purpose</th></tr>
      <tr>
        <td><code>path/to/file.py</code></td>
        <td><span class="badge badge-new">Created</span></td>
        <td>Description</td>
      </tr>
    </table>
  </section>

  <!-- Detailed Changes -->
  <section>
    <h2>5. Detailed Changes</h2>

    <h3>path/to/created_file.py <span class="badge badge-new">Created</span></h3>
    <p>Brief description of the file.</p>
    <h4><code>method_name()</code> &mdash; lines N&ndash;M</h4>
    <ul>
      <li>What this method does and key implementation details.</li>
    </ul>

    <hr style="border: none; border-top: 1px solid var(--border); margin: 1.25rem 0;">

    <h3>path/to/modified_file.py <span class="badge badge-mod">Modified</span></h3>
    <p>What was changed and why.</p>
    <pre><code>// code showing the exact change</code></pre>
  </section>

  <!-- Test Coverage (optional) -->
  <section>
    <h2>6. Test Coverage</h2>
    <p>Summary sentence.</p>
    <table>
      <tr><th>Category</th><th>Count</th><th>What's Tested</th></tr>
      <tr><td>Category name</td><td>N</td><td>Description</td></tr>
    </table>
  </section>

  <!-- Design Decisions (optional) -->
  <section>
    <h2>7. Design Decisions</h2>
    <ol>
      <li><strong>Decision</strong> &mdash; Rationale.</li>
    </ol>
  </section>

  <!-- Configuration Example (optional) -->
  <section>
    <h2>8. Configuration Example</h2>
    <pre><code>[Section]
key = value</code></pre>
  </section>

</div>
</body>
</html>
```

## Style Rules

- **Section numbering** — number sections sequentially starting at 1, skipping omitted sections (renumber so there are no gaps)
- **Badges** — use `badge-new` for created files, `badge-mod` for modified, `badge-done` for updated/completed items
- **Code references** — always wrap file paths, class names, function names, and parameter names in `<code>` tags
- **HTML entities** — use `&mdash;` for em-dashes, `&bull;` for bullets in subtitles, `&minus;` for minus signs in math, `&ge;` / `&le;` for comparison operators
- **No external resources** — no CDN links, fonts, or scripts. The file must render fully offline.
- **Keep it concise** — one sentence per bullet/list item. Tables over paragraphs whenever possible.
