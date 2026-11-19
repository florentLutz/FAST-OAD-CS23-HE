# Updating the powertrain builder action flows

`docs/_visual/powertrain_builder_action_flows.html` draws one flow chart per user action of the powertrain builder: what the user does, which Python callback or browser JavaScript runs, and which helpers it calls, in order. This file explains how to bring that page up to date after the builder code changes, on whatever branch the maintainer names.

It is written so that a person or an AI agent can follow it step by step. The two scripts it needs are included at the end.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `REPO` | `https://github.com/florentLutz/FAST-OAD-CS23-HE.git` | Repository to read the code from |
| `BRANCH` | Given by the maintainer (at the last update: `dev_pt_file_editor_gui`) | Branch whose code the page must describe. Not necessarily `main`: the builder may not be merged there yet. |

Always use the branch the maintainer gives for this update. If none is given, ask for it rather than assuming `main`.

## Scope

**Read** (the source of truth):
- `src/fastga_he/gui/power_train_builder/*.py`, on `BRANCH`.

**Edit**:
- `docs/_visual/powertrain_builder_action_flows.html` only.

**Out of scope**: do not read the tests to decide what the code does, and do not chart or mention them (`src/fastga_he/gui/unit_tests/power_train_builder_tests/`). Do not change the other files of `docs/_visual/`, the `.rst` documentation, or the code itself.

## Procedure

### 1. Get the branch

```bash
git clone --filter=blob:none --no-checkout "$REPO" fastoad_he
cd fastoad_he
git fetch origin "$BRANCH"
git checkout "origin/$BRANCH" -- src/fastga_he/gui/power_train_builder docs/_visual
```

In an existing clone, `git fetch origin "$BRANCH"` then `git checkout "origin/$BRANCH"` (or check out the branch) does the same.

### 2. Find what changed since the last update

The page records which commit it describes, in a comment near the top:

```html
<!-- action-flows-sync: branch=dev_pt_file_editor_gui commit=8e37420 -->
```

List the builder changes since that commit:

```bash
SYNCED=8e37420   # the commit from the marker
git diff --stat "$SYNCED" "origin/$BRANCH" -- src/fastga_he/gui/power_train_builder
git log --oneline "$SYNCED..origin/$BRANCH" -- src/fastga_he/gui/power_train_builder
```

Then read the full diff of those files. If the marker is missing, or the commit is not in the branch's history (the branch was rebased, or the page was synced from another branch), review every action of the page against the code instead.

### 3. Run the checker

Save the checker (at the end of this file) outside the repository, then run it from the repository root:

```bash
python /path/to/check_action_flows.py .
```

It changes nothing and reports:

| Line | Meaning | What to do |
|---|---|---|
| `STALE name` | A name in a chart (step, sub-call, or `<code>` in a note) found nowhere in the builder code | Find what replaced it in the diff (usually a rename) and update every use in the page |
| `MODULE name` | A step's module label does not match the file that defines the function | Fix the label (see the table under *Writing the charts*) |
| `UNCHARTED name` | A callback or `_JS_*` constant wired in the code that no chart mentions | Usually a new user action or a new step of an existing one: chart it. If it is deliberately not a user action, add it to `NOT_AN_ACTION` in the checker and say so in your report |
| `SYNC ...` | The branch and commit the page was last updated from | Information only |

The checker only checks names. It cannot tell whether the order of the calls, the branches, the notes or the action descriptions are still right: step 4 covers that.

### 4. Update the charts

For each change found in step 2, open the functions involved and update the actions that use them:

- **New user entry point** (a new button, field, select, keyboard or canvas interaction, or a new `on_change` / `on_click` / `js_on_click` callback): add an action in the matching group of `GROUPS`.
- **New, removed or reordered call** inside a charted function: update that function's block, keeping the calls in the order the code makes them.
- **New condition** in a charted function: add or change a branch (`B(...)`).
- **Renamed function, attribute or column**: rename it everywhere in the page, notes included.
- **Changed behaviour, label or message shown to the user** (button text, field title, warning text, status message): update the action's trigger, result and notes to match the code's wording.
- **Moved function**: update its module label.

Check every chart that shows a changed function, not just the first one found: shared helpers such as `_push_undo`, `_rebuild_all_ports` and `_refresh_connections_table` appear in many actions. Search the page for each changed name.

Then rerun the checker until it reports `0 problem(s)`.

### 5. Update the sync marker

Set the marker to the branch and the short commit the page now describes:

```bash
git rev-parse --short "origin/$BRANCH"
```

```html
<!-- action-flows-sync: branch=<BRANCH> commit=<short commit> -->
```

### 6. Check the page renders

With Playwright installed (`pip install playwright` then `playwright install chromium`), run the render check at the end of this file:

```bash
python /path/to/render_check.py docs/_visual/powertrain_builder_action_flows.html
```

It opens every action and fails on a script error or an empty chart. Without Playwright, at least open the page in a browser and click through the actions you changed. Look at those charts too: the check cannot tell whether a chart reads well.

### 7. Report

Give the maintainer:
- the branch and commit the page now describes, and the previous ones;
- the actions added, the actions changed (one line each on what changed), and the names renamed;
- anything left as it was on purpose, such as a name added to `NOT_AN_ACTION`, or a change you could not place in a chart.

## Writing the charts

All content is in the `GROUPS` array in the page's `<script>`. Do not change the page's layout, styles or rendering code: only `GROUPS` and the sync marker.

### Actions

```js
{ id: "unknown-id", t: "Type the id prefix or type of an unknown component",
  trig: "What the user does, with <b>button / field names</b> as shown on screen.",
  undo: "Yes, one step",   // or "No (…)", "With Apply", "Not an edit", …
  res: "What the user sees afterwards.",
  flow: [ … ] },
```

- `id` is used in links (`#unknown-id`): keep existing ids unchanged.
- An `undo` starting with `Yes` is shown highlighted.
- Groups, in order: Start, Build, Configure, Delete, History, View, File and session. Put a new action in the group that matches what the user is doing.

### Steps

| Builder | Use |
|---|---|
| `S(kind, name, module, note, subCalls)` | One step. `subCalls` is an optional list of names it calls, drawn as small chips |
| `C(kind, name, module, note, [steps])` | A function whose own calls are drawn inside its block, in order |
| `B([condition, [steps]], …, [condition, [steps], true])` | Branches; `true` marks the "otherwise" arm |
| `E(text)` | End of a path, e.g. "Committed by Apply." |
| `PUSH_UNDO`, `RESTORE` | Shared sub-call lists of `_push_undo` and `_restore_from_snapshot` |

Kinds:

| Kind | Colour in the legend | For |
|---|---|---|
| `ui` | User / browser | What the user clicks or types |
| `js` | Browser-side JavaScript | `CustomJS` code, `_JS_*` constants |
| `cb` | Python callback | The Python function Bokeh calls (`on_click`, `on_change`, `on_event`) |
| `h` | Handler helper | Methods of the handler mixins |
| `d` | Data and state helper | Data helpers and the attributes / columns being set |
| `u` | Undo / redo history | `_push_undo`, snapshots, `EditHistoryStack` |
| `io` | File, dialog or session | Dialogs, reading / writing files, ending the session |

Module labels and the file each one stands for:

| Label | File in `src/fastga_he/gui/power_train_builder/` |
|---|---|
| `Handler` | `power_train_builder_handler.py` |
| `CanvasMode` | `power_train_builder_handler_canvas_mode.py` |
| `Placement` | `power_train_builder_handler_placement.py` |
| `IO` | `power_train_builder_handler_io.py` |
| `Ports` | `power_train_builder_handler_ports.py` |
| `ConfigPanel` | `power_train_builder_handler_config_panel.py` |
| `Unknown` | `power_train_builder_handler_unknown.py` |
| `History` | `power_train_builder_history.py` |
| `State` | `power_train_builder_state.py` |
| `Metadata` | `power_train_builder_metadata.py` |
| `Reader` | `power_train_network_reader.py` |
| `Writer` | `power_train_network_writer.py` |
| `Layout` | `power_train_network_layout.py` |
| `Launcher`, `Launcher JS` | `power_train_builder_launcher.py` |
| `Toolbox JS` | `power_train_builder_toolbox_config_table.py` |
| `Browser` | Not code: the user's action |

A new module needs a new label: add it to this table and to `MODULE_FILES` in the checker.

### Style

- Names exactly as in the code. Put code names inside notes in `<code>…</code>`, so the checker sees them.
- Notes are short, plain sentences about what the step does for the user or the design, not how. One line is usually enough; leave the note empty when the name says it all.
- Draw the calls a reader needs to follow the action. Leave out trivial ones (logging, small formatting helpers) unless they matter to the result.
- Triggers and results use the words shown on screen (button labels, field titles, messages), copied from the code.
- Content is HTML: escape `<`, `>` and `&` as `&lt;`, `&gt;` and `&amp;`.

### Line endings

Keep the line endings the file already has: a script that rewrites the file should read and write it in binary, or as text with `newline=""`. The repository stores the file with LF line endings.

## Scripts

### check_action_flows.py

```python
"""
Check docs/_visual/powertrain_builder_action_flows.html against the builder code.

Usage:  python check_action_flows.py <repo_root>

Reports, without changing anything:
  1. STALE      names in the charts (steps, sub-calls, <code> in notes) found nowhere in the code
  2. MODULE     steps whose module label does not match the file that defines the function
  3. UNCHARTED  Python callbacks and browser JS (_JS_*) wired in the code but in no chart
  4. SYNC       the branch / commit the HTML was last updated from (sync marker)
Exit code 1 if anything is found in 1-3, else 0.
"""

import re
import sys
from pathlib import Path

BUILDER_DIR = "src/fastga_he/gui/power_train_builder"
HTML_PATH = "docs/_visual/powertrain_builder_action_flows.html"

# Module label used in the charts -> file defining its functions
MODULE_FILES = {
    "Handler": "power_train_builder_handler.py",
    "CanvasMode": "power_train_builder_handler_canvas_mode.py",
    "Placement": "power_train_builder_handler_placement.py",
    "IO": "power_train_builder_handler_io.py",
    "Ports": "power_train_builder_handler_ports.py",
    "ConfigPanel": "power_train_builder_handler_config_panel.py",
    "Unknown": "power_train_builder_handler_unknown.py",
    "History": "power_train_builder_history.py",
    "State": "power_train_builder_state.py",
    "Metadata": "power_train_builder_metadata.py",
    "Reader": "power_train_network_reader.py",
    "Writer": "power_train_network_writer.py",
    "Layout": "power_train_network_layout.py",
    "Launcher": "power_train_builder_launcher.py",
    "Launcher JS": "power_train_builder_launcher.py",
    "Toolbox JS": "power_train_builder_toolbox_config_table.py",
}

# Wired in the code but deliberately not a user action of its own
NOT_AN_ACTION = {"_JS_PIN_LEGEND"}  # keeps the legend pinned while the view moves

IDENTIFIER = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


def chart_names(html: str):
    """(steps, other_names): steps as (kind, name, module); other names from sub-calls and notes."""
    script = re.findall(r"<script>(.*?)</script>", html, re.S)[-1]
    groups = script[: script.index("const ACTIONS")]
    steps = re.findall(r'\b[SC]\(\s*"(\w+)",\s*"([^"]*)",\s*"([^"]*)"', groups)
    others = set()
    # String arrays: sub-calls, PUSH_UNDO, RESTORE
    for array in re.findall(r'\[\s*"[^"\]]*"(?:\s*,\s*"[^"\]]*")*\s*\]', groups):
        others.update(re.findall(r'"([^"]*)"', array))
    others.update(re.findall(r"<code>([^<]+)</code>", groups))
    return steps, others


def identifiers(label: str):
    """Code identifiers in a chart label, or [] for a plain description."""
    label = re.sub(r"\s*\(.*\)$", "", label.strip())  # "add_component (per node)"
    label = label.split(" = ")[0]  # "delete_mode = True"
    parts = [part.strip() for part in re.split(r" / | \| ", label)]
    return [part for part in parts if IDENTIFIER.match(part)]


def main(repo_root: Path) -> int:
    html_text = (repo_root / HTML_PATH).read_text(encoding="utf-8")
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((repo_root / BUILDER_DIR).glob("*.py"))
    }
    all_code = "\n".join(sources.values())
    defined_in = {}  # def / class name -> files
    for file_name, text in sources.items():
        for name in re.findall(r"^\s*(?:def|class)\s+(\w+)", text, re.M):
            defined_in.setdefault(name, set()).add(file_name)

    def in_code(name: str) -> bool:
        return all(re.search(rf"\b{re.escape(part)}\b", all_code) for part in name.split("."))

    steps, others = chart_names(html_text)
    problems = 0

    stale = set()
    for kind, name, _ in steps:
        if kind != "ui":
            stale.update(n for n in identifiers(name) if not in_code(n))
    for label in others:
        stale.update(n for n in identifiers(label) if not in_code(n))
    for name in sorted(stale):
        print(f"STALE      {name}: in the HTML, nowhere in {BUILDER_DIR}")
    problems += len(stale)

    seen = set()
    for kind, name, module in steps:
        for ident in identifiers(name):
            function = ident.split(".")[-1]
            expected = MODULE_FILES.get(module)
            if kind == "ui" or expected is None or function not in defined_in:
                continue
            if expected not in defined_in[function] and (ident, module) not in seen:
                seen.add((ident, module))
                actual = ", ".join(sorted(defined_in[function]))
                print(f"MODULE     {ident}: labelled {module!r}, defined in {actual}")
                problems += 1

    charted = {ident for _, name, _ in steps for ident in identifiers(name)}
    charted |= {ident for label in others for ident in identifiers(label)}
    wired = set()
    for text in sources.values():
        wired.update(re.findall(r"\bon_(?:change|click|event)\([^)]*?self\.(\w+)", text))
        wired.update(re.findall(r"functools\.partial\(\s*self\.(\w+)", text))
        wired.update(re.findall(r"on_event\(\w+,\s*handler\.(\w+)", text))
        wired.update(re.findall(r"^(_JS_\w+)\s*=", text, re.M))
    for name in sorted(wired - charted - NOT_AN_ACTION):
        print(f"UNCHARTED  {name}: wired in the code, in no chart (new user action?)")
        problems += 1

    marker = re.search(r"<!-- action-flows-sync: (.*?) -->", html_text)
    print(f"SYNC       {marker.group(1) if marker else 'no sync marker in the HTML'}")
    print(f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
```

### render_check.py

```python
"""Open every chart of the action-flows page in headless Chromium; report script errors and empty charts."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

page_url = Path(sys.argv[1]).resolve().as_uri()
with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(page_url)
    action_ids = page.eval_on_selector_all("nav.index a[data-id]", "links => links.map(l => l.dataset.id)")
    empty = []
    for action_id in action_ids:
        page.goto(f"{page_url}#{action_id}")
        page.wait_for_timeout(100)
        if page.locator(".sheet .node").count() == 0:
            empty.append(action_id)
    browser.close()
print(f"{len(action_ids)} actions, script errors: {errors or 'none'}, empty charts: {empty or 'none'}")
sys.exit(1 if errors or empty or not action_ids else 0)
```
