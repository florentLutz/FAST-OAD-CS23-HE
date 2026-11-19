# Updating the powertrain builder action flows: instructions for an LLM session

This file is meant to be given to an LLM (you) at the start of a session. It tells you how to bring `docs/_visual/powertrain_builder_action_flows.html` up to date with the powertrain builder code on a branch the maintainer names. Follow it step by step, for any kind of code change: a new feature, a refactor, a rename, a removed feature or a bug fix.

The page draws one flow chart per user action of the builder: what the user does, which Python callback or browser JavaScript runs, and which helpers it calls, in order. Your job is to make every chart match the code, and nothing else.

## Starting a session (for the maintainer)

Give the model this file and the branch, for example:

> Follow `docs/_visual/update_action_flow_pt_builder.md` to update the action flows page for branch `<branch>`.

If the model cannot reach GitHub, also upload every `.py` file of `src/fastga_he/gui/power_train_builder/` and the current `docs/_visual/powertrain_builder_action_flows.html`, all from that branch.

## Your task

1. Get the builder code and the page from the branch.
2. Find what changed in the code since the page was last updated.
3. Update the charts so they match the code.
4. Check the page with the two scripts at the end of this file.
5. Hand back the updated page and a short report.

## Inputs

| Input | Default | Meaning |
|---|---|---|
| `BRANCH` | None | Branch whose code the page must describe. It does not have to be `main` |
| `REPO` | `https://github.com/florentLutz/FAST-OAD-CS23-HE.git` | Repository to read the code from |
| `BUILDER_DIR` | `src/fastga_he/gui/power_train_builder` | Folder of the builder code |
| `PAGE` | `docs/_visual/powertrain_builder_action_flows.html` | The action flows page |

If the maintainer has not named a branch, ask for it before doing anything else. Do not assume `main`.

## Rules

**Always:**
- Treat the code on `BRANCH` as the only source of truth. Read the whole body of each function you chart; never chart from a function's name, its docstring, the old chart, or memory.
- Copy every name and every piece of wording the user sees (button labels, field titles, warnings, status messages) from the code as it is on the branch, even if the conversation used other words.
- Change only the `GROUPS` array of the page's script and the sync marker.
- Keep every existing action `id`.
- Keep the page's line endings as they are (see *Line endings*).
- Run both scripts before handing the page back, and give their results in the report.

**Never:**
- Read the tests to work out what the code does, or chart or mention them (`src/fastga_he/gui/unit_tests/`).
- Edit the code, the tests, the `.rst` documentation or any other file. If you notice a bug in the code, put it in the report instead.
- Write a name in a chart that you have not seen in the code.
- Change the page's layout, styles or rendering code.
- Commit or push unless the maintainer asks you to.
- Run `git checkout -- <path>`, `git reset --hard` or `git stash` over files you have edited but not committed: they replace your edits with the committed version. To try something out, work on a copy, or commit your edits locally first.

**Stop and ask the maintainer when:**
- no branch was given;
- the sync marker names a different branch from the one given;
- a page or code file the maintainer uploaded differs from the one on the branch (ask which one to start from);
- a change does not fit any existing group, or would need a new kind of drawing;
- you cannot tell whether something is a user action of its own or a step of another one;
- a removed feature's action seems still wanted (for example it is linked from the documentation).

Ask your questions together, in one message. If the maintainer is not there to answer, take the most reasonable choice, carry on, and list each choice in the report.

## Procedure

### 1. Get the branch

**With repository access** (from the shell):

```bash
REPO=https://github.com/florentLutz/FAST-OAD-CS23-HE.git
BRANCH=<branch given by the maintainer>
BUILDER_DIR=src/fastga_he/gui/power_train_builder
PAGE=docs/_visual/powertrain_builder_action_flows.html

git clone --filter=blob:none --no-checkout "$REPO" fastoad_he
cd fastoad_he
git fetch origin "$BRANCH"
git checkout "origin/$BRANCH" -- "$BUILDER_DIR" "$(dirname "$PAGE")"
```

In an existing clone, run `git fetch origin "$BRANCH"`, then check out `origin/$BRANCH`.

**Without repository access**: ask the maintainer to upload the builder `.py` files and the page from the branch (see *Starting a session*). Put the uploaded files in the same folder layout (`BUILDER_DIR` and `docs/_visual/`) so the scripts find them. Without the git history, step 2 is a full review, unless the maintainer also provides the diff.

### 2. Find what changed since the last update

The page records the branch and commit it describes, in a comment near the top:

```html
<!-- action-flows-sync: branch=<branch> commit=<short commit> -->
```

List the builder changes since that commit:

```bash
SYNCED=<commit from the marker>
git log --oneline "$SYNCED..origin/$BRANCH" -- "$BUILDER_DIR"
git diff --stat "$SYNCED" "origin/$BRANCH" -- "$BUILDER_DIR"
git diff "$SYNCED" "origin/$BRANCH" -- "$BUILDER_DIR"
```

Read the whole diff, not only the stat. Then decide how far to go:

| Situation | What to review |
|---|---|
| The marker's commit is in the branch's history | The changes listed above |
| No marker; the marker names another branch (ask first, see *Rules*); or `git log` fails because the commit is not in this branch (rebased or squashed); or no git history | Every action of the page, against the code |
| No change in `BUILDER_DIR` since the marker | Nothing to update. Still run step 3 to confirm, and say so in the report |

Commits that change only the page or this file also move the branch. They don't need a chart update.

### 3. Run the checker

Save `check_action_flows.py` (at the end of this file) outside the repository, then run it from the repository root:

```bash
python /path/to/check_action_flows.py .
# with other paths:
python /path/to/check_action_flows.py . --builder-dir <dir> --html <page>
```

It changes nothing, and reports:

| Line | Meaning | What to do |
|---|---|---|
| `STALE name` | A name in a chart (step, sub-call, or `<code>` in a note) found nowhere in the builder code | It was renamed or removed: find what replaced it in the diff and update every use in the page, or remove the step |
| `MODULE name` | A step's module label does not match the file that defines the function | The function moved: fix the label |
| `LABEL label` | A module label the checker does not know, or a known label whose file is gone | A module was added, renamed or removed: update the module table below and `MODULE_FILES` in the checker together, and say so in the report |
| `UNCHARTED name` | A Python callback or `_JS_*` constant wired in the code that no chart shows | Usually a new user action, or a new step in an existing one: chart it. If it should stay out of the charts, add it to `NOT_AN_ACTION` in the checker with the reason, and say so in the report |
| `NOTE file` | A builder file with no module label | Information only. It needs a label once one of its functions is charted |
| `SYNC ...` | The branch and commit the page was last updated from | Information only |

The checker only compares names. It cannot tell whether the order of the calls, the branches, the notes, or the wording shown to the user are still right: step 4 covers that. A clean result is necessary, not sufficient.

### 4. Update the charts

Go through each change from step 2, and update every action it affects:

| Change in the code | Change in the page |
|---|---|
| New user entry point: a button, field, select, canvas interaction, or a new `on_change` / `on_click` / `on_event` / `js_on_click` callback | Add an action in the matching group |
| Removed user entry point | Remove its action. Keep the other actions' ids unchanged |
| New, removed or reordered call inside a charted function | Update that function's block, keeping the calls in the order the code makes them |
| New or changed condition | Add or change a branch (`B(...)`) |
| Renamed function, attribute, column or constant | Rename it everywhere in the page, notes included |
| Moved function | Update its module label |
| Changed label, message or behaviour the user sees (button text, field title, warning, status line, what undo does) | Update the action's trigger, result, undo and notes, with the code's wording |
| Internal change with no effect on any charted call or on what the user sees | Nothing |

Shared helpers (for example the undo push, the port rebuild, or the Connections table refresh) appear in many actions. For each changed name, search the whole page and update every chart that shows it, not just the first one found.

Edit the page in small, targeted replacements, checking each one matches exactly once, rather than rewriting the whole script. Rerun the checker until it reports `0 problem(s)`.

### 5. Update the sync marker

Set the marker to the branch and the current commit of the branch:

```bash
git rev-parse --short "origin/$BRANCH"
```

```html
<!-- action-flows-sync: branch=<BRANCH> commit=<short commit> -->
```

If the page has no marker yet, add it as the first line inside `<head>`. Without git history, ask the maintainer for the commit; if they don't know it, leave the marker as it was and say so in the report.

### 6. Check the page renders

With Playwright installed (`pip install playwright`, then `playwright install chromium`), run `render_check.py` (at the end of this file):

```bash
python /path/to/render_check.py "$PAGE"
# with an installed Chrome / Chromium instead:
python /path/to/render_check.py "$PAGE" --chrome /path/to/chrome
```

It opens every action, and fails on a script error, an empty chart, or a page with no action. If you can take screenshots, look at each chart you changed: a script cannot tell whether a chart reads well. If you cannot run a browser at all, check at least that the page's script parses (for example `node --check` on the extracted script), and say in the report that the page was not rendered.

### 7. Hand back the page and report

Give the maintainer the updated page as a file, with the same name and line endings as the original. Then report in this form, leaving out empty lines:

```text
Page updated to: <branch> @ <commit> (was: <marker before>)
Added actions: <id>: <title>, ...
Changed actions:
- <id>: <what changed, one line>
Removed actions: <id>, ...
Renamed in the page: <old> -> <new>, ...
Left out on purpose: <name>: <reason>, ...
Choices made without an answer: <choice>, ...
Not resolved: <change you could not place in a chart, or a code bug you noticed>
Checks: checker <n> problem(s); render check <n> actions, <errors> (or: not run, because ...)
```

If this file or its scripts needed changes too (see *Maintaining this file*), hand them back as well and list the changes in the report.

## Writing the charts

All the content is in the `GROUPS` array of the page's `<script>`.

### Actions

```js
{ id: "short-kebab-case-id", t: "Verb phrase naming the user action",
  trig: "What the user does, with <b>button and field names</b> as shown on screen.",
  undo: "Yes, one step",   // or "No (...)", "With Apply", "Not an edit", ...
  res: "What the user sees afterwards.",
  flow: [ /* steps */ ] },
```

- The `id` is used in links (`#short-kebab-case-id`). Never change an existing id; give a new action a new one.
- An `undo` starting with `Yes` is shown highlighted.
- Put a new action in the existing group that matches what the user is doing. A new group needs the maintainer's agreement.

### Steps

| Builder | Use |
|---|---|
| `S(kind, name, module, note, subCalls)` | One step. `subCalls` is an optional list of names it calls, drawn as small chips |
| `C(kind, name, module, note, [steps])` | A function whose own calls are drawn inside its block, in order |
| `B([condition, [steps]], ..., [condition, [steps], true])` | Branches; `true` marks the "otherwise" arm |
| `E(text)` | End of a path, e.g. "Committed by Apply." |
| Shared lists (e.g. `PUSH_UNDO`, `RESTORE`) | Sub-call lists reused by several actions, defined before `GROUPS`. Reuse them rather than copying their content |

Kinds, as defined by the page's legend (if the legend has changed, follow the page):

| Kind | Legend | For |
|---|---|---|
| `ui` | User / browser | What the user clicks or types |
| `js` | Browser-side JavaScript | `CustomJS` code, `_JS_*` constants |
| `cb` | Python callback | The Python function Bokeh calls (`on_click`, `on_change`, `on_event`) |
| `h` | Handler helper | Methods of the handler mixins |
| `d` | Data and state helper | Data helpers, and the attributes or columns being set |
| `u` | Undo / redo history | Undo pushes, snapshots, the history stack |
| `io` | File, dialog or session | Dialogs, reading and writing files, ending the session |

Module labels, and the file in `BUILDER_DIR` each one stands for. This table and `MODULE_FILES` in the checker must always list the same labels:

| Label | File |
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

For a new module, pick a short label after its role (like those above), and add it to both.

### Style

- Use names exactly as in the code. Put code names in notes inside `<code>...</code>`, so the checker sees them.
- Keep notes to short, plain sentences about what the step does for the user or the design. One line is usually enough; leave the note empty when the name says it all.
- Draw the calls a reader needs to follow the action. Leave out trivial ones (logging, small formatting helpers) unless they change the result.
- Write triggers and results with the words shown on screen, copied from the code.
- Use the code's words for its data, too. One item of a `ColumnDataSource` (one node of `placed_nodes_source`, one edge of `edge_source`) is an **entry**, as in `append_entry` and `drop_entry`, and its position is a node or edge **index**. Keep **row** for rows the user sees: the Options table and the Connections panel.
- The content is HTML: write `<`, `>` and `&` as `&lt;`, `&gt;` and `&amp;`.

### Line endings

The repository stores the page with LF line endings, but a copy the maintainer uploads may have CRLF (Windows). Keep whichever the file you were given has. Edit it in a way that does not convert them: in Python, read and write in binary, or as text with `newline=""`. Before handing the page back, check that its line endings are still all of one kind.

## Maintaining this file

Update this file, and the scripts embedded below, in the same session as the page when:

- **A builder module is added, renamed or removed:** update the module table and `MODULE_FILES` (the checker reports it as `LABEL` or `NOTE`).
- **The builder code or the page moves:** update the inputs table and the defaults at the top of the checker.
- **The page's legend or step builders change:** update the kinds and steps tables. If the step builders no longer look like `S("kind", "name", "module", ...)`, or the script no longer has `const ACTIONS`, the checker stops with a message: update `chart_names()` in the checker.
- **A callback should stay out of the charts:** add it to `NOT_AN_ACTION` with the reason.
- **Something in these instructions was wrong or missing** during the session: fix it here, and list the fix in the report.

Keep the embedded scripts runnable: after changing one, copy it out of this file, run it, and check its output.

## Scripts

### check_action_flows.py

```python
"""
Check the powertrain builder action-flows page against the builder code.

Usage:
    python check_action_flows.py [REPO_ROOT] [--builder-dir DIR] [--html FILE]

Reports, without changing anything:
  STALE      names in the charts (steps, sub-calls, <code> in notes) found nowhere in the code
  MODULE     steps whose module label does not match the file that defines the function
  LABEL      module labels used in the charts that MODULE_FILES does not know
  UNCHARTED  Python callbacks and browser JS (_JS_*) wired in the code but in no chart
  NOTE       builder files with no module label (information only)
  SYNC       the branch / commit the page was last updated from (information only)
Exit code 1 if there is any STALE, MODULE, LABEL or UNCHARTED line, else 0.
"""

import argparse
import re
import sys
from pathlib import Path

DEFAULT_BUILDER_DIR = "src/fastga_he/gui/power_train_builder"
DEFAULT_HTML = "docs/_visual/powertrain_builder_action_flows.html"

# Module label used in the charts -> file (in the builder folder) defining its
# functions. Keep in step with the module table of
# docs/_visual/update_action_flow_pt_builder.md
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
# Labels that stand for no file
NON_CODE_LABELS = {"Browser"}

# Wired in the code but deliberately not charted, with the reason
NOT_AN_ACTION = {
    "_JS_PIN_LEGEND": "keeps the legend pinned while the view moves; no user action",
}

IDENTIFIER = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


def chart_names(html: str):
    """(steps, other_names): steps as (kind, name, module); other names from sub-calls and notes."""
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    if not scripts or "const ACTIONS" not in scripts[-1]:
        sys.exit("The page's script has no 'const ACTIONS': update chart_names() for its new layout.")
    charts = scripts[-1][: scripts[-1].index("const ACTIONS")]
    steps = re.findall(r'\b[SC]\(\s*"(\w+)",\s*"([^"]*)",\s*"([^"]*)"', charts)
    if not steps:
        sys.exit("No S(...) / C(...) step found: update chart_names() for the page's new layout.")
    others = set()
    # String arrays: sub-calls and shared lists (PUSH_UNDO, RESTORE, ...)
    for array in re.findall(r'\[\s*"[^"\]]*"(?:\s*,\s*"[^"\]]*")*\s*\]', charts):
        others.update(re.findall(r'"([^"]*)"', array))
    others.update(re.findall(r"<code>([^<]+)</code>", charts))
    return steps, others


def identifiers(label: str):
    """Code identifiers in a chart label, or [] for a plain description."""
    label = re.sub(r"\s*\(.*\)$", "", label.strip())  # "add_component (per node)"
    label = label.split(" = ")[0]  # "delete_mode = True"
    parts = [part.strip() for part in re.split(r" / | \| ", label)]
    return [part for part in parts if IDENTIFIER.match(part)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo_root", nargs="?", default=".", type=Path)
    parser.add_argument("--builder-dir", default=DEFAULT_BUILDER_DIR)
    parser.add_argument("--html", default=DEFAULT_HTML)
    args = parser.parse_args()

    html_text = (args.repo_root / args.html).read_text(encoding="utf-8")
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((args.repo_root / args.builder_dir).glob("*.py"))
    }
    if not sources:
        sys.exit(f"No Python file in {args.repo_root / args.builder_dir}")
    all_code = "\n".join(sources.values())
    defined_in = {}  # def / class name -> files
    for file_name, text in sources.items():
        for name in re.findall(r"^\s*(?:def|class)\s+(\w+)", text, re.M):
            defined_in.setdefault(name, set()).add(file_name)

    def in_code(name: str) -> bool:
        return all(re.search(rf"\b{re.escape(part)}\b", all_code) for part in name.split("."))

    steps, others = chart_names(html_text)
    problems = []

    # STALE
    stale = set()
    for kind, name, _ in steps:
        if kind != "ui":
            stale.update(ident for ident in identifiers(name) if not in_code(ident))
    for label in others:
        stale.update(ident for ident in identifiers(label) if not in_code(ident))
    problems += [f"STALE      {name}: in the page, nowhere in {args.builder_dir}" for name in sorted(stale)]

    # LABEL and MODULE
    for module in sorted({module for _, _, module in steps} - set(MODULE_FILES) - NON_CODE_LABELS):
        problems.append(f"LABEL      {module!r}: not in MODULE_FILES (new module? add it to the table)")
    for label, file_name in sorted(MODULE_FILES.items()):
        if file_name not in sources:
            problems.append(f"LABEL      {label!r}: its file {file_name} is not in {args.builder_dir}")
    reported = set()
    for kind, name, module in steps:
        expected = MODULE_FILES.get(module)
        for ident in identifiers(name):
            function = ident.split(".")[-1]
            if kind == "ui" or expected is None or function not in defined_in:
                continue
            if expected not in defined_in[function] and (ident, module) not in reported:
                reported.add((ident, module))
                actual = ", ".join(sorted(defined_in[function]))
                problems.append(f"MODULE     {ident}: labelled {module!r}, defined in {actual}")

    # UNCHARTED
    charted = {ident for _, name, _ in steps for ident in identifiers(name)}
    charted |= {ident for label in others for ident in identifiers(label)}
    wired = set()
    for text in sources.values():
        wired.update(re.findall(r"\bon_(?:change|click|event)\([^)]*?self\.(\w+)", text))
        wired.update(re.findall(r"functools\.partial\(\s*self\.(\w+)", text))
        wired.update(re.findall(r"on_event\(\w+,\s*handler\.(\w+)", text))
        wired.update(re.findall(r"^(_JS_\w+)\s*=", text, re.M))
    problems += [
        f"UNCHARTED  {name}: wired in the code, in no chart (new user action, or new step?)"
        for name in sorted(wired - charted - set(NOT_AN_ACTION))
    ]

    for line in problems:
        print(line)
    for file_name in sorted(set(sources) - set(MODULE_FILES.values()) - {"__init__.py"}):
        print(f"NOTE       {file_name}: no module label (fine unless one of its functions is charted)")
    marker = re.search(r"<!-- action-flows-sync: (.*?) -->", html_text)
    print(f"SYNC       {marker.group(1) if marker else 'no sync marker in the page'}")
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
```

### render_check.py

```python
"""
Open every action of the action-flows page in headless Chromium.

Usage:
    python render_check.py PAGE.html [--chrome PATH]

Fails on a script error, an empty chart, or a page with no action.
Needs Playwright: pip install playwright && playwright install chromium
(or --chrome PATH to use an installed Chrome / Chromium instead).
"""

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("page", type=Path)
parser.add_argument("--chrome", help="Path of a Chrome / Chromium executable")
args = parser.parse_args()

page_url = args.page.resolve().as_uri()
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path=args.chrome)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(page_url)
    action_ids = page.eval_on_selector_all(
        "nav.index a[data-id]", "links => links.map(link => link.dataset.id)"
    )
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