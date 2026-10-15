#!/usr/bin/env python
"""Inject the Results-tab data block into the report page template.

Usage:
    python -I scripts/build_page.py --template report_page/page.template.html \
        --data results/results_page_data.json --out report_page/balmer_audit.html \
        [--panel models_panel.html]

--panel inserts the Models tab (its text holds model results, so it is kept outside this public
repository); without it the placeholder is dropped and the page has the Validation and Results tabs.
"""
import argparse
import json

PLACEHOLDER = "__RESULTS_DATA__"
PANEL = "<!--__MODELS_PANEL__-->"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", nargs="+", required=True)
    ap.add_argument("--panel", default=None, help="HTML of the Models tab (needs model data in --data)")
    a = ap.parse_args()
    with open(a.template, encoding="utf-8") as fh:
        page = fh.read()
    if page.count(PLACEHOLDER) != 1:
        raise SystemExit("template must contain the data placeholder exactly once")
    with open(a.data, encoding="utf-8") as fh:
        data = json.dumps(json.load(fh), separators=(",", ":"), ensure_ascii=False)
    data = data.replace("</", "<\\/")          # cannot close the <script> element early
    page = page.replace(PLACEHOLDER, data)
    panel = ""
    if a.panel:
        with open(a.panel, encoding="utf-8") as fh:
            panel = fh.read()
    page = page.replace(PANEL + "\n\n", panel + "\n" if panel else "").replace(PANEL, panel)
    for out in a.out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(page)
        print("wrote", out, len(page.encode()), "bytes")


if __name__ == "__main__":
    main()
