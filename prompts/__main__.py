"""
Inspect and manage the prompt library.

    python -m prompts list                      all templates, with status (locked / draft / modified)
    python -m prompts show celtic_cross_v2      metadata and text
    python -m prompts diff celtic_cross_v1 celtic_cross_v2
    python -m prompts render celtic_cross_v2 --vars vars.json     (or --vars '{"cards": {...}}')
    python -m prompts verify                    exit 1 if a locked template was edited
    python -m prompts lock [ID ...] [--force]   lock drafts so their text can't change
"""
import argparse
import json
import sys
from pathlib import Path

from prompts.library import default_library
from prompts.template import PromptError


def cmd_list(lib, args) -> int:
    rows = [(t.id, lib.status(t), ", ".join(t.required) or "-", ", ".join(t.optional) or "-", t.description)
            for t in lib.all()]
    width = max((len(r[0]) for r in rows), default=10)
    print(f"{'template':<{width}}  {'status':<8}  {'required':<12}  {'optional':<10}  description")
    for tid, status, req, opt, desc in rows:
        print(f"{tid:<{width}}  {status:<8}  {req:<12}  {opt:<10}  {desc}")
    return 0


def cmd_show(lib, args) -> int:
    t = lib.resolve(args.template)
    print(f"{t.id}  [{lib.status(t)}]  sha256 {t.sha256[:12]}…  {t.source or ''}")
    print(f"description: {t.description}")
    print(f"required:    {list(t.required)}\noptional:    {list(t.optional)}")
    if t.based_on:
        print(f"based on:    {t.based_on}")
    if t.changes:
        print(f"changes:     {t.changes}")
    if t.system:
        print(f"\n--- system ---\n{t.system}")
    print(f"\n--- user ---\n{t.user}")
    return 0


def cmd_diff(lib, args) -> int:
    print(lib.diff(args.a, args.b))
    return 0


def cmd_render(lib, args) -> int:
    text = Path(args.vars).read_text(encoding="utf-8") if Path(args.vars).is_file() else args.vars
    rendered = lib.get(args.template).render(json.loads(text))
    if rendered.system:
        print(f"--- system ---\n{rendered.system}\n")
    print(f"--- user ---\n{rendered.user}")
    return 0


def cmd_verify(lib, args) -> int:
    problems = lib.verify()
    for p in problems:
        print("PROBLEM:", p)
    drafts = [t.id for t in lib.all() if lib.status(t) == "draft"]
    print(f"{len(lib.all())} templates; {len(problems)} problem(s); drafts: {drafts or 'none'}")
    return 1 if problems else 0


def cmd_lock(lib, args) -> int:
    written = lib.lock(args.templates or None, force=args.force)
    print(f"locked: {written}" if written else "nothing to lock")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m prompts", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("show").add_argument("template")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    r = sub.add_parser("render")
    r.add_argument("template")
    r.add_argument("--vars", required=True, help="JSON object of variables, or a path to a JSON file")
    sub.add_parser("verify")
    lk = sub.add_parser("lock")
    lk.add_argument("templates", nargs="*", help="template ids (default: every draft)")
    lk.add_argument("--force", action="store_true", help="also re-lock locked templates whose text changed")
    args = parser.parse_args(argv)
    try:
        return {"list": cmd_list, "show": cmd_show, "diff": cmd_diff, "render": cmd_render,
                "verify": cmd_verify, "lock": cmd_lock}[args.command](default_library(reload=True), args)
    except (PromptError, KeyError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
