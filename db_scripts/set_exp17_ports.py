"""Point the exp17 configs at whichever ports the models are actually served on.

The generator, the solver and the embedder are three separate endpoints, and in
this setup they are brought up by hand as needed, so the ports change between
sessions. Editing eight yaml files by hand invites exactly the kind of silent
mismatch that cost a full battery in exp16 — this rewrites them all at once and
prints what it did.

    # generator on 7115, solver and query-gen on 7114, embedder on 7216
    python db_scripts/set_exp17_ports.py --gen 7115 --solver 7114 --embed 7216

    # only move the generator, leave the rest alone
    python db_scripts/set_exp17_ports.py --gen 8001

    # check what the configs currently point at
    python db_scripts/set_exp17_ports.py --show

The generator block is identified by the ``# GEN_URL`` marker the configs carry,
so a generator and a solver sharing one port stay distinguishable.
"""

import argparse
import glob
import os
import re
import sys

GEN_URL_RE = re.compile(r'^(\s*url:\s*")[^"]*("\s*#\s*GEN_URL.*)$', re.M)
GEN_MODEL_RE = re.compile(r'^(\s*model_name:\s*")[^"]*("\s*#\s*GEN_MODEL.*)$', re.M)
ANY_URL_RE = re.compile(r'^(\s*url:\s*")http://([^:]+):(\d+)(/v1"\s*(?:#.*)?)$', re.M)


COMPONENT_RE = re.compile(r"^  (\w+):\s*$")


def _find_repo_root() -> str:
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        current = start
        while True:
            if os.path.isfile(os.path.join(current, "run_ds1000.py")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    sys.exit("could not find the repo root — run this from inside the repo")


def classify(line: str, component: str) -> str:
    """Which endpoint a url line belongs to.

    The component name is authoritative; the ``# GEN_URL`` marker is a belt-and-
    braces fallback so a hand-edited config still resolves correctly.
    """
    if "GEN_URL" in line or component == "generator":
        return "gen"
    if component == "embedder":
        return "embed"
    return "solver"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default="test_configs_experimental_17")
    parser.add_argument("--gen", type=int, help="generator model port")
    parser.add_argument("--solver", type=int, help="solver + query-gen port")
    parser.add_argument("--embed", type=int, help="embedder port")
    parser.add_argument("--host", default=None,
                        help="also rewrite the host (default: leave as is)")
    parser.add_argument("--gen-model", help="override the generator model id")
    parser.add_argument("--show", action="store_true",
                        help="print current endpoints and exit")
    args = parser.parse_args()

    os.chdir(_find_repo_root())
    paths = sorted(glob.glob(os.path.join(args.configs, "*.yaml")))
    if not paths:
        sys.exit(f"no configs under {args.configs}")

    for path in paths:
        text = open(path, encoding="utf-8").read()
        lines = text.split("\n")
        component = ""
        changed = 0
        report = []

        for i, line in enumerate(lines):
            block = COMPONENT_RE.match(line)
            if block:
                component = block.group(1)
                continue

            match = ANY_URL_RE.match(line)
            if not match:
                continue
            role = classify(line, component)
            port = {"gen": args.gen, "solver": args.solver,
                    "embed": args.embed}[role]
            host = args.host or match.group(2)
            report.append((role, host, port or int(match.group(3))))
            if port is None and args.host is None:
                continue
            lines[i] = f"{match.group(1)}http://{host}:{port or match.group(3)}{match.group(4)}"
            changed += 1

        new_text = "\n".join(lines)
        if args.gen_model:
            new_text, n = GEN_MODEL_RE.subn(
                lambda m: m.group(1) + args.gen_model + m.group(2), new_text)
            changed += n

        name = os.path.basename(path)
        if args.show:
            summary = "  ".join(f"{r}={h}:{p}" for r, h, p in report)
            print(f"  {name:<34} {summary}")
            continue

        if changed:
            open(path, "w", encoding="utf-8").write(new_text)
        print(f"  {name:<34} {changed} endpoint(s) updated")

    if not args.show:
        print("\nDone. Verify with: python db_scripts/set_exp17_ports.py --show")


if __name__ == "__main__":
    main()
