"""Sanity-check the generator endpoint before committing hours to a battery.

A generator failure is expensive and quiet: the run proceeds, produces almost no
documents, and finishes with a plausible-looking PASS@1 that actually measures
the untouched corpus. This sends one real generation prompt and reports whether
the answer is usable — model reachable, output parses, no reasoning preamble.

    python db_scripts/check_gen_endpoint.py --config \
        test_configs_experimental_17/m3_howto_qwen3_8b.yaml

    # or point at an endpoint directly
    python db_scripts/check_gen_endpoint.py --url http://0.0.0.0:7115/v1 \
        --model Qwen/Qwen3-8B --no-thinking
"""

import argparse
import os
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required")


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


SYSTEM = ("You are a senior Python engineer writing short, precise "
          "knowledge-base documents for a retrieval system.")

PROBE = (
    "Write ONE short how-to document about `np.argsort` (numpy) for a retrieval "
    "index.\nStructure:\n  TITLE: a question a developer would type ('How do I "
    "...?').\n  Then 2-4 sentences: the recommended approach and the mistake "
    "people make.\n  Then a minimal runnable example, 2-8 lines.\n"
    "Under 120 words plus the code. Format STRICTLY as:\n"
    "TITLE: <the question>\n<text, code in ``` fences>"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="read url/model from this exp17 config")
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--no-thinking", action="store_true",
                        help="send chat_template_kwargs.enable_thinking=false")
    parser.add_argument("--no-system-role", action="store_true",
                        help="fold the system text into the user turn (Gemma)")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    os.chdir(_find_repo_root())

    url, model, no_think = args.url, args.model, args.no_thinking
    if args.config:
        config = yaml.safe_load(open(args.config, encoding="utf-8"))
        params = config["components"]["generator"]["params"]
        url = url or params.get("url")
        model = model or params.get("model_name")
        if params.get("enable_thinking") is False:
            no_think = True
        if params.get("use_system_role") is False:
            args.no_system_role = True
    if not url or not model:
        sys.exit("need --config, or both --url and --model")

    print(f"- endpoint : {url}")
    print(f"- model    : {model}")
    print(f"- thinking : {'disabled' if no_think else 'server default'}")
    print(f"- system   : {'folded into user turn' if args.no_system_role else 'system role'}")

    from openai import OpenAI
    client = OpenAI(base_url=url, api_key="vllm", timeout=args.timeout)

    try:
        served = [m.id for m in client.models.list().data]
    except Exception as exc:
        print(f"\nFAILED to reach the endpoint: {exc}")
        print("Is the model up? Check the port with set_exp17_ports.py --show")
        return 1
    print(f"- served   : {', '.join(served) or '(none)'}")
    if model not in served:
        print(f"\n!! '{model}' is not in the served list. vLLM matches the model "
              f"id exactly; fix it with:\n"
              f"   python db_scripts/set_exp17_ports.py --gen-model '{served[0]}'"
              if served else "")

    extra = {}
    if no_think:
        extra["chat_template_kwargs"] = {"enable_thinking": False}

    print("\n- sending one generation prompt ...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=(
                [{"role": "user", "content": SYSTEM + "\n\n" + PROBE}]
                if args.no_system_role else
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": PROBE}]
            ),
            temperature=0.7,
            **({"extra_body": extra} if extra else {}),
        )
    except Exception as exc:
        print(f"FAILED: {exc}")
        return 1

    text = response.choices[0].message.content or ""
    reasoning = getattr(response.choices[0].message, "reasoning_content", None)

    has_think = "<think>" in text
    has_title = bool(re.search(r"^TITLE:\s*.+$", text, re.M))
    has_code = "```" in text

    print(f"- reply    : {len(text)} chars, "
          f"{response.usage.completion_tokens if response.usage else '?'} tokens")
    print(f"  TITLE line     : {'yes' if has_title else 'NO'}")
    print(f"  code fence     : {'yes' if has_code else 'NO'}")
    print(f"  <think> in body: {'YES — see below' if has_think else 'no'}")
    if reasoning:
        print(f"  reasoning_content: {len(reasoning)} chars "
              f"(server split it out, harmless)")

    print("\n--- reply ---")
    print(text[:900] + ("..." if len(text) > 900 else ""))

    ok = has_title and has_code and not has_think
    print("\n" + ("Looks good — the generator will parse this." if ok else
                  "PROBLEM: this reply would be dropped or mangled by the parser."))
    if has_think and not no_think:
        print("Try --no-thinking, and set enable_thinking: False in the config.")
    if has_think and no_think:
        print("The server ignored enable_thinking. The generators strip <think> "
              "defensively, so runs still work, but output tokens are wasted — "
              "consider serving with --reasoning-parser qwen3.")
    if not has_title:
        print("The model is not holding the format. Check it is an instruct "
              "model and not a base checkpoint.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
