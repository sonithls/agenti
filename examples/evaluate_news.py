"""Evaluate text or a JSON object from a file using your configured API."""

import argparse
import json
import os
import sys
from pathlib import Path

from importance_agent import ImportanceAgent, ImportanceAgentError, ProviderConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", type=Path, help="UTF-8 text file, or .json containing an object")
    parser.add_argument("--instructions", required=True, help="What matters and how it should be evaluated")
    args = parser.parse_args()
    try:
        base_url = os.environ["IMPORTANCE_BASE_URL"]
        model = os.environ["IMPORTANCE_MODEL"]
    except KeyError:
        parser.error("Set IMPORTANCE_BASE_URL and IMPORTANCE_MODEL first.")
    try:
        raw = args.data_file.read_text(encoding="utf-8")
        data = json.loads(raw) if args.data_file.suffix.lower() == ".json" else raw
        agent = ImportanceAgent(
            provider=ProviderConfig(base_url=base_url, model=model),
            instructions=args.instructions,
        )
        result = agent.evaluate(data)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0
    except ImportanceAgentError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError, ValueError):
        print("Could not read input. Supply a UTF-8 text file or a valid JSON object.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
