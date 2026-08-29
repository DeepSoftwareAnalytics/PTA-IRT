#!/usr/bin/env python3
"""LLM process-summary from extracted action quadruples.

After a JSON record is prepared (see README), call:

  python traj/summarize.py --input record.json --out summary.json
  python traj/summarize.py --input-dir quads/ --out-dir traj_summaries/lite/agent_x

Default model: DeepSeek-V4-Flash (temperature 0, thinking disabled).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
FIELD_LIMIT = 2000
ISSUE_LIMIT = 4000
PATCH_LIMIT = 8000
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_api_key(keys_cfg: Path | None) -> str:
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        if key := os.environ.get(name):
            return key
    if keys_cfg and keys_cfg.is_file():
        for line in keys_cfg.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY") and v.strip():
                return v.strip()
    raise RuntimeError(
        "Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) in the environment, or in keys.cfg"
    )


def make_client(api_key: str, *, base_url: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def quadruples_to_text(quads: list, *, field_limit: int = FIELD_LIMIT) -> str:
    blocks = []
    for i, q in enumerate(quads):
        if not isinstance(q, dict):
            continue
        step = q.get("step", i)
        blocks.append(
            f"### step {step}\n"
            f"REASONING:\n{str(q.get('reasoning', ''))[:field_limit]}\n\n"
            f"TOOL: {q.get('tool_name', '')}\n"
            f"PARAMETERS:\n{str(q.get('parameters', ''))[:field_limit]}\n"
            f"RESULT:\n{str(q.get('result', ''))[:field_limit]}\n"
        )
    return "\n".join(blocks)


def build_user_prompt(
    *,
    task_goal: str,
    steps_blob: str,
    ground_truth: str,
    prediction: str,
    n_steps: int,
) -> str:
    gt = ground_truth[:PATCH_LIMIT] if ground_truth else "(not provided)"
    pred = prediction[:PATCH_LIMIT] if prediction else "(not provided)"
    return (
        f"Summarize this trajectory ({n_steps} steps, quadruples: reasoning, tool, parameters, result).\n\n"
        f"TASK GOAL (from issue):\n{task_goal[:ISSUE_LIMIT]}\n\n"
        f"ACTION STEPS:\n{steps_blob}\n\n"
        f"GROUND TRUTH PATCH:\n{gt}\n\n"
        f"AGENT PREDICTION (submitted patch):\n{pred}\n"
    )


def summarize_with_llm(client, model: str, system: str, user: str) -> str:
    extra_body: dict = {"thinking": {"type": "disabled"}}
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        extra_body=extra_body,
    )
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        text = (getattr(msg, "reasoning_content", None) or "").strip()
    return text


def _steps_blob(record: dict, *, max_steps: int | None, field_limit: int) -> tuple[str, int]:
    if blob := str(record.get("steps_blob") or record.get("action_steps") or "").strip():
        n = int(record.get("n_steps") or blob.count("### step") or 0)
        return blob, n
    quads = list(record.get("quadruples") or [])
    if max_steps:
        quads = quads[:max_steps]
    if not quads:
        raise ValueError(
            "Record needs 'quadruples' (list of {step, reasoning, tool_name, parameters, result}) "
            "or a pre-serialized 'steps_blob'"
        )
    return quadruples_to_text(quads, field_limit=field_limit), len(quads)


def run_record(
    record: dict,
    *,
    system: str,
    dry_run: bool,
    client=None,
    model: str = DEEPSEEK_DEFAULT_MODEL,
    max_steps: int | None = None,
    field_limit: int = FIELD_LIMIT,
    drop_quadruples: bool = True,
) -> dict:
    steps_blob, n_steps = _steps_blob(record, max_steps=max_steps, field_limit=field_limit)
    user = build_user_prompt(
        task_goal=str(record.get("task_goal") or ""),
        steps_blob=steps_blob,
        ground_truth=str(record.get("ground_truth") or ""),
        prediction=str(record.get("prediction") or ""),
        n_steps=n_steps,
    )
    summary = "(dry-run: no LLM call)" if dry_run else summarize_with_llm(client, model, system, user)
    out = {
        "instance_id": record.get("instance_id"),
        "n_steps": n_steps,
        "traj_quality": record.get("traj_quality") or "full",
        "summary": summary,
    }
    if not drop_quadruples:
        for key in ("task_goal", "ground_truth", "prediction", "quadruples", "steps_blob"):
            if key in record:
                out[key] = record[key]
    return out


def _iter_inputs(path: Path) -> list[tuple[Path, dict]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return [(path, rec) for rec in data if isinstance(rec, dict)]
    if isinstance(data, dict):
        return [(path, data)]
    raise ValueError(f"Expected JSON object or list in {path}")


def _out_path(src: Path, out: Path | None, out_dir: Path | None, record: dict) -> Path:
    if out is not None:
        return out
    iid = str(record.get("instance_id") or src.stem).replace("_summary", "")
    dest_dir = out_dir if out_dir is not None else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{iid}_summary.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="One JSON object or a JSON list of records")
    p.add_argument("--input-dir", type=Path, help="Directory of *.json records")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--model", default=DEEPSEEK_DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEEPSEEK_BASE_URL)
    p.add_argument("--keys-cfg", type=Path, default=Path("keys.cfg"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--field-limit", type=int, default=FIELD_LIMIT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--keep-quadruples",
        action="store_true",
        help="Write input quadruples/patches into the output JSON (default: drop them)",
    )
    args = p.parse_args()
    if bool(args.input) == bool(args.input_dir):
        raise SystemExit("Provide exactly one of --input or --input-dir")

    jobs: list[tuple[Path, dict]] = []
    if args.input:
        jobs.extend(_iter_inputs(args.input))
    else:
        for fp in sorted(args.input_dir.glob("*.json")):
            jobs.extend(_iter_inputs(fp))
    if not jobs:
        raise SystemExit("No JSON records found")

    system = (PROMPT_DIR / "summary_system.txt").read_text(encoding="utf-8")
    client = None
    if not args.dry_run:
        client = make_client(load_api_key(args.keys_cfg), base_url=args.base_url)

    n_ok = 0
    for src, rec in jobs:
        dest = _out_path(src, args.out if len(jobs) == 1 else None, args.out_dir, rec)
        if args.skip_existing and dest.is_file():
            try:
                prev = json.loads(dest.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                prev = {}
            if str(prev.get("summary") or "").strip() and prev.get("summary") != "(dry-run: no LLM call)":
                print(f"skip {dest}")
                continue
        result = run_record(
            rec,
            system=system,
            dry_run=args.dry_run,
            client=client,
            model=args.model,
            max_steps=args.max_steps,
            field_limit=args.field_limit,
            drop_quadruples=not args.keep_quadruples,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{result['n_steps']} steps -> {dest}")
        n_ok += 1
    print(f"wrote {n_ok} summaries")


if __name__ == "__main__":
    main()
