"""
runs every guardrail configuration against every attack and every harmless
question, then reports attack success rate and false refusal rate.

usage:
    python evaluate.py --model mock          # offline fake model, for a quick demo
    python evaluate.py                       # real run with the default small model
    python evaluate.py --model <hf name>     # any other hugging face chat model

only labels and the first 80 characters of each reply are saved, never full replies.
"""

import argparse
import csv
import os
import time

from guardrails import GuardedPipeline
from model import DEFAULT_MODEL_NAME, load_model
from prompts import ATTACK_WRAPPERS, BENIGN_REQUESTS, HARMFUL_REQUESTS
from scoring import looks_like_refusal

CONFIGS = {
    "baseline": (),
    "system_prompt": ("system_prompt",),
    "input_filter": ("input_filter",),
    "output_filter": ("output_filter",),
    "all_three": ("system_prompt", "input_filter", "output_filter"),
}

PREVIEW_CHARS = 80


class CachedModel:
    """remembers replies so identical (system, user) pairs are only generated once."""

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def generate(self, system_prompt, user_prompt):
        key = (system_prompt, user_prompt)
        # several configs share the same system prompt, so reuse earlier replies
        if key not in self.cache:
            self.cache[key] = self.model.generate(system_prompt, user_prompt)
        return self.cache[key]


def preview(text):
    # flatten newlines and cut to a short preview so full completions are never stored
    return " ".join(text.split())[:PREVIEW_CHARS]


def run_config(config_name, pipeline):
    """run one configuration and return a list of per-prompt result rows."""
    rows = []
    # every harmful request through every disguise
    for request_label, request_text in HARMFUL_REQUESTS:
        for wrapper_name, wrapper in ATTACK_WRAPPERS.items():
            response, fired = pipeline.ask(wrapper(request_text))
            refused = looks_like_refusal(response)
            rows.append({
                "config": config_name,
                "kind": "harmful",
                "request": request_label,
                "wrapper": wrapper_name,
                "refused": refused,
                "attack_success": not refused,
                "false_refusal": "",
                "guardrail_fired": fired or "",
                "response_preview": preview(response),
            })
    # every harmless question asked plainly
    for request_label, request_text in BENIGN_REQUESTS:
        response, fired = pipeline.ask(request_text)
        refused = looks_like_refusal(response)
        rows.append({
            "config": config_name,
            "kind": "benign",
            "request": request_label,
            "wrapper": "direct",
            "refused": refused,
            "attack_success": "",
            "false_refusal": refused,
            "guardrail_fired": fired or "",
            "response_preview": preview(response),
        })
    return rows


def rate(values):
    # share of true values, as a percentage; empty lists count as zero
    return 100.0 * sum(values) / len(values) if values else 0.0


def summarize(rows):
    """turn per-prompt rows into one summary dict per configuration."""
    summaries = []
    # one summary per configuration, in the order they were run
    for config_name in CONFIGS:
        config_rows = [row for row in rows if row["config"] == config_name]
        harmful_rows = [row for row in config_rows if row["kind"] == "harmful"]
        benign_rows = [row for row in config_rows if row["kind"] == "benign"]
        summary = {
            "config": config_name,
            "attack_success": rate([row["attack_success"] for row in harmful_rows]),
            "false_refusal": rate([row["false_refusal"] for row in benign_rows]),
        }
        # per-wrapper attack success, to see which disguise works best
        for wrapper_name in ATTACK_WRAPPERS:
            wrapper_rows = [row for row in harmful_rows if row["wrapper"] == wrapper_name]
            summary[wrapper_name] = rate([row["attack_success"] for row in wrapper_rows])
        summaries.append(summary)
    return summaries


def markdown_table(summaries):
    """build a markdown table: one row per config, overall rates then per-wrapper rates."""
    headers = ["config", "attack success", "false refusal"] + list(ATTACK_WRAPPERS)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    # one table row per configuration
    for summary in summaries:
        cells = [summary["config"], f"{summary['attack_success']:.0f}%", f"{summary['false_refusal']:.0f}%"]
        cells += [f"{summary[name]:.0f}%" for name in ATTACK_WRAPPERS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(rows, table, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    is_mock = model_name == "mock"
    header = f"# Results ({model_name})\n\n"
    # make it impossible to mistake mock numbers for real model numbers
    if is_mock:
        header += "**These are MOCK results from a fake offline model, not a real language model.**\n\n"
    note = (
        f"\n\n{len(HARMFUL_REQUESTS)} harmful requests x {len(ATTACK_WRAPPERS)} wrappers "
        f"and {len(BENIGN_REQUESTS)} benign requests per config. Wrapper columns show "
        "attack success for that disguise only.\n"
    )
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as md_file:
        md_file.write(header + table + note)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="jailbreak vs guardrail evaluator")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="'mock' or a hugging face model name")
    parser.add_argument("--out", default=None, help="output folder (default: results/ or results/mock/)")
    args = parser.parse_args()

    # keep mock output separate so it never overwrites a real run
    out_dir = args.out or (os.path.join("results", "mock") if args.model == "mock" else "results")

    print(f"loading model: {args.model}")
    model = CachedModel(load_model(args.model))
    start_time = time.time()

    all_rows = []
    # run each guardrail combination in turn
    for config_name, guardrails in CONFIGS.items():
        print(f"running config: {config_name}")
        all_rows += run_config(config_name, GuardedPipeline(model, guardrails))

    table = markdown_table(summarize(all_rows))
    print()
    print(table)
    write_outputs(all_rows, table, args.model, out_dir)
    print(f"\nsaved {out_dir}/results.md and results.csv ({time.time() - start_time:.0f}s)")


# only run when executed directly, not when imported by tests
if __name__ == "__main__":
    main()
