"""
Check the model catalog (cmn/bedrock_models.py) against your Bedrock account.

  python tools/check_models.py            # do the listed model IDs exist?
  python tools/check_models.py --ping     # ...and does each one answer "Reply with OK"?
  python tools/check_models.py --all      # include unlisted catalog entries

Existence is checked with ListInferenceProfiles (global./us. IDs) and
ListFoundationModels (on-demand IDs) in each region the catalog uses. --ping
sends one tiny Converse request per model (maxTokens=256, no sampling params),
which costs a few cents in total for the listed models.

Exits with status 1 if any checked model is missing or fails its ping.
Run from the project root.
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3  # noqa: E402

import cmn_settings  # noqa: E402
from cmn.bedrock_models import FOUNDATION_MODELS  # noqa: E402

PING_PROMPT = "Reply with OK"


def available_ids(region):
    """Model IDs Bedrock offers in a region -> lifecycle status ('ACTIVE', 'LEGACY', 'PROFILE')."""
    bedrock = boto3.client("bedrock", region_name=region)
    ids = {}
    for page in bedrock.get_paginator("list_inference_profiles").paginate():
        for p in page["inferenceProfileSummaries"]:
            ids[p["inferenceProfileId"]] = "PROFILE"
    for m in bedrock.list_foundation_models()["modelSummaries"]:
        ids[m["modelId"]] = m.get("modelLifecycle", {}).get("status", "ACTIVE")
    return ids


def ping(model, region):
    """Send one tiny request; returns (ok, detail)."""
    runtime = boto3.client("bedrock-runtime", region_name=region)
    try:
        r = runtime.converse(
            modelId=model.model_id,
            messages=[{"role": "user", "content": [{"text": PING_PROMPT}]}],
            inferenceConfig={"maxTokens": 256},
        )
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"]).strip()
        return True, f"{r['metrics']['latencyMs']}ms {r['stopReason']} {text[:20]!r}"
    except Exception as e:  # report every failure, never stop the run
        message = str(e)
        if "end of its life" in message:
            return False, "END OF LIFE"
        return False, f"{type(e).__name__}: {message.split(': ', 1)[-1][:90]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ping", action="store_true", help="send a tiny request to each model")
    parser.add_argument("--all", action="store_true", help="include unlisted catalog entries")
    args = parser.parse_args()

    models = list(FOUNDATION_MODELS) if args.all else FOUNDATION_MODELS.listed_models()
    region_of = {m.model_id: m.region or cmn_settings.AWS_REGION for m in models}
    offered = {region: available_ids(region) for region in set(region_of.values())}

    pings = {}
    if args.ping:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {m.model_id: pool.submit(ping, m, region_of[m.model_id]) for m in models}
            pings = {model_id: f.result() for model_id, f in futures.items()}

    failures = 0
    for model in models:
        region = region_of[model.model_id]
        status = offered[region].get(model.model_id)
        exists = status is not None
        ok = exists and pings.get(model.model_id, (True, ""))[0]
        failures += not ok
        line = f"{'OK  ' if ok else 'FAIL'}  {model.display_name:28} {model.model_id:52} {region:10}"
        line += f" {status or 'NOT FOUND':8}"
        if model.model_id in pings:
            line += f"  {pings[model.model_id][1]}"
        if not model.listed:
            line += "  (unlisted)"
        print(line)

    print(f"\n{len(models) - failures}/{len(models)} OK" + ("" if args.ping else " (existence only; add --ping to call each model)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
