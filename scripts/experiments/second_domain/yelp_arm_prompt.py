"""Yelp second-domain: prompted-LLM text arm + date+entity contamination probe.

Mirrors the SEC benchmark's C6 arm and its date+ticker probe. One inference per
unique (entity_id, event_time) event emits BOTH horizons as JSON {"h1": x, "h3": y};
the probe arm sees the business's real-world identity (name, city, state,
categories) and the month but ZERO review content — any skill it shows is identity
memorisation, not reading. Entity subsample keeps the run tractable; the protocol's
inner merge against the full AR arm restricts scoring to the subsample.

Subcommands:
  sample  --panel P --meta M --out EVENTS [--n-entities 2000 --seed 2026]
  infer   --events EVENTS --model MODEL --arm {fulltext,probe} --out RAW.jsonl [--tp 4]
  collect --events EVENTS --raw-fulltext F --raw-probe P2 --panel PANEL --out-dir D
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_LO, CLIP_HI = 1.0, 5.0
MAX_TEXT_CHARS = 8_000
ARM_NAMES = {"fulltext": "llm70_chrono", "probe": "llm70_probe"}


def cmd_sample(args) -> None:
    panel = pd.read_parquet(args.panel, columns=["entity_id", "event_time", "split",
                                                 "horizon_months", "text"])
    ents = np.sort(panel.entity_id.unique())
    rng = np.random.default_rng(args.seed)
    take = set(rng.choice(ents, size=min(args.n_entities, len(ents)), replace=False))
    sub = panel[panel.entity_id.isin(take) & panel.split.isin(("val", "test"))]
    ev = (sub.drop_duplicates(["entity_id", "event_time"])
          [["entity_id", "event_time", "text"]]
          .sort_values(["entity_id", "event_time"], kind="mergesort")
          .reset_index(drop=True))
    meta = pd.read_parquet(args.meta).rename(columns={"business_id": "entity_id"})
    ev = ev.merge(meta[["entity_id", "name", "city", "state", "categories"]],
                  on="entity_id", how="left", validate="m:1")
    assert ev.name.notna().all(), "events missing business metadata"
    ev.to_parquet(args.out, index=False)
    print(f"[sample] {len(take):,} entities -> {len(ev):,} unique val/test events "
          f"-> {args.out}")


def build_prompt(row: dict, arm: str) -> str:
    month = pd.Timestamp(row["event_time"]).strftime("%B %Y")
    ident = (f'The business is "{row["name"]}" ({row["categories"] or "uncategorised"}) '
             f'in {row["city"]}, {row["state"]}.')
    task = ("Predict the average star rating (a number from 1.0 to 5.0) this business "
            "will receive from customers over the next 1 month and over the next 3 "
            'months. Respond with ONLY a JSON object: {"h1": <number>, "h3": <number>}')
    if arm == "fulltext":
        body = (f"{ident} Below are its customer reviews from {month}.\n\n"
                f"Reviews:\n{row['text'][:MAX_TEXT_CHARS]}\n\n{task}")
    else:  # probe: identity + month, zero review content
        body = (f"{ident} It received customer reviews in {month}, but the review "
                f"text is not available to you.\n\n{task}")
    return body


NUM = r"([0-9]+(?:\.[0-9]+)?)"
PAT = re.compile(r'"h1"\s*:\s*' + NUM + r'.*?"h3"\s*:\s*' + NUM, re.S)


def parse(text: str):
    m = PAT.search(text)
    if not m:
        return None
    h1, h3 = float(m.group(1)), float(m.group(2))
    if not (0.5 <= h1 <= 5.5 and 0.5 <= h3 <= 5.5):
        return None
    return float(np.clip(h1, CLIP_LO, CLIP_HI)), float(np.clip(h3, CLIP_LO, CLIP_HI))


def cmd_infer(args) -> None:
    ev = pd.read_parquet(args.events)
    print(f"[infer:{args.arm}] {len(ev):,} events, model={args.model}")
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=4096,
              gpu_memory_utilization=0.92)
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    msgs = [[{"role": "user", "content": build_prompt(r, args.arm)}]
            for r in ev.to_dict("records")]
    outs = llm.chat(msgs, sp)
    rows, retry_idx = [], []
    for i, o in enumerate(outs):
        got = parse(o.outputs[0].text)
        rows.append(got)
        if got is None:
            retry_idx.append(i)
    if retry_idx:  # one strict-format retry pass for parse failures
        print(f"[infer:{args.arm}] retrying {len(retry_idx)} parse failures")
        strict = [[{"role": "user", "content": build_prompt(ev.iloc[i].to_dict(), args.arm)
                    + ' Output exactly one line of JSON and nothing else.'}]
                  for i in retry_idx]
        for i, o in zip(retry_idx, llm.chat(strict, sp), strict=False):
            rows[i] = parse(o.outputs[0].text)
    with open(args.out, "w", encoding="utf-8") as fh:
        for (_, r), got in zip(ev.iterrows(), rows, strict=False):
            fh.write(json.dumps({
                "entity_id": r.entity_id, "event_time": str(r.event_time),
                "h1": None if got is None else got[0],
                "h3": None if got is None else got[1],
            }) + "\n")
    fail = sum(1 for x in rows if x is None)
    print(f"[infer:{args.arm}] wrote {args.out}; parse failures {fail}/{len(ev)} "
          f"({100 * fail / len(ev):.2f}%)")


def cmd_collect(args) -> None:
    panel = pd.read_parquet(args.panel, columns=["entity_id", "event_time", "split",
                                                 "horizon_months", "label"])
    panel = panel[panel.split.isin(("val", "test"))]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for arm_key, raw_path in (("fulltext", args.raw_fulltext), ("probe", args.raw_probe)):
        raw = pd.read_json(raw_path, lines=True)
        raw["event_time"] = pd.to_datetime(raw.event_time)
        long = raw.melt(id_vars=["entity_id", "event_time"],
                        value_vars=["h1", "h3"], var_name="hcol",
                        value_name="prediction")
        long["horizon_months"] = long.hcol.str.lstrip("h").astype(int)
        d = panel.merge(long[["entity_id", "event_time", "horizon_months",
                              "prediction"]],
                        on=["entity_id", "event_time", "horizon_months"],
                        how="inner", validate="1:1")
        n_nan = int(d.prediction.isna().sum())
        val_mean = float(d.loc[(d.split == "val") & d.prediction.notna(),
                               "prediction"].mean())
        d["prediction"] = d.prediction.fillna(val_mean).clip(CLIP_LO, CLIP_HI)
        arm = ARM_NAMES[arm_key]
        d["arm"] = arm
        d = d[["entity_id", "event_time", "split", "horizon_months", "label",
               "prediction", "arm"]]
        d.to_parquet(out_dir / f"preds_{arm}.parquet", index=False)
        stats[arm] = {"rows": len(d), "nan_filled": n_nan,
                      "fill_value": round(val_mean, 4)}
        print(f"[collect] {arm}: {len(d):,} rows ({n_nan} NaN filled with val mean "
              f"{val_mean:.3f}) -> preds_{arm}.parquet")
    (out_dir / "llm70_collect_stats.json").write_text(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--panel", required=True)
    s.add_argument("--meta", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-entities", type=int, default=2000)
    s.add_argument("--seed", type=int, default=2026)
    i = sub.add_parser("infer")
    i.add_argument("--events", required=True)
    i.add_argument("--model", required=True)
    i.add_argument("--arm", choices=("fulltext", "probe"), required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--tp", type=int, default=4)
    c = sub.add_parser("collect")
    c.add_argument("--events", required=True)
    c.add_argument("--raw-fulltext", required=True)
    c.add_argument("--raw-probe", required=True)
    c.add_argument("--panel", required=True)
    c.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    {"sample": cmd_sample, "infer": cmd_infer, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
