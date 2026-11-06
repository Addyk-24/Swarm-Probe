import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ACTS = ROOT / "data" / "ours" / "acts"
RELEASED = ROOT / "data" / "narcbench" / "activations" / "qwen3_32b"
SCENARIOS = ROOT / "data" / "narcbench" / "scenarios" / "qwen3_32b"


def load_acts(folder):
    """Load one activation folder: metadata (list, one dict per row) + {"layer_26": (N, 5120), ...}."""
    folder = Path(folder)
    meta = json.loads((folder / "metadata_gen.json").read_text())
    with np.load(folder / "activations_gen.npz") as data:
        acts = {k: data[k] for k in data.files}
    assert len(meta) == acts["layer_26"].shape[0]
    return meta, acts


def domain_of(scenario_id):
    """Authors' rule (reproduce_LR.py): 'cyber_02' -> 'cyber'."""
    return "_".join(scenario_id.split("_")[:-1])


def load_run_info(scenarios_dir) -> pd.DataFrame:
    """One row per run from <run_id>/run_config.json. Vote fields are None for Transfer runs."""
    rows = []
    for file in sorted(Path(scenarios_dir).glob("*/run_config.json")):
        cfg = json.loads(file.read_text())
        rows.append({
            "run_id": cfg["run_id"],
            "domain": domain_of(cfg["scenario_id"]),
            "success_cfg": cfg.get("collusion_success"),
            "voted_target_cfg": cfg.get("colluder_voted_target"),
            "target_cfg": cfg.get("target_option"),
        })
    return pd.DataFrame(rows)


def cell_of(row) -> str:
    """Return the 2x2 cell for one row. Check mode first: control runs also have success=False."""
    if row["mode"] == "control":
        return "control"
    if row["mode"] == "implicit":
        return "implicit"
    if row["mode"] == "collusion":
        if pd.isna(row["success_cfg"]):
            return "told_unjudged"
        return "told_did" if row["success_cfg"] else "told_didnt"
    return "unknown"


def sample_table(meta, run_info) -> pd.DataFrame:
    """One row per activation row; column i = row index into the arrays."""
    df = pd.DataFrame(meta)
    df["i"] = range(len(df))

    df = df.merge(
        run_info,
        on="run_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_cfg"),
    )
    df = df.sort_values("i").reset_index(drop=True)
    assert (df["i"].to_numpy() == np.arange(len(df))).all()

    df["cell"] = df.apply(cell_of, axis=1)
    return df


def all_run_info() -> pd.DataFrame:
    """Core + Transfer runs together (run_ids never overlap), so any activation set can be joined."""
    return pd.concat([load_run_info(SCENARIOS / "core"), load_run_info(SCENARIOS / "transfer")],
                     ignore_index=True)


def load_table(folder):
    """What later steps call: (table, acts) for one activation folder."""
    meta, acts = load_acts(folder)
    return sample_table(meta, all_run_info()), acts


def check(name, df):
    runs = df.drop_duplicates("run_id")
    print(f"{name}: {len(df)} rows | {df['run_id'].nunique()} runs | "
          f"rows per run {sorted(int(n) for n in df.groupby('run_id').size().unique())}")
    print("   runs per cell:", runs["cell"].value_counts().to_dict())
    assert df["domain"].notna().all(), "some rows found no run_config"


if __name__ == "__main__":
    core, _ = load_table(ACTS / "core_reextract")
    check("core_reextract", core)
    runs = core.drop_duplicates("run_id")
    print("   told_didnt with 0 colluder votes for target:",
          int(((runs["cell"] == "told_didnt") & (runs["voted_target_cfg"] == 0)).sum()))

    implicit, _ = load_table(ACTS / "transfer_implicit")
    check("transfer_implicit", implicit)

    transfer, _ = load_table(RELEASED / "transfer")
    check("transfer (released)", transfer)

    sysonly, _ = load_table(ACTS / "sysonly")
    check("sysonly", sysonly)
