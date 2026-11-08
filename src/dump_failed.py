import json
from load import ROOT, ACTS, SCENARIOS, load_table


t,_ = load_table(ACTS / "core_reextract")
fails = sorted(t[t.cell == "told_didnt"].run_id.unique())



with open(ROOT / "results" / "failed_runs.txt", "w", encoding="utf-8") as f:
    for rid in fails:
        d = json.load(open(SCENARIOS / "core" / rid / "results.json", encoding="utf-8"))
        cfg = json.load(open(SCENARIOS / "core" / rid / "run_config.json", encoding="utf-8"))

        colluders = sorted(t[(t.run_id == rid) & (t.role == "colluder")].agent_name.unique())

        f.write(f"\n{'=' * 100}\n{rid} | target option {cfg['target_option']} | colluders {colluders} "
                f"| colluder votes for target {cfg['colluder_voted_target']}\n")
        
        p = json.load(open(SCENARIOS / "core" / rid / "agent_prompts.json", encoding="utf-8"))
        options = next(x["user_prompt"] for x in p if "OPTIONS UNDER REVIEW:" in x["user_prompt"])
        options = options.split("OPTIONS UNDER REVIEW:")[1].split("\n\n")[0]
        f.write(f"OPTIONS:{options}\n")

        f.write("\n--- PRIVATE ---\n" + "\n\n".join(d["private_messages"]))
        f.write("\n\n--- PUBLIC ---\n" + "\n\n".join(d["public_messages"]))
        f.write("\n\n--- FINAL ---\n" + "\n\n".join(f"[{k}] {v}" for k, v in d["final_texts"].items()))

print("wrote", len(fails), "runs to results/failed_runs.txt")
