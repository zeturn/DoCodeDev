import json
import sys

path = sys.argv[1]
t = json.load(open(path, encoding="utf-8"))
print("STATUS:", t["result_status"], "| reason:", t["failure_reason"])
print("VALIDATION FAILURES:", t["validation_failures"])
print("SUMMARY:", json.dumps(t["summary"]))
print("EVENT COUNT:", len(t["events"]))
for e in t["events"]:
    et = e["event_type"]
    if et == "agent_state_after_tool":
        a = e["after"]
        tg = {k: v["status"] for k, v in a["task_graph_nodes"].items()}
        print("  agent_after %s: edit_epoch=%s cf=%s tg=%s" % (e["tool"], a["edit_epoch"], a["consecutive_failures"], tg))
    elif et == "finalization_attempt":
        print("  FINAL_ATTEMPT attempt=%s gate_allowed=%s gate_reason=%s sched_ready=%s tg_ready=%s cf_before=%s" % (
            e["attempt"], e["gate_allowed"], e["gate_reason"], e["scheduler_ready"], e["task_graph_ready"], e["consecutive_failures_before"]))
    elif et == "finalization_attempt_result":
        print("  FINAL_RESULT attempt=%s outcome=%s cf_after=%s" % (e["attempt"], e["outcome"], e["consecutive_failures_after"]))
    elif et in ("tool_started", "tool_completed", "tool_exception"):
        print("  %s %s exit=%s changed=%s" % (et, e["tool"], e.get("exit_code", ""), e.get("changed_paths_after")))
