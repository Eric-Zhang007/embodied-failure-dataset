import json, os, sys

d = json.load(open(sys.argv[1]))
for s in d.get("steps",[]):
    i = s.get("step_index_in_branch")
    a = s.get("action")
    ok = s.get("success")
    e = s.get("error_message","")
    p = s.get("action_params",{})
    r = s.get("eb_reasoning","")[:150]
    print("[%d] %s ok=%s" % (i, a, ok), end="")
    if p: print(" params=%s" % str(p), end="")
    if e: print(" err=" + e[:60], end="")
    if r: print(" | " + r, end="")
    print()
