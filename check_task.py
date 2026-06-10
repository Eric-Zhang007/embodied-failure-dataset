#!/usr/bin/env python3
import json, os

d = "/home/zjc/embodied-failure-dataset/data/json_2.1.0/train/"
for name in os.listdir(d):
    if "pick_two" in name:
        sub = os.path.join(d, name)
        trials = [x for x in os.listdir(sub) if x.startswith("trial")]
        if trials:
            fn = os.path.join(sub, trials[0], "traj_data.json")
            data = json.load(open(fn))
            print("%s: task_type=%s scene=%s" % (name[:50], data.get("task_type"), data.get("scene",{}).get("floor_plan")))
