"""Validate the agent's episode-termination rule on human replays: every time the rule
fires (lives drop, player_state 6/11, or player below the screen), does the human
actually lose a life (lives counter drops or the play ends) within the next 20 s?"""
import glob
import json
import os
import sys

import numpy as np

feat = sys.argv[1]
fired = lost = 0
lead = []
for p in sorted(glob.glob(os.path.join(feat, "*.meta.npz"))):
    m = np.load(p)
    lives, ps, ys, ctrl = m["ram_lives"].astype(int), m["ram_player_state"], m["ram_player_y_screen"], m["control"]
    n = len(lives)
    t = 1
    while t < n:
        rule = ps[t] in (6, 11) or ys[t] >= 2
        if rule and ctrl[t - 1]:
            fired += 1
            nxt = np.flatnonzero(lives[t:t + 300] < lives[t - 1])
            ok = len(nxt) > 0 or t + 300 >= n
            lost += ok
            if len(nxt):
                lead.append(int(nxt[0]))
            t += 300
        else:
            t += 1
print(json.dumps({"rule_fired_from_control": fired, "followed_by_life_lost_or_play_end": lost,
                  "precision": lost / max(fired, 1),
                  "median_steps_from_rule_to_lives_drop": float(np.median(lead)) if lead else None}))
