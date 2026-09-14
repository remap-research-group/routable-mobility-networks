"""Helpers shared by gaps.scan / gaps.join / gaps.intersections."""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..facility import majority_type
from ..geom import plen, walk_chains


def assemble(lines, props, nbr):
    """Chain lines along `nbr` and merge their properties.

    The joined stretches are inferred from geometry, not observed from symbols,
    so each output records how much of its length is observation:
      n_joins, gap_total_m (inferred), obs_len_m (observed), obs_ratio.
    """
    res_lines, res_props = [], []
    for seq, gaps in walk_chains(lines, nbr):
        geom = np.vstack([g for _, g in seq])
        cls = Counter()
        nsig, conf, obs = 0, 0.0, 0.0
        for i, _ in seq:
            p = props[i]
            cls.update(Counter(p.get("classes") or {}))
            nsig += int(p.get("n_signs", 0) or 0)
            conf = max(conf, float(p.get("conf_max", 0) or 0))
            obs += plen(lines[i])
        total = plen(geom)
        res_lines.append(geom)
        res_props.append(dict(
            type=majority_type(cls), n_signs=nsig, conf_max=round(conf, 3),
            classes=dict(cls), n_joins=len(gaps), gap_total_m=round(float(sum(gaps)), 1),
            obs_len_m=round(obs, 1), obs_ratio=round(obs / total, 3) if total > 0 else 1.0,
            source="bikelane", members=",".join(str(i) for i, _ in seq),
        ))
    return res_lines, res_props


def which_end(line, pt, eps: float = 0.5):
    d0 = float(np.linalg.norm(line[0] - pt))
    d1 = float(np.linalg.norm(line[-1] - pt))
    if min(d0, d1) > eps:
        return None
    return 0 if d0 <= d1 else 1
