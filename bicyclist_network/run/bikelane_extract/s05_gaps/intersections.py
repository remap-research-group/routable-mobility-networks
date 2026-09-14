"""gaps intersections — remaining gaps after `gaps join`, with intersection
crossings connected through a node instead of a straight line.

Ported from 16_connect_intersections. Input is <region>_bikelanes_joined.geojson.

Two kinds of gap, handled differently:
  straight   — same as `gaps join` (lines merged)               source=geom_join
  crossing   — a node is placed, both sides link to it            source=intersection
               (lines are NOT merged: they are different roads)

A straight line through an intersection leaves no node there, so a third or
fourth leg could never share it and turns would not exist in the graph. With
a node, all legs meet. The node is the nearest OSM junction (from `bikelane
osm`) within node_snap_m, else the midpoint of the two endpoints. Bike-lane
geometry is never moved; only the node position is chosen.

Outputs:
  <region>_bikelanes_final.geojson     bike lanes (straight joins applied)
  <region>_intersection_links.geojson  endpoint→node links (node_kind, node_snap_m)
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from ..config import Config
from ..geom import build_chains, gap_candidates, is_crossing, pick_unique_ends, plen
from ..io import backup_if_exists, crs_block, read_lines, write_lines
from .common import assemble


def load_junctions(nodes_path) -> np.ndarray:
    if not nodes_path.exists():
        return np.zeros((0, 2))
    js = [ft["geometry"]["coordinates"][:2]
          for ft in json.load(open(nodes_path))["features"]
          if ft["properties"].get("is_junction")]
    return np.asarray(js, float).reshape(-1, 2)


def junction_node(p, q, jnodes, max_snap_m):
    mid = (np.asarray(p) + np.asarray(q)) / 2.0
    if len(jnodes):
        d = np.linalg.norm(jnodes - mid, axis=1)
        i = int(np.argmin(d))
        if d[i] <= max_snap_m:
            return jnodes[i], "osm_junction", float(d[i])
    return mid, "midpoint", 0.0


def connect_via_node(lines, pairs, jnodes, max_snap_m):
    """Each crossing pair → a node + two links (endpoint → node)."""
    node_of, edges = {}, []
    for pr in pairs:
        p = lines[pr["i"]][0] if pr["wi"] == 0 else lines[pr["i"]][-1]
        q = lines[pr["j"]][0] if pr["wj"] == 0 else lines[pr["j"]][-1]
        npos, kind, dsnap = junction_node(p, q, jnodes, max_snap_m)
        key = (round(float(npos[0]), 1), round(float(npos[1]), 1))
        npos = node_of.setdefault(key, npos)
        for pt, li, w in ((p, pr["i"], pr["wi"]), (q, pr["j"], pr["wj"])):
            edges.append(dict(pt=pt, node=npos, kind=kind, dsnap=dsnap,
                              li=li, w=w, gap=pr["gap"]))
    return edges, node_of


def run(cfg: Config, argv=None, check: bool = False):
    p = cfg.get("gaps.intersections")
    xr = cfg.get("gaps.scan.crossing")
    src = cfg.path("bikelanes_joined")
    center = cfg.path("centerlines_final")
    nodes = cfg.path("osm_nodes")
    out_l, out_x = cfg.path("bikelanes_final"), cfg.path("intersection_links")
    print(f"gaps intersections  {cfg.region}\n  input: {src}\n  osm nodes: {nodes}")
    if check:
        for f, req in ((src, True), (center, True), (nodes, False)):
            st = "ok " if f.exists() else ("MISSING" if req else "absent (midpoints will be used)")
            print(f"  {st} {f}")
        return

    lines, props = read_lines(src)
    background = read_lines(center)[0] if center.exists() else []
    jnodes = load_junctions(nodes)
    print(f"  {len(lines)} bike lanes {sum(map(plen, lines))/1000:.2f} km, "
          f"{len(background)} background centerlines, {len(jnodes)} OSM junction nodes")

    cands = gap_candidates(lines, p["gap_max_m"], p["ang"], p["lat_max_m"], p["tail_m"])
    keep = pick_unique_ends(cands)
    for c in keep:
        c["xr"] = bool(background) and is_crossing(
            c["p"], c["q"], background, xr["radius_m"], xr["ang"], xr["min_len_m"])
    straight = [c for c in keep if not c["xr"]]
    crossing = [c for c in keep if c["xr"]]
    print(f"  candidates: {len(straight)} straight, {len(crossing)} crossings")
    if not keep:
        print("  nothing left to join — `gaps join` already closed every gap")

    # (a) straight joins — lines merged
    pairs = [dict(i=c["i"], j=c["j"], wi=c["wi"], wj=c["wj"], gap=c["gap"]) for c in straight]
    nbr, applied = build_chains(len(lines), pairs)
    res_lines, res_props = assemble(lines, props, nbr)

    # (b) crossings — node + links, lines untouched
    xr_pairs = [dict(i=c["i"], j=c["j"], wi=c["wi"], wj=c["wj"], gap=c["gap"]) for c in crossing]
    edges, node_of = connect_via_node(lines, xr_pairs, jnodes, p["node_snap_m"])
    if edges:
        print(f"  crossings → {len(node_of)} nodes, {len(edges)} links, "
              f"node source {dict(Counter(e['kind'] for e in edges))}")

    L0, L1 = sum(map(plen, lines)), sum(map(plen, res_lines))
    xlen = sum(float(np.linalg.norm(e["node"] - e["pt"])) for e in edges)
    print(f"  {len(lines)} lines {L0/1000:.2f} km → {len(res_lines)} lines {L1/1000:.2f} km "
          f"+ {xlen/1000:.2f} km intersection links ({len(applied)} straight joins)")

    _report_components(res_lines, edges)
    write_lines(out_l, res_lines, res_props, cfg.crs)
    _write_links(out_x, edges, cfg.crs)
    print("→ next: bikelane gaps network")
    return res_lines, res_props, edges


def _report_components(res_lines, edges):
    import networkx as nx
    G = nx.MultiGraph()
    key = lambda pt: (round(float(pt[0]), 1), round(float(pt[1]), 1))
    for ln in res_lines:
        G.add_edge(key(ln[0]), key(ln[-1]))
    for e in edges:
        G.add_edge(key(e["pt"]), key(e["node"]))
    print(f"  connected components: {nx.number_connected_components(G)} "
          f"({len(res_lines)} without intersection links)")


def _write_links(path, edges, crs):
    backup_if_exists(path)
    feats = [{"type": "Feature",
              "properties": {"id": n, "source": "intersection", "node_kind": e["kind"],
                             "node_snap_m": round(e["dsnap"], 1), "line": int(e["li"]),
                             "gap_m": round(e["gap"], 1),
                             "length_m": round(float(np.linalg.norm(e["node"] - e["pt"])), 1)},
              "geometry": {"type": "LineString",
                           "coordinates": [[float(e["pt"][0]), float(e["pt"][1])],
                                           [float(e["node"][0]), float(e["node"][1])]]}}
             for n, e in enumerate(edges)]
    json.dump({"type": "FeatureCollection", "crs": crs_block(crs), "features": feats},
              open(path, "w"))
    print(f"  saved {path.name}  ({len(feats)} links)")
