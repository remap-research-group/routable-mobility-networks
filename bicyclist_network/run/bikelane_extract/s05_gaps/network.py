"""gaps network — merge facility lines and intersection links into one edge layer
with shared node ids, plus a node layer.

Reads  main/<region>_bike_facilities.geojson  +  byproduct/<region>_intersection_links.geojson
Writes main/<region>_network_edges.geojson    +  main/<region>_network_nodes.geojson

Edges keep every facility attribute and gain:
  edge_id, from_node, to_node, edge_type ("facility" | "connector")
  `type` tells the three kinds apart in one field: "BikeOnly" / "Sharrow" for
  facility edges, "Intersection" for connector edges.
Nodes:
  node_id, node_type ("intersection" if any connector ends there, else
  "endpoint"), degree, node_kind (osm_junction / midpoint, intersections only)

Endpoints within `snap_m` (0.2 m) of each other share a node. Nothing is
moved: node coordinates are the first endpoint seen at that location.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..config import Config
from ..facility import INTERSECTION
from ..io import read_lines, write_lines, write_points


def build_network(fac_lines, fac_props, link_lines, link_props, snap_m=0.2):
    node_xy, node_key = [], {}

    def node_of(pt):
        k = (round(float(pt[0]) / snap_m), round(float(pt[1]) / snap_m))
        if k not in node_key:
            node_key[k] = len(node_xy)
            node_xy.append((float(pt[0]), float(pt[1])))
        return node_key[k]

    edges, props = [], []
    for ln, p in zip(fac_lines, fac_props):
        q = dict(p)
        q.pop("id", None)
        q.update(edge_type="facility", from_node=node_of(ln[0]), to_node=node_of(ln[-1]))
        edges.append(ln)
        props.append(q)
    conn_nodes = {}
    for ln, p in zip(link_lines, link_props):
        a, b = node_of(ln[0]), node_of(ln[-1])          # ln[0] = facility end, ln[-1] = node
        conn_nodes[b] = p.get("node_kind", "midpoint")
        props.append(dict(edge_type="connector", from_node=a, to_node=b,
                          node_kind=p.get("node_kind"), node_snap_m=p.get("node_snap_m"),
                          gap_m=p.get("gap_m"), facility_line=p.get("line"),
                          type=INTERSECTION, n_signs=0, obs_ratio=0.0))
        edges.append(ln)
    for i, q in enumerate(props):
        q["edge_id"] = i
    deg = Counter()
    for q in props:
        deg[q["from_node"]] += 1
        deg[q["to_node"]] += 1
    nodes = [dict(node_id=i, degree=deg[i],
                  node_type="intersection" if i in conn_nodes else "endpoint",
                  node_kind=conn_nodes.get(i))
             for i in range(len(node_xy))]
    return edges, props, node_xy, nodes


def run(cfg: Config, check: bool = False):
    fac, links = cfg.path("bikelanes_final"), cfg.path("intersection_links")
    out_e, out_n = cfg.path("network_edges"), cfg.path("network_nodes")
    print(f"gaps network  {cfg.region}\n  facilities: {fac}\n  links: {links}")
    if check:
        for f, req in ((fac, True), (links, False)):
            print(f"  {'ok ' if f.exists() else ('MISSING' if req else 'absent (no connectors)')} {f}")
        return
    fl, fp = read_lines(fac)
    ll, lp = read_lines(links) if links.exists() else ([], [])
    edges, props, node_xy, nodes = build_network(fl, fp, ll, lp)
    import networkx as nx
    G = nx.MultiGraph()
    for q in props:
        G.add_edge(q["from_node"], q["to_node"])
    n_int = sum(n["node_type"] == "intersection" for n in nodes)
    print(f"  {len(fl)} facility edges + {len(ll)} connectors → {len(edges)} edges, "
          f"{len(nodes)} nodes ({n_int} intersection), "
          f"{nx.number_connected_components(G)} connected components")
    write_lines(out_e, edges, props, cfg.crs)
    write_points(out_n, node_xy, nodes, cfg.crs)
    print(f"  main products → {out_e.parent}")
