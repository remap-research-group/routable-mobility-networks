"""centerlines graph — chunk polylines → dedup → graph → centerlines_<region>_voronoi.geojson

Ported from 08c_build_graph.py. Three problems it fixes over the original
in-script graph build:
  1. nx.Graph overwrote parallel edges between the same node pair → MultiGraph
  2. snap=3 m put both ends of short lines in one node (25 % of lines) → 1.5 m
  3. adjacent chunks (overlap=1 tile) extract the same road twice → geometric dedup
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..geom import plen
from ..io import write_lines
from .extract import load_all_chunks


def dedup(lines, cell=2.0, ang_bin=10.0, len_bin=5.0):
    """Lines with similar midpoint, heading and length are the same line.
    Neighbouring cells are checked so a grid edge cannot hide a duplicate;
    the longer (not clipped by a chunk border) of a pair survives."""
    store, out = {}, []
    for ln in lines:
        L = plen(ln)
        mid = ln.mean(axis=0)
        v = ln[-1] - ln[0]
        ang = np.degrees(np.arctan2(v[1], v[0])) % 180
        cx, cy = int(round(mid[0] / cell)), int(round(mid[1] / cell))
        hit = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for rec in store.get((cx + dx, cy + dy), []):
                    m2, a2, L2, idx = rec
                    da = abs(ang - a2) % 180
                    da = min(da, 180 - da)
                    if (np.linalg.norm(mid - m2) <= cell and da <= ang_bin
                            and abs(L - L2) <= max(len_bin, 0.1 * max(L, L2))):
                        hit = rec
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            if L > hit[2]:
                out[hit[3]] = ln
        else:
            out.append(ln)
            store.setdefault((cx, cy), []).append((mid, ang, L, len(out) - 1))
    return out


def build_graph(lines, snap_m=1.5, min_comp_len_m=20.0):
    import networkx as nx

    def nid(pt):
        return (round(float(pt[0]) / snap_m) * snap_m, round(float(pt[1]) / snap_m) * snap_m)

    G = nx.MultiGraph()
    selfloop = 0
    for ln in lines:
        u, v = nid(ln[0]), nid(ln[-1])
        selfloop += u == v
        G.add_edge(u, v, geometry=ln, length=plen(ln))
    n0 = nx.number_connected_components(G)
    for comp in list(nx.connected_components(G)):
        sub = G.subgraph(comp)
        if sum(d["length"] for _, _, d in sub.edges(data=True)) < min_comp_len_m:
            G.remove_nodes_from(comp)
    return G, selfloop, n0


def run(cfg: Config, check=False):
    import networkx as nx
    g = cfg.get("centerlines.graph")
    chunk_dir = cfg.path("chunks_dir")
    out = cfg.path("centerlines_voronoi")
    print(f"centerlines graph  {cfg.region}\n  chunks: {chunk_dir}")
    if check:
        n = len(list(chunk_dir.glob(f"{cfg.region}_*.npz")))
        print(f"  {'ok ' if n else 'MISSING'} {n} chunk files")
        return

    lines, nf = load_all_chunks(chunk_dir, cfg.region)
    print(f"  {nf} chunks → {len(lines)} polylines")
    keep = [ln for ln in lines if plen(ln) >= g["min_len_m"]]
    print(f"  dropped {len(lines)-len(keep)} shorter than {g['min_len_m']} m")
    uniq = dedup(keep, g["dedup_cell_m"])
    print(f"  dropped {len(keep)-len(uniq)} chunk-overlap duplicates → {len(uniq)}")

    G, selfloop, n0 = build_graph(uniq, g["snap_m"], g["min_comp_len_m"])
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(self-loops {selfloop}), components {n0} → {nx.number_connected_components(G)} "
          f"after dropping < {g['min_comp_len_m']} m")
    edges = [d["geometry"] for _, _, d in G.edges(data=True)]
    tot = sum(map(plen, edges))
    csz = sorted((sum(d["length"] for _, _, d in G.subgraph(c).edges(data=True))
                  for c in nx.connected_components(G)), reverse=True)
    if csz:
        print(f"  total {tot/1000:.1f} km | largest component {csz[0]/1000:.2f} km | "
              f"top 10 {sum(csz[:10])/1000:.2f} km ({100*sum(csz[:10])/tot:.0f}%)")
    write_lines(out, edges, None, cfg.crs)
    print("→ next: bikelane centerlines clean")
