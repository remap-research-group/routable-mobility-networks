"""osm — OSM drive network + junction nodes for the region's tile extent.

Only what `gaps intersections` needs: a node layer with `is_junction`
(degree ≥ 3 in the simplified drive graph), in the project CRS. Uses
network_type='drive' on purpose — OSM's bike tags are kept out of the
pipeline so they can serve as an independent comparison in validation.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .io import TileGrid, write_lines, write_points


def run(cfg: Config, argv=None, check: bool = False):
    import argparse
    ap = argparse.ArgumentParser(prog="bikelane osm")
    ap.parse_args(argv)
    p = cfg.get("osm")
    csv = cfg.path("tile_mappings_csv")
    out_n, out_e = cfg.path("osm_nodes", mkdir=not check), cfg.path("osm_edges")
    print(f"osm  {cfg.region}\n  extent from: {csv}\n  nodes → {out_n}")
    if check:
        print(f"  {'ok ' if csv.exists() else 'MISSING'} {csv}")
        return
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import box

    tg = TileGrid(csv, cfg.resolution_m, cfg.tile_px)
    px = np.array([v[0] for v in tg.geo.values()])
    py = np.array([v[1] for v in tg.geo.values()])
    x0, y1 = tg.to_utm(px.min(), py.min())
    x1, y0 = tg.to_utm(px.max() + tg.tile_px, py.max() + tg.tile_px)
    b = p["buffer_m"]
    poly = gpd.GeoSeries([box(x0 - b, y0 - b, x1 + b, y1 + b)], crs=cfg.crs).to_crs(4326).iloc[0]
    print(f"  bbox {x1-x0:.0f} × {y1-y0:.0f} m, network_type={p['network_type']}")

    G = ox.graph_from_polygon(poly, network_type=p["network_type"], simplify=True)
    G = ox.project_graph(G, to_crs=cfg.crs)
    nodes, edges = ox.graph_to_gdfs(G)
    deg = dict(G.degree())
    xy = [(g.x, g.y) for g in nodes.geometry]
    props = [dict(osmid=int(i), degree=int(deg[i]), is_junction=int(deg[i] >= p["junction_min_degree"]))
             for i in nodes.index]
    write_points(out_n, xy, props, cfg.crs)
    lines = [np.asarray(g.coords, float) for g in edges.geometry]
    eprops = [dict(osmid=str(r.get("osmid")), highway=str(r.get("highway")), name=str(r.get("name")))
              for _, r in edges.iterrows()]
    write_lines(out_e, lines, eprops, cfg.crs)
    print(f"  {sum(q['is_junction'] for q in props)} junction nodes of {len(props)}")
