# example/ — run the pipeline on the reference towns

Two examples, one per reference town: the **whole** area each town was run
on, so that your run can be compared with our results file for file.

| Example | Input | Tiles | Run time (1 GPU) | What it shows |
|---|---|---|---|---|
| `BOSTON` | MassGIS 2025 tiles of the Boston study area, 1024 px, 0.15 m/px | 1,925 (~4 GB) | ~1 h | dense downtown: mostly dedicated bike lanes, many intersection connectors — **start here** |
| `LEXINGTON` | the same for the whole town of Lexington, MA | 7,331 (~14 GB) | ~3 h | suburban town: shared lanes (sharrows) dominate |

The tiles are too large for git and come from the GitHub Release
`bikenet-v0.1`; everything else is in the repository. All commands assume the
environment from the root README (`pip install -e ".[all]"`) and you are in
`bicyclist_network/`.

```
example/
├── boston.yaml, lexington.yaml   # project files: imagery_root = input/, data_root = output/
├── input/<REGION>/
│   ├── Tile_Mappings.csv         # in the repository: tile name ↔ map coordinates
│   ├── imagery_info.json         # in the repository: CRS, resolution, tile size
│   └── tiles/*.jpg               # NOT in the repository — python download.py --tiles <REGION>
└── output/
    ├── provided/<REGION>/main/   # our results for the whole town (committed)
    └── bikelanes/<REGION>/…      # your run (git-ignored), same layout as any data_root
```

<br>

## 1. Get the tiles

```bash
python download.py --tiles BOSTON            # weights (if missing) + ~4 GB of tiles → example/input/BOSTON/tiles/
python download.py --tiles LEXINGTON         # ~14 GB
```

The release ships each town as `<REGION>_tiles_part01.zip`, `part02.zip`, …
(each under GitHub's 2 GB asset limit); `download.py` fetches every part,
checks it against the release's `SHA256SUMS.txt`, unpacks it and deletes the
zip. Manual alternative: download all parts of the town from
https://github.com/remap-research-group/routable-mobility-networks/releases/tag/bikenet-v0.1
and unzip each one into `example/input/<REGION>/` (they all contain a
`tiles/` folder, so they merge). Then:

```bash
bikelane config -c example/boston.yaml       # tiles_dir [ok], tile_mappings_csv [ok], weights [ok]?
```

<br>

## 2. Run it

```bash
bikelane predict     -c example/boston.yaml        # 1  masks                     (GPU, ~10 min)
bikelane centerlines -c example/boston.yaml all    # 2  lane centerlines          (CPU, ~30–60 min)
bikelane signs       -c example/boston.yaml detect # 3  symbols                   (GPU, ~10 min)
bikelane signs       -c example/boston.yaml filter
bikelane join        -c example/boston.yaml match  # 4
bikelane join        -c example/boston.yaml clean
bikelane osm         -c example/boston.yaml        #    OSM junction nodes (network access)
bikelane gaps        -c example/boston.yaml scan   # 5
bikelane gaps        -c example/boston.yaml join
bikelane gaps        -c example/boston.yaml intersections
bikelane gaps        -c example/boston.yaml network
```

Or copy the project file to the root and use the runner (inside `tmux`):

```bash
cp example/boston.yaml bikelane.yaml && bash run/run_all.sh
```

Every stage prints what it read, what it wrote and the counts to look at
(lines, km, components); `--check` before a stage lists its inputs. Both
towns use the default parameters — they are the towns the defaults were
fixed on — so the sweeps (`signs filter --sweep`, `join match --sweep`,
`gaps scan --sweep`) are optional here; run them to see how the numbers in
`run/README.md` § 6 were chosen.

<br>

## 3. What you get, and what to compare it with

```
output/bikelanes/BOSTON/
├── main/                                   the three deliverables
│   ├── boston_bike_facilities.geojson      typed facility lines: type (BikeOnly | Sharrow), n_signs, obs_ratio …
│   ├── boston_network_edges.geojson        facility lines + intersection connectors with shared node ids
│   └── boston_network_nodes.geojson        endpoints / intersections
└── byproduct/                              review layers
    ├── boston_bikelanes_match.geojson      every centerline within 2 m of an accepted symbol
    ├── boston_unmatched_signs.geojson      accepted symbols with no centerline (nearest_m)
    ├── boston_bikelanes_clean.geojson
    ├── boston_gap_join.geojson             straight-gap candidates (2-vertex lines)
    ├── boston_gap_crossing.geojson         intersection-crossing candidates
    ├── boston_bikelanes_joined.geojson
    └── boston_intersection_links.geojson   endpoint → node links before the network merge
output/predictions/BOSTON/*_pred.png        stage 1 class masks (0 bg, 1 lane, 2 curb, 3 virtual)
output/centerlines/BOSTON/                  all lane centerlines, not only bike lanes
output/signs/BOSTON/                        raw / accepted / dropped symbol detections (CSV + GeoJSON)
output/osm/BOSTON/                          OSM drive nodes with is_junction
output/logs/BOSTON/                         run logs, config snapshots, crs.txt
```

Our results for the same tiles are in `output/provided/<REGION>/main/` — the
same three files. Open yours and ours together in QGIS, or compare the counts
the last stage prints (edges, nodes, connected components) with the ones in
`output/provided/<REGION>/main/summary.txt`. Small differences are expected
from the OSM download date (`osm` fetches live data, which moves the
intersection nodes) and from GPU nondeterminism in the two models; the
facility lines and their types should be the same.

In `network_edges`, one field tells the three kinds of edge apart: `type` is
`BikeOnly` or `Sharrow` for a facility edge and `Intersection` for a connector
(`edge_type` says `facility` / `connector` as well). GeoJSONs are in the
imagery CRS (metres); see `run/README.md` § 5 for every property.

The extracted facilities (shared lanes orange, bike-only lanes blue):

<p align="center"><img src="../misc/example_boston.png" alt="Extracted bicycle facilities, Boston, MA" width="80%"></p>
<p align="center"><img src="../misc/example_lexington.png" alt="Extracted bicycle facilities, Lexington, MA" width="70%"></p>

<!-- PLACEHOLDER (no figure in the memo): a close-up of one intersection with network_edges coloured by type and the nodes on top -->

<br>

## Notes

* Raw imagery: MassGIS 2025 aerial imagery, https://www.mass.gov/info-details/massgis-data-2025-aerial-imagery —
  the tiles were cut with [`../download_MassGIS2025AerialImagery/`](../../download_MassGIS2025AerialImagery/)
  (`fetch.py` + `tile.py`), which is also how you get any other Massachusetts town.
* The Boston "study area" is the tile set the reference run used, not the full
  city limits; `example_input.json` is not needed — the extent is the extent
  of `Tile_Mappings.csv`.
