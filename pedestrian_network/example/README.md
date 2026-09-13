# example/ — run the pipeline on bundled imagery

Three ready-to-run examples to verify your environment and see every output
the tool produces, and every kind of input it accepts:

| Example | Input | What it shows |
|---|---|---|
| `dc_2023` | 9 model-ready GeoTIFF tiles, 0.08 m/px, leaf-on | the plain two-stage run |
| `dc_2025` | the same 9 tiles, 2025, leaf-off | same streets without canopy |
| `mass_2025` | one MassDOT JPEG2000 ortho tile, 4-band, 0.15 m/px, 1.5 × 1.5 km | inspection → prepare (resample + tile) → two-stage run on out-of-domain imagery |

## dc_2023 / dc_2025

The input is the **same 3×3 block of Washington, DC** (grid columns 1–3,
rows 22–24; ≈ 246 m × 246 m) captured on two dates:

* `input/dc_2023/` — 2023, leaf-**on**: much of the sidewalk network is hidden
  under tree canopy, which the model recovers as class 2 (tree over
  pedestrian infrastructure);
* `input/dc_2025/` — 2025, leaf-**off**: the same streets with the sidewalks
  visible.

Nine tiles per year, `tile_<TX>_<TY>.tif`, RGB GeoTIFF, 1024 × 1024 px at
0.08 m/px, EPSG:26985 (NAD83 / Maryland, metres). These tiles come from the
training data (`train/v0_2026sep_image/`); most are in the training split, so
treat the results as a demonstration of the outputs, not as a held-out
accuracy check.

## Run it

No training needed — `run/src/` holds the pretrained model. With the
environment from `run/README.md` active, from the `run/` folder:

```bash
# A. model-ready tiles (example/input/dc_2023: nine 0.08 m/px GeoTIFFs)
cd stage_1
python segmentation.py --inspect --dir ../../example/input/dc_2023           # 1. report only: metadata + READY?
python segmentation.py --dir ../../example/input/dc_2023 --out ../../example/output/dc_2023   # 2. inspect, load model, segment
cd ../stage_2
python build_network.py --input ../../example/output/dc_2023                  # 3. polygons, links, nodes

# B. an ortho that needs preparing (example/input/mass_2025: JPEG2000, 4-band, 0.15 m/px)
cd ../stage_1
python segmentation.py --inspect --dir ../../example/input/mass_2025          # -> NEEDS PREPARE
python prepare_image.py --dir ../../example/input/mass_2025 --out ../../example/output/mass_2025/prepared
python segmentation.py --dir ../../example/output/mass_2025/prepared --out ../../example/output/mass_2025 --overlay-px 1024
cd ../stage_2
python build_network.py --input ../../example/output/mass_2025
```

Results land in `output/<name>/` (rasters) and `output/<name>/network/`
(GeoJSON). Every script prints an inspection report of its inputs and, for
stage 1, a report of the loaded model before it starts.

## What you get

`output/dc_<year>/` (the committed outputs were produced exactly this way):

```
output/dc_2023/
├── classes/tile_<TX>_<TY>.tif               uint8 5-class map per tile
│                                            (0 bg, 1 visible_sidewalk, 2 tree, 3 entrance, 4 crosswalk)
├── confidence/tile_<TX>_<TY>_network.tif    calibrated confidence: pixel is pedestrian network (classes 1–4)
├── confidence/tile_<TX>_<TY>_crosswalk.tif  calibrated crosswalk confidence
├── confidence/tile_<TX>_<TY>_entrance.tif   calibrated midblock-entrance confidence
├── overlay/tile_<TX>_<TY>.jpg               imagery blended with the class colours (quick look)
├── stage1_summary.json                      class shares per tile, parameters, weights used
└── network/
    ├── links.geojson / links_wgs84.geojson                        centerline links (sidewalk | midblock | crosswalk | pseudo)
    ├── nodes.geojson / nodes_wgs84.geojson                        endpoints / junctions / type changes
    ├── network_polygons.geojson / network_polygons_wgs84.geojson  typed polygons (one per unique crosswalk)
    ├── network_mask.tif                                           connected-network mosaic of the 9 tiles
    └── network_stats.json                                         counts, km by type, widths, parameters
```

The confidence rasters are uint8 with a band scale of 1/255, so GDAL/QGIS/
rasterio show them as 0–1 probabilities (`p = sigmoid(logit / T)`, calibrated
on the validation split). The plain GeoJSONs are in EPSG:26985 (metres — use
for lengths and widths); the `_wgs84` copies are in WGS84 lon/lat, ready for
[geojson.io](https://geojson.io), kepler.gl or any web map.

Because stage 2 mosaics all nine tiles before skeletonizing, the network is
continuous across the tile boundaries; free ends at the outer edge of the block
are kept (cut by the data extent, not errors).

| | 2023 (leaf-on) | 2025 (leaf-off) |
|---|---|---|
| network area (m²) | 12,003 | 9,283 |
| links / nodes / polygons | 437 / 371 / 110 | 325 / 285 / 118 |
| unique crossings | 31 | 29 |

`preview.png` shows both years side by side: polygons (filled), links
(orange sidewalk, blue midblock entrance, red crosswalk, grey pseudo
connector) and nodes (white junction, yellow endpoint, cyan type change) over
the imagery.

## mass_2025 — an ortho that needs preparing

`input/mass_2025/19TCG285980.jp2` is one MassDOT 2025 orthoimage tile (north
of Boston; 10,000 × 10,000 px at 0.15 m/px, 4 bands RGB+NIR, JPEG2000,
EPSG:6348). It is the kind of file a user is likely to start from, and it is
not model-ready. Running stage 1 on the folder stops at the inspection report:

```
── 19TCG285980.jp2 ─────────────────────────────────────────────
  format      JP2OpenJPEG, 19.4 MB
  size        10000 x 10000 px | 4 band(s) uint8 (first 3 used as RGB)
  georef      yes | CRS EPSG:6348
  extent      1,500 x 1,500 m (225.0 ha) | x 328,500.00..330,000.00, y 4,698,000.00..4,699,500.00
  lat/lon     42.41536..42.42920 N, -71.08483..-71.06616 E
  resolution  0.150 m/px | model native 0.08 m/px -> RESAMPLE x1.875 to 18750 x 18750 px
  tiling      18750 x 18750 px = 352 Mpx as one image (~16 GB RAM in stage 1)
              -> recommend --tile-px 2048 (10 x 10 = 100 tiles)
  status      NEEDS PREPARE: resample from 0.150 to 0.08 m/px; split into 10 x 10 tiles
              -> python run/stage_1/prepare_image.py 19TCG285980.jp2 --tile-px 2048
```

So part B above prepares it first (`prepare_image.py --dir` handles every
raster in the folder: resample to 0.08 m/px, split into 10 × 10 tiles), then
runs the two stages on the prepared tiles. `prepared/` (100 tiles of
1875 × 1875 px, ≈ 360 MB) is git-ignored — it is regenerated by the prepare
step. JPEG2000 is read by the rasterio wheel from `requirements.txt`, so no
extra software is needed. The outputs have the same layout as the DC
examples; stage 2 mosaics the 100 tiles into one 18,750 px array, so the
network is continuous over the whole 1.5 km square. On an RTX A6000 stage 1
takes ~15 min and stage 2 ~13 min (one CPU core).

Expect softer, patchier predictions than on DC: the imagery is upsampled from
0.15 m/px and the model was trained on Washington, DC only.

## Use it as a template for your own area

Put your imagery in one folder and run stage 1 with `--inspect` first: the
report tells you whether the files are model-ready (RGB GeoTIFF, 0.08 m/px,
georeferenced) or need `run/stage_1/prepare_image.py` as in the MassDOT
example. Tiles are located by their georeferencing, so any grid works; if a
file's CRS cannot be resolved, add `--crs EPSG:xxxx`. See `run/README.md` for
every parameter.
