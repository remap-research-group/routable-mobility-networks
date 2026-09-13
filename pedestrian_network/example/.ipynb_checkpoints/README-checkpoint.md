# example/ — run the pipeline on bundled imagery

Three ready-to-run examples that exercise every output the tool produces and
every kind of input it accepts:

| Example | Input | What it shows |
|---|---|---|
| `dc_2023` | 9 model-ready GeoTIFF tiles, 0.08 m/px, leaf-on | the plain two-stage run |
| `dc_2025` | the same 9 tiles, 2025, leaf-off | same streets without canopy |
| `mass_2025` | one MassDOT JPEG2000 ortho tile, 4-band, 0.15 m/px, 1.5 × 1.5 km | inspection → prepare (resample + tile) → two-stage run on out-of-domain imagery |

All commands below assume the environment from `run/README.md` is active and
you are in the `run/` folder. Our results are shipped in `output/provided/<name>/`
(for `mass_2025` without the regenerable `prepared/` tiles); the commands write
your own run to `output/<name>/`, so the two can be compared side by side.

<br>

## dc_2023 | dc_2025

The **same 3×3 block of Washington, DC** (grid columns 1–3, rows 22–24;
≈ 246 × 246 m) on two dates:

* `input/dc_2023/` — leaf-**on**: much of the sidewalk network is hidden
  under tree canopy, which the model recovers as class 2 (tree over
  pedestrian infrastructure);
* `input/dc_2025/` — leaf-**off**: the same streets, with building shadows
  instead.

Nine tiles per year, `tile_<TX>_<TY>.tif`, RGB GeoTIFF, 1024 × 1024 px,
0.08 m/px, EPSG:26985. They come from the training data, so treat the results
as a demonstration of the outputs, not as a held-out accuracy check.

<p align="center"><img src="../misc/example_dc_input.png" alt="DC 2023 (leaf-on) and 2025 (leaf-off) input tiles" width="92%"></p>

### Run it

```bash
cd stage_1
python segmentation.py --inspect --dir ../../example/input/dc_2023                          # 1. report only: metadata + READY?
python segmentation.py --dir ../../example/input/dc_2023 --out ../../example/output/dc_2023  # 2. inspect, load model, segment
cd ../stage_2
python build_network.py --input ../../example/output/dc_2023                                 # 3. polygons, links, nodes
```

Repeat with `dc_2025`. Every script first prints an inspection report of its
inputs; stage 1 also reports the loaded model.

### What you get

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

The confidence rasters are uint8 with a band scale of 1/255, so GDAL, QGIS
and rasterio show them as 0–1 probabilities (`p = sigmoid(logit / T)`,
calibrated on the validation split). The plain GeoJSONs are in EPSG:26985
(metres — use them for lengths and widths); the `_wgs84` copies are in WGS84
lon/lat for web maps.

<p align="center"><img src="../misc/example_dc_rasters.png" alt="Imagery, network confidence and network mask for 2023 and 2025" width="92%"></p>

The network, over the imagery: polygons (filled), links (orange sidewalk,
blue midblock entrance, red crosswalk, grey pseudo connector) and nodes
(white junction, yellow endpoint, cyan type change). Stage 2 mosaics the nine
tiles before skeletonizing, so the network is continuous across tile
boundaries; free ends at the block edge are cut by the data extent, not
errors.

<p align="center"><img src="../misc/example_dc_network.png" alt="Links, nodes and polygons for 2023 and 2025" width="92%"></p>

<br>

## mass_2025 — an ortho that needs preparing

`input/mass_2025/19TCG285980.jp2` is one MassDOT 2025 orthoimage tile (north
of Boston): 10,000 × 10,000 px at 0.15 m/px, 4 bands RGB+NIR, JPEG2000,
EPSG:6348 — the kind of file a user is likely to start from, and not
model-ready.

<p align="center"><img src="../misc/example_mass_input.png" alt="MassDOT 2025 ortho tile 19TCG285980" width="70%"></p>

### Run it

```bash
cd stage_1
python segmentation.py --inspect --dir ../../example/input/mass_2025          # -> NEEDS PREPARE
python prepare_image.py --dir ../../example/input/mass_2025 --out ../../example/output/mass_2025/prepared
python segmentation.py --dir ../../example/output/mass_2025/prepared --out ../../example/output/mass_2025 --overlay-px 1024
cd ../stage_2
python build_network.py --input ../../example/output/mass_2025
```

The first command stops at the inspection report:

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

`prepare_image.py --dir` resamples every raster in the folder to 0.08 m/px
and splits it into 10 × 10 tiles (`prepared/`, ≈ 360 MB, git-ignored).
JPEG2000 is read by the rasterio wheel from `requirements.txt`, so no extra
software is needed. Stage 2 then mosaics the 100 tiles into one 18,750 px
array, so the network is continuous over the whole 1.5 km square. On an
RTX A6000, stage 1 takes ≈ 15 min and stage 2 ≈ 13 min (one CPU core).

### What you get

The same layout as the DC examples. Expect softer, patchier predictions: the
imagery is upsampled from 0.15 m/px and the model was trained on Washington,
DC only. With default parameters the run yields 5,199 links, 5,068 nodes and
2,801 polygons over the 1.5 km square (`output/provided/mass_2025/network/network_stats.json`).

<p align="center"><img src="../misc/example_mass_rasters.png" alt="MassDOT imagery, network confidence, network mask and network" width="92%"></p>

<br>

## Notes

Raw imagery sources:

* DC 2023: https://opendata.dc.gov/datasets/aerial-photography-orthophoto-2023/about
* DC 2025: https://opendata.dc.gov/datasets/aerial-photography-orthophoto-2025/about
* MassDOT 2025: https://www.mass.gov/info-details/massgis-data-2025-aerial-imagery
