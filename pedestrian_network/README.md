# WalkTrace

<p align="center">
  <img src="misc/overview.png" alt="Overview: aerial imagery to pedestrian network" width="92%">
</p>

**WalkTrace** is a tool for automated mapping of pedestrian infrastructure from aerial imagery. A dual-branch Swin Transformer segmentation model detects the components of the pedestrian network (i.e., sidewalks, footway, crosswalks, and midblock driveway entrances) including the portions hidden under tree canopy and shadows. The pixel predictions are converted into geo-referenced polygons with per-pixel confidence scores, and finally into a topologically connected centerline network of links and nodes, ready for pedestrian accessibility, connectivity, and routing analyses.

<br>

**Highlights**

* **Robust to tree occlusion and shadows**: Recovers the network hidden under tree canopy and shadows, so both on-leaf and off-leaf imagery work as input.
* **Beyond sidewalks and crosswalks**: Also detects midblock driveway entrances, an often-ignored component of the pedestrian network.
* **Polygons with confidence scores**: Output polygons preserve geometry (e.g., sidewalk width), and each prediction carries a calibrated per-pixel confidence score for prioritizing where field survey or street view verification is needed.
* **Centerline network construction**: Polygons are skeletonized into centerlines that form the links and nodes of a topologically connected pedestrian network.

<br>

## Updates

* **[September 2026]** Version 0 released (trained on **1,018** labeled image tiles in Washington, DC: 514 from 2023 leaf-on and 504 from 2025 leaf-off imagery)

<br>
<br>

## Getting Started

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Run Our Example](#run-our-example)
4. [Run Your Project](#run-your-project)

### Requirements

**Hardware**

* 1 CUDA-enabled GPU is recommended for inference (≥ 8 GB VRAM).
* The network-construction stage (polygons → centerlines) runs on CPU only.

**Software**

* Python ≥ 3.10
* PyTorch ≥ 2.0 with a CUDA build matching your driver
* Key dependencies (installed automatically): `timm`, `albumentations`, `rasterio`, `scipy`, `scikit-image`, `numpy`, `Pillow`

**Input imagery**

* Orthorectified RGB aerial imagery (recommended at a ground sampling distance of ~0.08 m/px).
* Either on-leaf or off-leaf imagery is supported; see [Run Your Project](#run-your-project) for how to prepare your own tiles.

<br>

### Installation

**1. Clone the repository** and enter the pedestrian-network folder:

```bash
git clone https://github.com/remap-research-group/routable-mobility-networks.git
cd routable-mobility-networks/pedestrian_network
```

**2. Create the environment.** Install PyTorch first — together with
torchvision and from the index that matches your CUDA driver — then the
remaining requirements:

```bash
conda create --name pednet python=3.10 -y
conda activate pednet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # cu121 / cu126 / cpu: see pytorch.org
pip install -r requirements.txt
```

**3. Download the pretrained weights.** The checkpoint (~240 MB) is too large
for git and is attached to the GitHub Release
[`pednet-v0.1`](https://github.com/remap-research-group/routable-mobility-networks/releases/tag/pednet-v0.1).
`download.py` (standard library only) fetches it, verifies the checksum and
puts it where the code expects it:

```bash
python download.py              # weights          -> run/src/best.pt
python download.py --dataset    # + training data  -> train/v0_2026sep_image/dc_2023, dc_2025  (~2.4 GB, only for train/)
```

Manual alternative: download `best.pt` from the release page into `run/src/`,
and unzip `dc_2023.zip` / `dc_2025.zip` into `train/v0_2026sep_image/`.
After step 3, `run/src/` holds everything the tool needs: `best.pt`,
`temperature.json` (confidence calibration), `train_config.json`, `model_card.json`.

<br>

### Run Our Example

`example/input/` contains three inputs so you can verify your environment,
GPU and the expected outputs before running your own data: nine model-ready
DC tiles from 2023 (leaf-on), the same nine tiles from 2025 (leaf-off), and
one MassDOT JPEG2000 ortho that first has to be prepared (resampled to
0.08 m/px and tiled). From `pedestrian_network/`:

```bash
# A. model-ready tiles: inspect -> segment -> build the network
python run/stage_1/segmentation.py --inspect --dir example/input/dc_2023
python run/stage_1/segmentation.py --dir example/input/dc_2023 --out example/output/dc_2023

python run/stage_2/build_network.py --input example/output/dc_2023

# B. an ortho that needs preparing (0.15 m/px, 4-band, JPEG2000)
python run/stage_1/segmentation.py --inspect --dir example/input/mass_2025
python run/stage_1/prepare_image.py --dir example/input/mass_2025 --out example/output/mass_2025/prepared
python run/stage_1/segmentation.py --dir example/output/mass_2025/prepared --out example/output/mass_2025 --overlay-px 1024

python run/stage_2/build_network.py --input example/output/mass_2025
```

Every run starts by printing an inspection report of each input (format,
size, bands, CRS, resolution, coverage, tiling advice) and a report of the
loaded model. Stage 1 writes the per-pixel products, stage 2 the network:

```
example/output/dc_2023/
├── classes/                  5-class prediction per tile (GeoTIFF)
├── confidence/               calibrated confidence per tile: *_network, *_crosswalk, *_entrance
├── overlay/                  quick-look JPEGs (imagery + class colours)
├── stage1_summary.json
└── network/
    ├── links.geojson         centerline links (sidewalk | midblock | crosswalk | pseudo), length + width
    ├── nodes.geojson         endpoints / junctions / type changes
    ├── network_polygons.geojson   typed polygons, one per unique crosswalk
    ├── *_wgs84.geojson       the same three layers in WGS84 lon/lat
    ├── network_mask.tif      connected-network mosaic
    └── network_stats.json    counts, km by type, parameters
```

See [example/README.md](example/README.md) for details,
[run/README.md](run/README.md) for the output formats, and each script's header
(or `--help`) for every option.

<br>

### Run Your Project

**1. Inspect your imagery.** Put your orthorectified RGB imagery (any GDAL
raster: GeoTIFF, JPEG2000, …) in one folder and let stage 1 report what it is
and whether it is model-ready (georeferenced, 0.08 m/px):

```bash
python run/stage_1/segmentation.py --inspect --dir <path/to/your/imagery>
```

**2. Prepare it if the report says `NEEDS PREPARE`** — other resolutions,
extra bands, or one huge ortho. This resamples to 0.08 m/px, keeps RGB, and
splits the result into tiles (`--tile-px`, default 2048):

```bash
python run/stage_1/prepare_image.py --dir <path/to/your/imagery> --out <path/to/prepared>
```

**3. Run inference (stage 1).** Segments every tile with a Hann-blended
sliding window, padding each tile with imagery from its neighbours so there
are no seams, and writes class maps plus temperature-calibrated confidence
rasters:

```bash
python run/stage_1/segmentation.py --dir <path/to/prepared-or-ready tiles> --out <path/to/output>
```

**4. Build the network (stage 2, CPU only).** Mosaics all tiles, then
converts the rasters into connected network polygons and the centerline
graph of links and nodes:

```bash
python run/stage_2/build_network.py --input <path/to/output>
```

**Output classes**

| ID | Class | Description |
|----|-------|-------------|
| 0 | background | everything else |
| 1 | sidewalk | sidewalk visible in the imagery |
| 2 | tree over pedestrian infrastructure | tree canopy occluding a sidewalk, crosswalk, or midblock entrance |
| 3 | midblock entrance | driveway entrance crossing the sidewalk visible in the imagery |
| 4 | crosswalk | pedestrian crossing visible in the imagery |

Classes 1–4 together constitute the pedestrian network; the confidence rasters and centerline network are derived from their union.

**Training your own model.** The released weights were trained on Washington,
DC imagery. If your region or imagery differs enough that the predictions are
weak, you can retrain or fine-tune the segmentation model on your own labeled
tiles: the full training pipeline (dataset preparation from a CVAT export,
training, calibration, evaluation, export) and the DC training dataset are in
[`train/`](train/README.md). Exporting a run with
`train/stage_1/5_export_weights.py` drops the new weights into `run/src/`, so
the two run stages above pick them up without any other change. The
network-construction stage has no trainable parameters and works with any
model that produces the same five classes.

<br>
<br>

## Repository Structure

```
pedestrian_network/
├── download.py   # fetch the released weights (and optionally the training dataset)
├── example/      # sample imagery (DC tiles, one MassDOT ortho) + expected outputs
├── run/          # everything needed to run the tool: stage_1 segmentation, stage_2 network, src/ weights
├── train/        # how the model was trained: training code (stage_1/) and dataset (v0_2026sep_image/, fetched by download.py --dataset)
└── misc/
```