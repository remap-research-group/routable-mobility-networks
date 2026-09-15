<!-- PLACEHOLDER — delete once Tile_Mappings.csv and imagery_info.json are in place -->
Example input for LEXINGTON.

In the repository:

    Tile_Mappings.csv     image_name, CRS_X, CRS_Y, Pixel_X, Pixel_Y for every tile of the town
    imagery_info.json     crs, resolution_m, tile_px, tile_overlap, stride_px, origin …

Not in the repository (git-ignored) — fetched from the GitHub Release `bikenet-v0.1`:

    tiles/tile_px<X>_py<Y>.jpg       python download.py --tiles LEXINGTON
