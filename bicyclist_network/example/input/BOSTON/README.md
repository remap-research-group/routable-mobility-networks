<!-- PLACEHOLDER — fill with example/make_input.py, then delete this file -->
Example input for BOSTON: a block of 1024 px tiles cut from the full imagery run.

Expected contents (written by `python example/make_input.py --src <imagery_root>/BOSTON --tile <top-left tile> --n 6`):

    tiles/tile_px<X>_py<Y>.jpg     the block (6 × 6 = 36 tiles ≈ 10 MB; up to 12 × 12 is fine)
    Tile_Mappings.csv              only the rows of those tiles
    imagery_info.json              CRS / resolution / tile size, copied from the full run
    example_input.json             what was cut and from where
