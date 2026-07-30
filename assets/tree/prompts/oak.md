# Oak texture set

Final size for every file: **64 x 64 PNG**, nearest-neighbour pixels only.
Do not ask the generator for PBR maps, text, a logo, watermark, shadows, or
lighting gradients.

## `oak_bark.png`

> Seamless tileable oak bark, medium warm brown, irregular deep vertical furrows, chunky restrained pixel-art detail, flat albedo only. Exactly 64x64 pixels, crisp hard pixels, no anti-aliasing, no text, logo, watermark, lighting gradient, or shadow.

## `oak_endgrain.png`

> Top-down cut oak log end grain, centred concentric growth rings, pale honey-brown wood and a slightly darker heartwood. One complete square log face only: no bark edge, no tiling, no perspective. Exactly 64x64 pixels, crisp hard pixel art, no anti-aliasing, text, logo, watermark, lighting gradient, or shadow.

## `oak_leaves.png` — colour plus alpha

> Seamless tileable sparse oak canopy sprite, small rounded leaf clusters in rich natural green with a few darker pixels and subtle veins, isolated on a perfectly flat #ff00ff chroma-key background. No branches, sky, ground, shadow, or gradient. Exactly 64x64 pixels, crisp hard pixel art, no anti-aliasing, no text, logo, or watermark.

After generation, remove only the exact `#ff00ff` background to alpha. The
result must remain RGBA: leaf pixels opaque and background transparent. Do not
generate a second AI mask — extracting it from this same image keeps colour and
silhouette exactly aligned.
