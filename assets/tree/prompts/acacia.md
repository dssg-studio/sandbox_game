# Acacia texture set

Final size for every file: **64 x 64 PNG**, nearest-neighbour pixels only.
Do not ask the generator for PBR maps, text, a logo, watermark, shadows, or
lighting gradients.

## `acacia_bark.png`

> Seamless tileable acacia bark, warm orange-brown, dark cracked patches and subtle peeling strips, chunky pixel-art detail, flat albedo only. Exactly 64x64 pixels, crisp hard pixels, no anti-aliasing, no text, logo, watermark, lighting gradient, or shadow.

## `acacia_endgrain.png`

> Top-down cut acacia log end grain, centred irregular growth rings, golden-orange wood with a darker warm heartwood. One complete square log face only: no bark edge, no tiling, no perspective. Exactly 64x64 pixels, crisp hard pixel art, no anti-aliasing, text, logo, watermark, lighting gradient, or shadow.

## `acacia_leaves.png` — colour plus alpha

> Seamless tileable acacia canopy sprite, sparse clusters of broad flat oval leaves in olive green and warm yellow-green, isolated on a perfectly flat #ff00ff chroma-key background. No branches, sky, ground, shadow, or gradient. Exactly 64x64 pixels, crisp hard pixel art, no anti-aliasing, no text, logo, or watermark.

After generation, remove only the exact `#ff00ff` background to alpha. The
result must remain RGBA: leaf pixels opaque and background transparent. Do not
generate a second AI mask — extracting it from this same image keeps colour and
silhouette exactly aligned.
