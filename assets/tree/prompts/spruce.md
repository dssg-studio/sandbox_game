# Spruce texture set

Final size for every file: **64 x 64 PNG**, nearest-neighbour pixels only.
Do not ask the generator for PBR maps, text, a logo, watermark, shadows, or
lighting gradients.

## `spruce_bark.png`

> Seamless tileable spruce bark, dark cool brown-gray, thin vertical plates and fine narrow fissures, restrained pixel-art detail, flat albedo only. Exactly 64x64 pixels, crisp hard pixels, no anti-aliasing, no text, logo, watermark, lighting gradient, or shadow.

## `spruce_endgrain.png`

> Top-down cut spruce log end grain, centred tight subtle growth rings, pale muted yellow-brown wood. One complete square log face only: no bark edge, no tiling, no perspective. Exactly 64x64 pixels, crisp hard pixel art, no anti-aliasing, text, logo, watermark, lighting gradient, or shadow.

## `spruce_leaves.png` — colour plus alpha

> Seamless tileable spruce needle canopy sprite, dense clustered dark-green needle sprays with restrained blue-green highlights, isolated on a perfectly flat #ff00ff chroma-key background. No branches, sky, ground, shadow, or gradient. Exactly 64x64 pixels, crisp hard pixel art, no anti-aliasing, no text, logo, or watermark.

After generation, remove only the exact `#ff00ff` background to alpha. The
result must remain RGBA: needle pixels opaque and background transparent. Do
not generate a second AI mask — extracting it from this same image keeps colour
and silhouette exactly aligned.
