# palette-decimator

exactly what it sounds like -- writing this initially in python so I can focus on different methods of quantizing/decimating in terms of how they look, rather than worrying about implementation/efficiency yet

current methods
- no dithering
- uniform distribution (aka white noise rng to round up/down between nearest two colors)
- triangular distribution (same thing but with potentially more/less noise depending on settings)
- bayer matrix ordered dither (relatively cheap patterned dithering)

todos:
- move to using tiled white noise instead of sampling for every pixel
- blue noise generation (and perhaps other colors of noise) and dithering
- actual error diffusion dither methods (floyd-steinberg, sierra, etc.)
- actual palette generation and comparison/testing (beyond dithering techniques)
- more scoring methods? (ordered dither dramatically outperforms rng for 1-bit color, but score is about the same)
- support larger images (currently autoshrinking them to below 1'000'000px for speed)
- export to indexed color pngs

I will probably rewrite this in Rust or another language for speed later
