# Star Merge

A drop-and-merge puzzle game in pixel-art space. Drop small bodies into the box. When two of the same kind touch, they merge into the next one up. The chain has 20 steps, from Ceres to a Black Hole, and two Black Holes collapse into a singularity.

Ceres → Pluto → Moon → Mercury → Mars → Venus → Earth → Kepler-22b → Neptune → Uranus → Saturn → Jupiter → HD 189733b → Proxima Centauri → Sun → Sirius → Arcturus → Rigel → Betelgeuse → Black Hole

## Files
- `index.html`: the whole game. It has no build step, and the music and sound effects are synthesized with Web Audio.
- `sprites/`: the pixel art. There is one 64×64 PNG per body, 4× upscaled copies in `sprites/x4/`, a name sheet in `sheet.png`, and the play-field background in `bg.png`.
- `tools/make_sprites.py`: regenerates everything in `sprites/`. It needs Python, Pillow and numpy.

```
python tools/make_sprites.py
```
