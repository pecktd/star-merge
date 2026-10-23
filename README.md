# Star Merge

A drop-and-merge puzzle game in pixel-art space. Drop small bodies into the box. When two of the same kind touch, they merge into the next one up. The chain has 20 steps, from Ceres to a Black Hole, and two Black Holes collapse into a singularity.

Ceres → Pluto → Moon → Mercury → Mars → Venus → Earth → Kepler-22b → Neptune → Uranus → Saturn → Jupiter → HD 189733b → Proxima Centauri → Sun → Sirius → Arcturus → Rigel → Betelgeuse → Black Hole

In the menu, choose a difficulty (**Easy, Normal or Hard**: pieces are 1.2×, 1.4× or 1.6× the base size), a mode (**Timed** at 1, 2 or 3 minutes, or **Endless**), then press **START**. Each combination keeps its own best score.

## Files
- `index.html`: the whole game. It has no build step, and the music and sound effects are synthesized with Web Audio.
- `sprites/`: the pixel art. There is one 64×64 PNG per body, 4× upscaled copies in `sprites/x4/`, a name sheet in `sheet.png`, and the play-field background in `bg.png`.
- `firestore.rules`: security rules for the Firebase project `pkrig-games`, which handles Google sign-in (required to play), cloud-saved bests and the Top 10 leaderboards. Paste them into Console → Firestore Database → Rules after changing them. Sign-in only works on authorized domains (pkrig.com and localhost).
- `tools/make_sprites.py`: regenerates everything in `sprites/`. It needs Python, Pillow and numpy.

```
python tools/make_sprites.py
```
