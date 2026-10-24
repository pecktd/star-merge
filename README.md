# Star Merge

A drop-and-merge puzzle game in pixel-art space. Drop small bodies into the box. When two of the same kind touch, they merge into the next one up. The chain has 20 steps, from Ceres to a Black Hole, and two Black Holes collapse into a singularity.

Ceres → Pluto → Moon → Mercury → Mars → Venus → Earth → Kepler-22b → Neptune → Uranus → Saturn → Jupiter → HD 189733b → Proxima Centauri → Sun → Sirius → Arcturus → Rigel → Betelgeuse → Black Hole

In the menu, choose a difficulty (**Easy, Normal or Hard**: pieces are 1.2×, 1.4× or 1.6× the base size), a mode (**Timed** at 1, 2 or 3 minutes, or **Endless**), then press **START**. Each combination keeps its own best score.

**BATTLE** is an online match for 2 players. Each player enters a name, or uses their Google account name. One player presses **Create room** and shares the 4-letter code, and the other types it in and presses **Join**. Every merge from Mercury up drops asteroids into the opponent's box (bigger bodies and combos send more; a Singularity sends 8). After a short warning the asteroids fall. Asteroids never merge: to smash one, make a merge with a body that is touching it (+10 points). Your own merges cancel asteroids that are still on their way to you. The first player to overflow loses. A mini view of the opponent's box sits in the HUD. Battles can't be paused, aren't on the Top 10, and both players play at the host's difficulty. The connection is peer-to-peer (WebRTC through the free PeerJS broker), so it needs no server and no Firestore changes.

## Files
- `index.html`: the whole game. It has no build step, and the music and sound effects are synthesized with Web Audio.
- `sprites/`: the pixel art. There is one 64×64 PNG per body, 4× upscaled copies in `sprites/x4/`, a name sheet in `sheet.png`, and the play-field background in `bg.png`.
- `firestore.rules`: security rules for the Firebase project `pkrig-games`, which handles optional Google sign-in, guest players (a typed name plus anonymous sign-in), cloud-saved bests and the Top 10 leaderboards. Paste them into Console → Firestore Database → Rules after changing them. Sign-in only works on authorized domains (pkrig.com and localhost).
- `tools/make_sprites.py`: regenerates everything in `sprites/`. It needs Python, Pillow and numpy.

```
python tools/make_sprites.py
```
