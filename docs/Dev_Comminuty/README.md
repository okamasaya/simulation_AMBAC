# AMBAC — English articles for developer communities

English versions of the Zenn posts (Japanese originals live in `articles/` and `books/`).
Images are referenced from the repository's `images/ambac/` folder with absolute GitHub raw URLs,
so these files can be pasted into dev.to, Qiita (English), Medium, Hashnode, etc.

| # | English file | Japanese original (Zenn) | Status |
|---|---|---|---|
| 1 | [01_overview.md](01_overview.md) | `articles/ambac-overview-gundam-physics.md` | see progress log |
| 2 | [02_what_we_did_and_next.md](02_what_we_did_and_next.md) | `articles/ambac-what-we-did-and-next.md` | see progress log |
| 3 | [03_side_story_bar_iphone_imu.md](03_side_story_bar_iphone_imu.md) | `articles/ambac-bangai-bar-iphone-imu.md` | draft (experiment not done) |
| 4 | [04_sf_commentary.md](04_sf_commentary.md) | `articles/ambac-sf-kosho-mook.md` (from `report/mook/mook.tex`) | see progress log |
| — | (book, 14 chapters) | `books/ambac-cpu-simulation/` | not translated yet |

## Progress log (so work can resume after a context reset)

- 2026-11-26: started. Plan: (1) new Zenn article from the mook (SF commentary), (2) English versions of articles 1–4 in this folder. The book is not translated in this pass.
- 2026-11-26: done — Zenn `articles/ambac-sf-kosho-mook.md` (new), English `01_overview.md`.
- Cross-links between posts are still `TODO` until the Zenn user name / dev-community URLs are known.

## Translation notes

- Keep numbers exactly as in the Japanese originals (they come from `data/`).
- Calibration numbers use the corrected values (docs/13): the "~2 %" bias came from the virtual robot's first-order integrator (`implicitfast`); with RK4 the per-trial scatter is 1.2–1.4 % and the bias is not significant.
- Gundam references: *Mobile Suit Gundam* (1979), *Mobile Suit Zeta Gundam* (1985), *Gundam Century* (1981). Keep the Gundam framing as motivation; the technical content is the backbone.
