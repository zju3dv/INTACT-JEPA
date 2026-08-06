# INTACT Project Website

The public project website is served from this directory at
<https://zju3dv.github.io/INTACT-JEPA/>. The repository publishes the paper story,
audited figures, rendered media, and privacy-safe projected measurements; it
does not embed model weights or raw research artifacts.

Run it locally from the repository root:

```bash
python -m http.server 4173
```

Then open `http://localhost:4173/docs/`.

The site is a static build with a vendored Three.js viewer:

- `index.html` contains the long-form paper story;
- `site.css` defines the responsive editorial layout;
- `site.js` provides reading progress, section state, mobile navigation, and
  restrained scroll reveals;
- `hero-particles.js` provides the full-viewport, pointer-reactive particle
  opening without shipping latent vectors;
- `latent-viewer.js` provides the interactive 2D/3D representation viewer;
- `action-alignment.js` provides the interactive E5 intent/action viewer;
- `vendor/three/` contains the pinned Three.js runtime and its MIT notice;
- `assets/` contains versioned paper visuals, web video, and quantized latent
  geometry trajectories.

## Interactive geometry contract

The **Unified-task JEPA representation geometry** viewer compares three real
training outcomes: collapse, partial structure, and high-rank geometry. It
exposes all 7,075 available saved states rather than interpolated latent frames.
Its public labels describe only the observed outcomes: complete collapse, partial
collapse, and well-formed geometry.
The 2D and 3D views are independently PCA-initialized t-SNE projections over
fixed observations from PushT, Cube, Reacher, and TwoRoom. Post-hoc Procrustes
alignment keeps view orientation coherent without using the previous frame as
an optimizer initialization.
The default 3D mode presents a freely orbitable workspace, while the optional
2D and split modes provide the corresponding dense paper-style projection.
Each frame is robustly normalized for morphology, so cross-frame and cross-run
claims rely on the original-space metrics rather than an absolute projected
radius.

The browser receives only signed-16-bit projected coordinates and scalar
metrics. It does not receive raw latents, model weights, checkpoint paths, or
host metadata. t-SNE remains qualitative; effective rank and mean pairwise
cosine are computed in the original latent space and displayed separately.

## Research media and native data views

The 1080p hero film is rebuilt with `tools/build_website_films.py`. Its title,
method animation, E1–E5 CKA/SR sequence, result chart, and headline sequence are
rendered natively; its control wall reads synchronized four-task rollout videos
and its background particle space reads the exported measured latent projection.
The public sharing cover is rebuilt with `tools/build_social_cover.py`, which
renders native 4K PNG and 2K JPEG editions with the headline results and four
research affiliations. The 2K landscape edition is also the Open Graph and X
card image.
The high-resolution media editions are rebuilt with `tools/build_hf_media.py`.
It produces web-optimized 2560x1440 and 3840x2160 H.264 films for both the
QR-free page film and the sharing film; the latter additionally receives an
automated closing-QR decode check. The website keeps the compact 1080p source
for autoplay and exposes the 2K/4K page editions as explicit downloads.
The shared-encoder matrix is native HTML, and the interactive CKA/SR figure
reads `assets/goal-intact-alignment.json` to remain sharp and inspectable at
every viewport. The E5 intent/action viewer reads
`assets/action-alignment-e5.json`, which contains only standardized PCA-3D
coordinates and scalar metrics. It includes neither raw latents nor checkpoints.
The desktop header presents the four research affiliations beside INTACT, while
the mobile header places them in a dedicated full-width row. The footer retains
the official InSpatio and RoboParty Lab wordmarks. Each mark links to its
institution; all trademarks remain with their owners. The university marks and
the film opening and QR-enabled share edition use assets published by the
[Zhejiang University identity system](https://www.zju.edu.cn/514/listm.htm) and
the [Tsinghua University visual identity system](https://vi.tsinghua.edu.cn/gk/xxbz/xh.htm).
Its closing card links to the project page, GitHub repository, InSpatio, and
RoboParty Lab while preserving the QR quiet zone.

The permanent Community entry point lives at `community/`. The WeChat group has
more than 200 members and therefore requires a person-to-person invitation.
The page publishes Junhan's personal WeChat contact card and the public contact
email rather than a short-lived group QR code. The permanent Community-page QR
remains the recommended share target.

Before each public update:

1. verify that no private paths, checkpoints, or raw latents enter `docs/`;
2. rebuild and decode-check the hero film;
3. test the interactive views on desktop and mobile viewports;
4. confirm paper, canonical, social-card, and repository links;
5. run the manual Pages workflow and inspect the live deployment.

## GitHub Pages readiness

The repository includes a manual-only deployment workflow at
`.github/workflows/pages.yml`. It uploads `docs/` directly and does not run on
push, so website changes remain staged until an explicit deployment:

1. open **Actions > Deploy project website**;
2. run the workflow from `main`;
3. verify <https://zju3dv.github.io/INTACT-JEPA/> after the job completes.
