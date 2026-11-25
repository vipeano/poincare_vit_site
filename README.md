# Poincaré ViT project website

Static site, no build step: `index.html` (project page and full validation galleries) and `explore.html`
(interactive explorer of real network outputs).  Served as is by GitHub Pages.

| path | what |
|---|---|
| `index.html` | project page; the gallery view opens at `#gallery-<task>` (instance, semantic, binary, completion, pretraining) |
| `explore.html` | explorer: standard map (K × number of orbits) and 8 validation maps (number of orbits), with ground truth |
| `gallery.json`, `g/` | all 995 validation maps of the Edmond galleries (doi:10.17617/3.NBUIL8), lossless WebP, ~245 MB |
| `x/` | explorer data: per scene an RGB atlas (orbit id, predicted instance, semantic class + chaotic bit), ground truth, metadata |
| `fig/` | figures from the paper (`overview.jpg` from Figure_1.pdf, `ood.jpg` from Figure_6_webpage.pdf, `hh_*.jpg` from Figure_6/hh_e0125_*.png) |
| `tools/build_gallery.py` | regenerates `g/` and `gallery.json` from the gallery folders at the repository root (~20 s) |
| `tools/build_explorer.py` | regenerates `x/` by running the headline networks (MLX, ~1 min; needs Metal, so not from a sandbox) |
| `.nojekyll` | tells GitHub Pages to serve the files untouched |

## Preview locally

    cd website && python3 -m http.server 8000     # then open http://localhost:8000

(`fetch` does not work from `file://`, so open it through a server.)

## Deploy on GitHub Pages

Kept as its own repository, so the ~250 MB of images stay out of the code repository:

    cd website
    git init -b main && git add . && git commit -m "Project website"
    gh repo create vipeano/poincare_vit_site --public --source . --push
    gh api -X POST repos/vipeano/poincare_vit_site/pages -f "source[branch]=main" -f "source[path]=/"

The site is at https://vipeano.github.io/poincare_vit_site/ (repository vipeano/poincare_vit_site).  Pages limits: 1 GB per site, 100 MB per file
(largest file here: `x/standard.png`, 2 MB).

## Placeholders still open

arXiv number (hero button, BibTeX).
