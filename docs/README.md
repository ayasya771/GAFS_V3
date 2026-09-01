# Browser demo

This folder is published by GitHub Pages (settings: deploy from branch,
`/docs`). It is plain HTML, CSS and JavaScript with no build step, no bundler
and no third-party dependency, so what is committed here is exactly what runs.

    index.html            the page
    assets/css/app.css    palette and layout, light and dark
    assets/js/nn.js       dense linear algebra on Float32Array
    assets/js/generator.js  the TFT generator forward pass, ported from PyTorch
    assets/js/pipeline.js   the preprocessing pipeline, ported from pandas
    assets/js/stats.js      stylized-fact estimators and risk analytics
    assets/js/charts.js     SVG charts with hover, theme-aware
    assets/js/app.js        page controller
    model/generator.bin   float32 weights, tensors concatenated
    model/generator.json  manifest: architecture, tensor offsets, fitted scalers
    model/parity.json     reference input, noise and output exported from PyTorch
    data/*.csv            the market panels the page fetches
    data/index.json       panel catalogue

`model/` and `data/` are build artifacts. Regenerate them after training with

    python scripts/build_site.py --ckpt outputs/demo/run/ckpt_final.pt

Ports drift silently, so the page measures rather than assumes: at load it
runs `model/parity.json` through its own forward pass and shows the maximum
deviation from the PyTorch output in the header. `tests/test_web_bundle.py`
makes the same comparison in CI, along with a check that the JavaScript
preprocessing reproduces the pandas feature matrix.

Serving locally needs HTTP, because `fetch` will not read the model files from
a `file://` page:

    python -m http.server -d docs 8000
