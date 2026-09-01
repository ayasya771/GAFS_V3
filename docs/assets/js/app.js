
(function () {
  'use strict';

  const state = {
    spec: null,
    model: null,
    parity: null,
    datasets: [],
    panel: null,
    panelName: '',
    built: null,
    anchors: [],
    shocks: {},
    weights: [],
    lastRun: null,
    baseline: null,
    lastCtx: null,
    validation: null,
    windowStart: 0
  };

  const $ = function (id) { return document.getElementById(id); };
  const yieldUI = function () { return new Promise(function (r) { setTimeout(r, 0); }); };
  const pct = function (v) { return (v * 100).toFixed(2) + '%'; };

  function setStatus(text, kind) {
    $('model-status').textContent = text;
    $('model-badge').className = 'badge' + (kind ? ' ' + kind : '');
  }

  function statCard(k, v, n) {
    return '<div class="stat"><div class="k">' + k + '</div><div class="v">' + v +
      '</div>' + (n ? '<div class="n">' + n + '</div>' : '') + '</div>';
  }

  function table(el, headers, rows, align) {
    let html = '<thead><tr>';
    for (let i = 0; i < headers.length; i++) html += '<th>' + headers[i] + '</th>';
    html += '</tr></thead><tbody>';
    for (let r = 0; r < rows.length; r++) {
      html += '<tr>';
      for (let c = 0; c < rows[r].length; c++) {
        const cell = rows[r][c];
        const cls = (align && align[c]) ? ' class="' + align[c] + '"' : '';
        html += '<td' + cls + '>' + cell + '</td>';
      }
      html += '</tr>';
    }
    el.innerHTML = html + '</tbody>';
  }


  async function boot() {
    try {
      setStatus('Loading weights...', 'busy');
      const loaded = await Generator.load('model');
      state.model = loaded.model;
      state.spec = loaded.model.manifest;
      state.parity = loaded.parity;
      const ok = state.parity && state.parity.maxAbs < 1e-4;
      setStatus(
        state.parity
          ? 'Model live - matches PyTorch to ' + state.parity.maxAbs.toExponential(1)
          : 'Model live',
        ok ? 'ok' : 'err'
      );

      state.datasets = await fetch('data/index.json').then(function (r) { return r.json(); });
      const sel = $('dataset');
      sel.innerHTML = state.datasets.map(function (d, i) {
        return '<option value="' + i + '">' + d.label + ' (' + d.rows + ' rows' +
          (d.trained_on ? ', model fitted on this' : ', held out') + ')</option>';
      }).join('');
      let initial = state.datasets.findIndex(function (d) { return d.trained_on; });
      if (initial < 0) initial = 0;
      sel.value = String(initial);
      await loadDataset(initial);
      renderModelTab();
    } catch (err) {
      setStatus('Load failed: ' + err.message, 'err');
      const note = document.createElement('p');
      note.className = 'empty';
      note.textContent = 'Could not start: ' + err.message +
        '. If you opened this file directly, serve the folder over HTTP instead ' +
        '(for example: python -m http.server) so the model and data can be fetched.';
      $('panel-data').prepend(note);
      throw err;
    }
  }

  async function loadDataset(index) {
    const meta = state.datasets[index];
    const text = await fetch('data/' + meta.file).then(function (r) { return r.text(); });
    adoptPanel(Pipeline.parseCSV(text), meta.label);
  }

  function adoptPanel(panel, name) {
    state.panel = panel;
    state.panelName = name;
    state.built = Pipeline.buildFeatures(panel, state.spec, {});
    state.weights = new Array(state.built.assets.length).fill(1 / state.built.assets.length);
    state.lastRun = null;
    state.baseline = null;
    state.validation = null;
    state.windowStart = 100;
    $('range-start').value = 100;
    buildAnchorOptions();
    buildShockControls();
    buildWeightControls();
    buildAssetOptions();
    renderDataTab();
    renderPipelineTab();
    $('fan-charts').innerHTML = '<p class="empty">Press Generate to draw scenarios.</p>';
    $('gen-stats').innerHTML = '';
    $('table-risk').innerHTML = '';
    $('chart-horizon').innerHTML = '';
    $('chart-paths').innerHTML = '';
    $('stress-compare').innerHTML = '<p class="empty">Store a baseline, apply shocks, then generate again.</p>';
    $('table-validate').innerHTML = '';
    $('val-stats').innerHTML = '';
    ['chart-dist', 'chart-acf', 'chart-corr'].forEach(function (id) { $(id).innerHTML = ''; });
  }


  function windowSlice() {
    const n = state.built.nRows;
    const frac = 0.1 + 0.9 * (state.windowStart / 100);
    const shown = Math.max(60, Math.min(n, Math.round(n * frac)));
    return { start: n - shown, end: n };
  }

  function renderDataTab() {
    const b = state.built, panel = state.panel;
    const win = windowSlice();
    const span = win.end - win.start;
    const step = Math.max(1, Math.ceil(span / 1200));
    $('range-label').textContent = b.dates[win.start] + ' to ' + b.dates[win.end - 1];

    const priceSeries = b.assets.map(function (a, i) {
      const vals = [];
      const base = b.aux.close[win.start * b.assets.length + i];
      for (let r = win.start; r < win.end; r += step) {
        vals.push(b.aux.close[r * b.assets.length + i] / base * 100);
      }
      return { name: a, values: vals };
    });
    const labels = [];
    for (let r = win.start; r < win.end; r += step) labels.push(b.dates[r]);
    Charts.line($('chart-prices'), {
      series: priceSeries, height: 270, xTickLabels: labels,
      yLabel: 'index (start = 100)', title: 'Price history'
    });

    const macroSeries = b.macro.map(function (m, i) {
      const col = panel.columns[m];
      const vals = [];
      for (let r = win.start; r < win.end; r += step) vals.push(col[b.rowIndex[r] + 1]);
      return { name: m, values: vals };
    });

    const zSeries = macroSeries.map(function (s) {
      const m = Stats.moments(s.values);
      return { name: s.name + ' (z)', values: s.values.map(function (v) { return (v - m.mean) / (m.std || 1); }) };
    });
    Charts.line($('chart-macro'), {
      series: zSeries, height: 270, xTickLabels: labels,
      yLabel: 'standard deviations', title: 'Macro state'
    });

    const spanYears = (b.nRows / 252).toFixed(1);
    $('data-stats').innerHTML =
      statCard('Panel', state.panelName, panel.length + ' raw rows') +
      statCard('Model-ready rows', b.nRows.toLocaleString(), spanYears + ' years of trading days') +
      statCard('Assets', b.assets.length, b.assets.join(', ')) +
      statCard('Macro inputs', b.macro.length, b.macro.join(', ')) +
      statCard('Bad ticks removed', b.assets.reduce(function (acc, a) {
        return acc + b.raw.flagged[a].length; }, 0), 'MAD filter, whole panel');

    const rows = b.assets.map(function (a) {
      const rets = Array.from(b.raw.returns[a]);
      const m = Stats.moments(rets);
      const w = Stats.toWindows(rets, state.spec.arch.horizon);
      return [
        a,
        pct(m.annVol),
        m.skew.toFixed(3),
        m.kurtosis.toFixed(2),
        Stats.hillAlpha(rets).toFixed(2),
        Stats.leverage(w.data, w.count, w.windowLen).toFixed(4),
        Stats.windowACF(w.data, w.count, w.windowLen, 5, true)
          .reduce(function (x, y) { return x + y; }, 0).toFixed(4)
      ];
    });
    table($('table-data-stats'),
      ['Series', 'Annualised vol', 'Skew', 'Excess kurtosis', 'Hill alpha',
       'Leverage corr', 'Sum ACF|r| lags 1-5'], rows);
  }


  function buildAssetOptions() {
    $('pipe-asset').innerHTML = state.built.assets.map(function (a) {
      return '<option value="' + a + '">' + a + '</option>';
    }).join('');
  }


  function injectedSeries(asset, mode) {
    const raw = Float64Array.from(state.panel.columns[asset + '_close']);
    const injected = { spikes: [], crashStart: -1 };
    if (mode === 'spike' || mode === 'both') {
      const rng = NN.rng(99);
      for (let i = 0; i < 12; i++) {
        const idx = 120 + Math.floor(rng.uniform() * (raw.length - 260));
        const dir = rng.uniform() < 0.5 ? -1 : 1;
        raw[idx] *= 1 + dir * (0.28 + rng.uniform() * 0.5);
        injected.spikes.push(idx);
      }
    }
    if (mode === 'crash' || mode === 'both') {
      const at = Math.floor(raw.length * 0.62);
      for (let i = at; i < raw.length; i++) raw[i] *= 0.78;
      injected.crashStart = at;
    }
    return { series: raw, injected: injected };
  }

  function renderPipelineTab() {
    const b = state.built;
    const asset = $('pipe-asset').value || b.assets[0];
    const threshold = parseFloat($('mad-threshold').value);
    $('mad-threshold-label').textContent = threshold.toFixed(1);
    const mode = $('inject').value;
    const pre = state.spec.preprocess;

    const src = injectedSeries(asset, mode);
    const res = Pipeline.madClean(src.series, pre.mad_window, threshold, pre.mad_reversal_tol);

    const n = src.series.length;
    const step = Math.max(1, Math.floor(n / 1400));
    const rawVals = [], cleanVals = [], labels = [];
    for (let i = 0; i < n; i += step) {
      rawVals.push(src.series[i]);
      cleanVals.push(res.clean[i]);
      labels.push(state.panel.dates[i]);
    }
    Charts.line($('chart-mad'), {
      series: [
        { name: 'As loaded', values: rawVals, color: Charts.palette().series[1] },
        { name: 'After despiking', values: cleanVals, color: Charts.palette().series[0] }
      ],
      height: 260, xTickLabels: labels, yLabel: 'price', forceLegend: true
    });

    let caught = 0;
    for (let i = 0; i < src.injected.spikes.length; i++) {
      if (res.flagged.indexOf(src.injected.spikes[i]) >= 0) caught++;
    }

    const crashTouched = src.injected.crashStart >= 0 && res.flagged.some(function (i) {
      return Math.abs(i - src.injected.crashStart) <= 1 &&
             src.injected.spikes.indexOf(i) < 0;
    });
    let note = res.flagged.length + ' point' + (res.flagged.length === 1 ? '' : 's') + ' removed.';
    if (src.injected.spikes.length) {
      note += ' ' + caught + ' of ' + src.injected.spikes.length +
        ' injected bad ticks were caught.';
    }
    if (src.injected.crashStart >= 0) {
      note += crashTouched
        ? ' At this threshold the injected crash was itself clipped: the filter is too aggressive here.'
        : ' The injected 22% crash was left intact, which is the point: a bad tick reverts, a crash does not.';
    }
    $('mad-note').textContent = note;

    const rets = Pipeline.logReturns(res.clean);
    const vol = Pipeline.rollingStd(rets, pre.vol_window);
    const scaled = new Float64Array(rets.length).fill(NaN);
    for (let i = 1; i < rets.length; i++) {
      if (Number.isFinite(vol[i - 1])) scaled[i] = rets[i] / (vol[i - 1] + pre.eps);
    }
    const rl = [], sl = [], dl = [];
    for (let i = 0; i < rets.length; i += step) {
      rl.push(rets[i]); sl.push(scaled[i]); dl.push(state.panel.dates[i + 1]);
    }
    Charts.line($('chart-returns'), {
      series: [{ name: 'log return', values: rl }], height: 210,
      xTickLabels: dl, yLabel: 'r_t', forceLegend: true
    });
    Charts.line($('chart-scaled'), {
      series: [{ name: 'scaled return', values: sl, color: Charts.palette().series[2] }],
      height: 210, xTickLabels: dl, yLabel: 'r_t / vol_{t-1}', forceLegend: true
    });

    const scaledMoments = Stats.moments(Array.from(scaled));
    const rawMoments = Stats.moments(Array.from(rets));
    $('pipeline-stats').innerHTML =
      statCard('Forward-fill limit', pre.ffill_limit + ' rows', 'then the row is dropped') +
      statCard('MAD window', pre.mad_window, 'centered, reversal tolerance ' + pre.mad_reversal_tol) +
      statCard('Raw return kurtosis', rawMoments.kurtosis.toFixed(2), 'excess') +
      statCard('Scaled kurtosis', scaledMoments.kurtosis.toFixed(2), 'fat tails survive scaling') +
      statCard('Scaled std', scaledMoments.std.toFixed(3), 'target is near 1');

    renderFracdiff(asset);
  }

  function renderFracdiff(asset) {
    const d = parseFloat($('fracdiff-d').value);
    $('fracdiff-d-label').textContent = d.toFixed(2);
    const pre = state.spec.preprocess;
    const w = Pipeline.ffdWeights(d, pre.fracdiff_threshold);
    $('fracdiff-width').textContent = w.length + ' lags';

    const newestFirst = w.slice().reverse().slice(0, Math.min(40, w.length));
    Charts.line($('chart-ffd-weights'), {
      series: [{ name: 'weight', values: newestFirst, color: Charts.palette().series[1] }],
      height: 210, yLabel: 'w_k', xLabel: 'lag', forceLegend: true
    });

    const close = state.panel.columns[asset + '_close'];
    const logp = new Float64Array(close.length);
    for (let i = 0; i < close.length; i++) logp[i] = Math.log(close[i]);
    const ffd = Pipeline.fracDiff(logp, d, pre.fracdiff_threshold);
    const step = Math.max(1, Math.floor(ffd.length / 1200));
    const vals = [], labels = [];
    for (let i = 0; i < ffd.length; i += step) { vals.push(ffd[i]); labels.push(state.panel.dates[i]); }
    Charts.line($('chart-ffd-series'), {
      series: [{ name: 'fracdiff log price', values: vals, color: Charts.palette().series[2] }],
      height: 210, xTickLabels: labels, forceLegend: true
    });

    const clean = Stats.finite(Array.from(ffd));
    let acf = NaN;
    if (clean.length > 10) {
      acf = Stats.correlation(clean.slice(0, -1), clean.slice(1));
    }
    $('fracdiff-acf').textContent = Number.isFinite(acf) ? acf.toFixed(3) : '-';
  }


  function buildAnchorOptions() {
    const b = state.built, k = state.spec.arch.lookback, h = state.spec.arch.horizon;
    const first = k - 1;
    const last = b.nRows - 1;
    const opts = [];
    state.anchors = [];
    const stride = Math.max(1, Math.floor((last - first) / 400));
    for (let r = first; r <= last; r += stride) {
      state.anchors.push(r);
      const realised = r + h < b.nRows ? '' : ' (no realised path)';
      opts.push('<option value="' + r + '">' + b.dates[r] + realised + '</option>');
    }
    if (state.anchors[state.anchors.length - 1] !== last) {
      state.anchors.push(last);
      opts.push('<option value="' + last + '">' + b.dates[last] + ' (no realised path)</option>');
    }
    const sel = $('anchor');
    sel.innerHTML = opts.join('');

    const target = Math.max(first, b.nRows - 1 - h);
    let best = 0;
    for (let i = 0; i < state.anchors.length; i++) {
      if (Math.abs(state.anchors[i] - target) < Math.abs(state.anchors[best] - target)) best = i;
    }
    sel.selectedIndex = best;
  }

  function buildShockControls() {
    const macro = state.spec.columns.macro;
    const host = $('shock-fields');
    host.style.display = 'contents';
    host.innerHTML = macro.map(function (m) {
      const std = state.spec.preprocess.macro_stats[m].std;
      const bound = Math.max(1, Math.round(3 * std));
      const stepSize = bound >= 20 ? 1 : bound >= 5 ? 0.5 : 0.1;
      return '<div class="field"><label for="shock-' + m + '">' + m + ' shock</label>' +
        '<div class="range-row"><input type="range" id="shock-' + m + '" data-macro="' + m +
        '" min="' + (-bound) + '" max="' + bound + '" step="' + stepSize +
        '" value="0"><output id="shock-out-' + m + '">0</output></div></div>';
    }).join('');
    state.shocks = {};
    host.querySelectorAll('input[type=range]').forEach(function (input) {
      input.addEventListener('input', function () {
        const key = input.dataset.macro;
        const v = parseFloat(input.value);
        state.shocks[key] = v;
        $('shock-out-' + key).textContent = (v > 0 ? '+' : '') + v;
      });
    });
  }

  function buildWeightControls() {
    const assets = state.built.assets;
    $('weight-controls').innerHTML = assets.map(function (a, i) {
      return '<div class="field"><label for="w-' + i + '">' + a + ' weight</label>' +
        '<input type="number" id="w-' + i + '" min="0" max="1" step="0.05" value="' +
        (1 / assets.length).toFixed(2) + '" style="width:96px"></div>';
    }).join('');
    assets.forEach(function (a, i) {
      $('w-' + i).addEventListener('change', function () {
        const raw = assets.map(function (_, j) { return Math.max(0, parseFloat($('w-' + j).value) || 0); });
        const sum = raw.reduce(function (x, y) { return x + y; }, 0) || 1;
        state.weights = raw.map(function (v) { return v / sum; });
        if (state.lastRun) renderRisk();
      });
    });
  }

  async function runGeneration() {
    const btn = $('run-generate');
    btn.disabled = true;
    const bar = $('gen-progress');
    bar.hidden = false;
    try {
      const row = parseInt($('anchor').value, 10);
      const total = parseInt($('n-scenarios').value, 10);
      const seed = parseInt($('seed').value, 10) || 0;
      const spec = state.spec, b = state.built;
      const ctx0 = Pipeline.contextAt(b, spec, row);
      const cond = Pipeline.applyShocks(ctx0.cond, spec, state.shocks);

      const t0 = performance.now();
      const xHist = NN.mat(spec.arch.lookback, spec.arch.n_features, ctx0.xHist);
      const encoded = state.model.encode(xHist, NN.mat(1, cond.length, cond));
      state.lastCtx = encoded;

      const rng = NN.rng(seed);
      const chunk = 50;
      const chunks = [];
      let done = 0;
      while (done < total) {
        const size = Math.min(chunk, total - done);
        const Z = state.model.sampleNoise(size, rng);
        chunks.push(state.model.decode(encoded, Z));
        done += size;
        bar.firstElementChild.style.width = (done / total * 100) + '%';
        await yieldUI();
      }
      const cols = chunks[0].cols;
      const merged = NN.mat(total, cols);
      let offset = 0;
      for (let i = 0; i < chunks.length; i++) {
        merged.data.set(chunks[i].data, offset);
        offset += chunks[i].data.length;
      }
      const paths = Pipeline.toPricePaths(merged, spec, ctx0, spec.arch.horizon, b.assets.length);
      const elapsed = performance.now() - t0;

      state.lastRun = {
        paths: paths, ctx: ctx0, row: row, elapsed: elapsed,
        shocks: Object.assign({}, state.shocks), seed: seed, n: total
      };
      renderGeneration();
      renderModelTab();
    } catch (err) {
      $('fan-charts').innerHTML = '<p class="empty">Generation failed: ' + err.message + '</p>';
    } finally {
      bar.hidden = true;
      bar.firstElementChild.style.width = '0';
      btn.disabled = false;
    }
  }

  function percentileBands(paths, assetIdx) {
    const B = paths.nScenarios, h = paths.horizon, A = paths.nAssets;
    const steps = h + 1;
    const out = { p05: [], p10: [], p25: [], p50: [], p75: [], p90: [], p95: [] };
    const buf = new Float64Array(B);
    for (let t = 0; t < steps; t++) {
      for (let b = 0; b < B; b++) {
        buf[b] = paths.prices[(b * steps + t) * A + assetIdx] /
                 paths.prices[b * steps * A + assetIdx] * 100;
      }
      const sorted = Array.from(buf).sort(function (x, y) { return x - y; });
      out.p05.push(Stats.quantile(sorted, 0.05));
      out.p10.push(Stats.quantile(sorted, 0.10));
      out.p25.push(Stats.quantile(sorted, 0.25));
      out.p50.push(Stats.quantile(sorted, 0.50));
      out.p75.push(Stats.quantile(sorted, 0.75));
      out.p90.push(Stats.quantile(sorted, 0.90));
      out.p95.push(Stats.quantile(sorted, 0.95));
    }
    return out;
  }

  function realisedPath(row, assetIdx) {
    const b = state.built, h = state.spec.arch.horizon, A = b.assets.length;
    if (row + h >= b.nRows) return null;
    const base = b.aux.close[row * A + assetIdx];
    const vals = [];
    for (let t = 0; t <= h; t++) vals.push(b.aux.close[(row + t) * A + assetIdx] / base * 100);
    return vals;
  }

  function renderGeneration() {
    const run = state.lastRun, b = state.built;
    const host = $('fan-charts');
    host.innerHTML = '';
    const shocked = Object.keys(run.shocks).filter(function (k) { return run.shocks[k]; });

    $('gen-stats').innerHTML =
      statCard('Origin', b.dates[run.row], 'row ' + run.row + ' of ' + b.nRows) +
      statCard('Scenarios', run.n.toLocaleString(), run.paths.horizon + '-step paths') +
      statCard('Compute', (run.elapsed / 1000).toFixed(2) + ' s',
        (run.elapsed / run.n).toFixed(1) + ' ms per path, in this tab') +
      statCard('Macro shocks', shocked.length ? shocked.length + ' applied' : 'none',
        shocked.length ? shocked.map(function (k) {
          return k + ' ' + (run.shocks[k] > 0 ? '+' : '') + run.shocks[k]; }).join(', ') : 'baseline state') +
      statCard('Seed', run.seed, 'same seed reproduces the run');

    const grid = document.createElement('div');
    grid.className = 'grid-2 grid-wide';
    host.appendChild(grid);


    const cells = [];
    for (let a = 0; a < b.assets.length; a++) {
      const wrap = document.createElement('div');
      const title = document.createElement('h3');
      title.textContent = b.assets[a];
      wrap.appendChild(title);
      const chart = document.createElement('div');
      chart.className = 'chart';
      wrap.appendChild(chart);
      grid.appendChild(wrap);
      cells.push(chart);
    }

    for (let a = 0; a < b.assets.length; a++) {
      const chart = cells[a];
      const q = percentileBands(run.paths, a);
      const realised = realisedPath(run.row, a);
      Charts.fan(chart, {
        height: 250,
        bands: [
          { name: '5-95%', lo: q.p05, hi: q.p95 },
          { name: '10-90%', lo: q.p10, hi: q.p90 },
          { name: '25-75%', lo: q.p25, hi: q.p75 }
        ],
        median: q.p50,
        overlay: realised ? { name: 'Realised', values: realised } : null,
        yLabel: 'index (origin = 100)', xLabel: 'days ahead'
      });
    }
    renderRisk();
    renderSamplePaths();
    renderStress();
  }

  function renderRisk() {
    const run = state.lastRun, b = state.built;
    const rows = Stats.riskSummary(run.paths, state.weights);
    const body = rows.map(function (r, i) {
      const name = i < b.assets.length ? b.assets[i] : 'Portfolio';
      return [name, pct(r.meanRet), pct(r.p05), pct(r.p50), pct(r.p95),
        pct(r.var95), pct(r.var99), pct(r.es95), pct(r.es99),
        pct(r.medianDrawdown), pct(r.tailDrawdown)];
    });
    table($('table-risk'),
      ['Series', 'Mean', 'P05', 'Median', 'P95', 'VaR 95', 'VaR 99', 'ES 95', 'ES 99',
       'Median max DD', 'Tail max DD'], body);

    const hist = rows.map(function (r, i) {
      return {
        name: i < b.assets.length ? b.assets[i] : 'Portfolio',
        values: Array.from(r.horizonReturns)
      };
    });
    Charts.histogram($('chart-horizon'), {
      series: hist, height: 250, logY: false, bins: 44,
      xLabel: 'horizon return', yLabel: 'density'
    });
    state.lastRisk = rows;
  }

  function renderSamplePaths() {
    const run = state.lastRun, b = state.built;
    const A = b.assets.length, h = run.paths.horizon, steps = h + 1;
    const assetIdx = 0;
    const count = Math.min(24, run.paths.nScenarios);
    const stride = Math.max(1, Math.floor(run.paths.nScenarios / count));
    const series = [];
    const colors = Charts.palette();
    for (let i = 0; i < count; i++) {
      const bIdx = i * stride;
      const vals = [];
      for (let t = 0; t < steps; t++) {
        vals.push(run.paths.prices[(bIdx * steps + t) * A + assetIdx] /
                  run.paths.prices[bIdx * steps * A + assetIdx] * 100);
      }
      series.push({ name: 'path ' + (i + 1), values: vals, color: colors.seq3 || colors.ramp[2], width: 1 });
    }
    const realised = realisedPath(run.row, assetIdx);
    const median = [];
    for (let t = 0; t < steps; t++) {
      const col = [];
      for (let bIdx = 0; bIdx < run.paths.nScenarios; bIdx++) {
        col.push(run.paths.prices[(bIdx * steps + t) * A + assetIdx] /
                 run.paths.prices[bIdx * steps * A + assetIdx] * 100);
      }
      col.sort(function (x, y) { return x - y; });
      median.push(Stats.quantile(col, 0.5));
    }
    const tipSeries = [{ name: 'Median', values: median, color: colors.ramp[4] }];
    if (realised) {
      series.push({ name: 'Realised', values: realised, color: colors.ink, width: 2, dashed: true });
      tipSeries.push({ name: 'Realised', values: realised, color: colors.ink });
    }
    Charts.line($('chart-paths'), {
      series: series, height: 250, xLabel: 'days ahead',
      yLabel: b.assets[assetIdx] + ' (origin = 100)',
      legend: false, tipSeries: tipSeries
    });
    const legendNote = document.createElement('p');
    legendNote.className = 'note';
    legendNote.textContent = count + ' individual draws for ' + b.assets[assetIdx] +
      (realised ? ', with the path that actually occurred dashed in ink.' : '.');
    $('chart-paths').appendChild(legendNote);
  }

  function renderStress() {
    const host = $('stress-compare');
    if (!state.baseline || !state.lastRun) {
      host.innerHTML = '<p class="empty">Store a baseline, apply shocks, then generate again.</p>';
      return;
    }
    const base = Stats.riskSummary(state.baseline.paths, state.weights);
    const now = Stats.riskSummary(state.lastRun.paths, state.weights);
    const b = state.built;
    const rows = [];
    for (let i = 0; i < now.length; i++) {
      const name = i < b.assets.length ? b.assets[i] : 'Portfolio';
      const dVar = now[i].var95 - base[i].var95;
      const dEs = now[i].es99 - base[i].es99;
      const dDd = now[i].tailDrawdown - base[i].tailDrawdown;
      rows.push([
        name, pct(base[i].var95), pct(now[i].var95),
        (dVar >= 0 ? '+' : '') + pct(dVar),
        pct(base[i].es99), pct(now[i].es99),
        (dEs >= 0 ? '+' : '') + pct(dEs),
        (dDd >= 0 ? '+' : '') + pct(dDd)
      ]);
    }
    const shockText = Object.keys(state.lastRun.shocks)
      .filter(function (k) { return state.lastRun.shocks[k]; })
      .map(function (k) { return k + ' ' + (state.lastRun.shocks[k] > 0 ? '+' : '') + state.lastRun.shocks[k]; })
      .join(', ') || 'none';
    host.innerHTML = '<p class="note">Baseline origin ' + b.dates[state.baseline.row] +
      ', seed ' + state.baseline.seed + '. Current run shocks: ' + shockText +
      '.</p><div class="scroll-x"><table id="table-stress"></table></div>';
    table($('table-stress'),
      ['Series', 'VaR95 base', 'VaR95 now', 'Change', 'ES99 base', 'ES99 now', 'Change', 'Tail DD change'],
      rows);
  }


  async function runValidation() {
    const btn = $('run-validate');
    btn.disabled = true;
    const bar = $('val-progress');
    bar.hidden = false;
    try {
      const spec = state.spec, b = state.built;
      const k = spec.arch.lookback, h = spec.arch.horizon, A = b.assets.length;
      const nAnchors = parseInt($('n-anchors').value, 10);
      const perAnchor = parseInt($('per-anchor').value, 10);


      const testStart = Math.max(k - 1,
        Math.floor(b.nRows * (1 - spec.window.test_frac)) + k + h);
      const usable = b.nRows - 1 - testStart;
      if (usable < 5) throw new Error('Panel too short for a held-out comparison.');

      const rows = [];
      for (let i = 0; i < nAnchors; i++) {
        rows.push(testStart + Math.round(i * usable / Math.max(1, nAnchors - 1)));
      }

      const rng = NN.rng(20240 + nAnchors);
      const generated = new Float64Array(rows.length * perAnchor * h * A);
      let filled = 0;
      const t0 = performance.now();
      for (let i = 0; i < rows.length; i++) {
        const ctx = Pipeline.contextAt(b, spec, rows[i]);
        const xHist = NN.mat(k, spec.arch.n_features, ctx.xHist);
        const encoded = state.model.encode(xHist, NN.mat(1, ctx.cond.length, ctx.cond));
        const Z = state.model.sampleNoise(perAnchor, rng);
        const out = state.model.decode(encoded, Z);
        const paths = Pipeline.toPricePaths(out, spec, ctx, h, A);
        generated.set(paths.returns, filled);
        filled += paths.returns.length;
        bar.firstElementChild.style.width = ((i + 1) / rows.length * 100) + '%';
        if (i % 2 === 0) await yieldUI();
      }
      const elapsed = performance.now() - t0;
      const nWindows = rows.length * perAnchor;


      const realStartRow = b.rowIndex[testStart];
      const results = [];
      const realSeries = [], fakeSeries = [];
      for (let a = 0; a < A; a++) {
        const realAll = Array.from(b.raw.returns[b.assets[a]]).slice(realStartRow);
        const rw = Stats.toWindows(realAll, h);
        const fake = new Float64Array(nWindows * h);
        for (let w = 0; w < nWindows; w++) {
          for (let t = 0; t < h; t++) fake[w * h + t] = generated[(w * h + t) * A + a];
        }
        const realFlat = Array.from(rw.data.subarray(0, rw.count * h));
        const fakeFlat = Array.from(fake);
        results.push({
          asset: b.assets[a],
          real: {
            m: Stats.moments(realFlat),
            acfAbs: Stats.windowACF(rw.data, rw.count, h, 12, true),
            acfRet: Stats.windowACF(rw.data, rw.count, h, 12, false),
            lev: Stats.leverage(rw.data, rw.count, h),
            hill: Stats.hillAlpha(realFlat)
          },
          fake: {
            m: Stats.moments(fakeFlat),
            acfAbs: Stats.windowACF(fake, nWindows, h, 12, true),
            acfRet: Stats.windowACF(fake, nWindows, h, 12, false),
            lev: Stats.leverage(fake, nWindows, h),
            hill: Stats.hillAlpha(fakeFlat)
          },
          wasserstein: Stats.wasserstein(realFlat, fakeFlat),
          realFlat: realFlat,
          fakeFlat: fakeFlat
        });
        realSeries.push(realFlat);
        fakeSeries.push(fakeFlat);
      }

      const realCorr = Stats.correlationMatrix(realSeries);
      const fakeCorr = Stats.correlationMatrix(fakeSeries);
      state.validation = {
        results: results, realCorr: realCorr, fakeCorr: fakeCorr,
        nWindows: nWindows, anchors: rows.length, elapsed: elapsed,
        dist: Stats.matrixDistance(realCorr, fakeCorr),
        from: b.dates[testStart]
      };
      renderValidation();
    } catch (err) {
      $('table-validate').innerHTML = '';
      $('val-stats').innerHTML = '<p class="empty">Validation failed: ' + err.message + '</p>';
    } finally {
      bar.hidden = true;
      bar.firstElementChild.style.width = '0';
      btn.disabled = false;
    }
  }

  function renderValidation() {
    const v = state.validation;
    const meanAbs = function (arr) {
      let s = 0, n = 0;
      for (let i = 0; i < arr.length; i++) {
        if (Number.isFinite(arr[i])) { s += Math.abs(arr[i]); n++; }
      }
      return n ? s / n : NaN;
    };

    $('val-stats').innerHTML =
      statCard('Held-out origins', v.anchors, 'from ' + v.from) +
      statCard('Generated windows', v.nWindows.toLocaleString(), 'evaluated in-browser') +
      statCard('Compute', (v.elapsed / 1000).toFixed(1) + ' s', 'this tab, single thread') +
      statCard('Correlation distance', v.dist.frobenius.toFixed(3),
        'mean absolute entry ' + v.dist.meanAbs.toFixed(3));

    const rows = [];
    for (let i = 0; i < v.results.length; i++) {
      const r = v.results[i];
      rows.push([r.asset, 'Annualised volatility', pct(r.real.m.annVol), pct(r.fake.m.annVol)]);
      rows.push([r.asset, 'Excess kurtosis', r.real.m.kurtosis.toFixed(2), r.fake.m.kurtosis.toFixed(2)]);
      rows.push([r.asset, 'Skewness', r.real.m.skew.toFixed(3), r.fake.m.skew.toFixed(3)]);
      rows.push([r.asset, 'Hill tail exponent', r.real.hill.toFixed(2), r.fake.hill.toFixed(2)]);
      rows.push([r.asset, 'Leverage corr(r, |r+l|)', r.real.lev.toFixed(4), r.fake.lev.toFixed(4)]);
      rows.push([r.asset, 'Mean |ACF| of returns', meanAbs(r.real.acfRet).toFixed(4), meanAbs(r.fake.acfRet).toFixed(4)]);
      rows.push([r.asset, 'ACF |r| lags 1-5', meanAbs(r.real.acfAbs.slice(0, 5)).toFixed(4), meanAbs(r.fake.acfAbs.slice(0, 5)).toFixed(4)]);
      rows.push([r.asset, 'Wasserstein distance', '-', r.wasserstein.toExponential(2)]);
    }
    table($('table-validate'), ['Series', 'Metric', 'Real', 'Generated'], rows);

    const first = v.results[0];
    Charts.histogram($('chart-dist'), {
      series: [
        { name: 'Real', values: first.realFlat, color: Charts.palette().series[0] },
        { name: 'Generated', values: first.fakeFlat, color: Charts.palette().series[1] }
      ],
      height: 260, logY: true, xLabel: first.asset + ' log return', yLabel: 'density'
    });
    Charts.acf($('chart-acf'), {
      series: [
        { name: 'Real', values: first.real.acfAbs, color: Charts.palette().series[0] },
        { name: 'Generated', values: first.fake.acfAbs, color: Charts.palette().series[1] }
      ],
      height: 260, xLabel: 'lag (days)', yLabel: 'ACF of |returns|'
    });
    Charts.heatmaps($('chart-corr'), {
      labels: state.built.assets,
      matrices: [
        { title: 'Real', values: v.realCorr },
        { title: 'Generated', values: v.fakeCorr }
      ]
    });
  }


  function renderModelTab() {
    const spec = state.spec, A = spec.arch;
    const p = state.parity;
    $('model-stats').innerHTML =
      statCard('Parameters', spec.param_count.toLocaleString(), 'float32, ' +
        (spec.param_count * 4 / 1048576).toFixed(2) + ' MB downloaded once') +
      statCard('Training steps', spec.trained_steps.toLocaleString(), 'generator updates') +
      statCard('Context', A.lookback + ' -> ' + A.horizon, 'days of history to days ahead') +
      statCard('Hidden width', A.hidden, A.heads + ' attention heads, z dim ' + A.z_dim) +
      statCard('Parity with PyTorch', p ? p.maxAbs.toExponential(1) : 'n/a',
        p ? 'max abs deviation over ' + p.n + ' outputs' : 'reference not loaded');

    const items = [
      ['Variable selection', 'Learns which of the ' + A.n_features +
        ' inputs matter at each step, conditioned on the macro state.',
        A.n_features + ' variables'],
      ['Gated residual networks', 'Let a simple linear mapping bypass the nonlinearity when that is enough, which is what keeps training stable.', 'throughout'],
      ['Encoder LSTM', 'Reads the ' + A.lookback + '-step history, initialised from the macro context.', A.hidden + ' units'],
      ['Decoder LSTM', 'Driven by fresh Gaussian noise at every step: this is where path randomness enters.', 'z dim ' + A.z_dim],
      ['Causal attention', 'Maps history and the decoder state onto the projection horizon, masked so step i never sees step i+1.', A.heads + ' heads'],
      ['Output head', 'Noise is concatenated again before projection, so it reaches the output directly.', A.n_assets + ' assets']
    ];
    $('arch-list').innerHTML = items.map(function (it) {
      return '<li><span class="name">' + it[0] + '</span><span class="desc">' + it[1] +
        '</span><span class="count">' + it[2] + '</span></li>';
    }).join('');

    const t = spec.training;
    table($('table-hyper'),
      ['Hyperparameter', 'Value', 'Role'],
      [
        ['Critic steps per generator step', t.n_critic, 'keeps the Wasserstein estimate tight'],
        ['Gradient penalty weight', t.lambda_gp, 'enforces the 1-Lipschitz constraint'],
        ['NT-Xent temperature', t.tau, 'sharpness of the contrastive objective'],
        ['Contrastive weight', t.w_contrastive, 'critic-side latent organisation'],
        ['Uniformity weight', t.w_uniform, 'spreads generated samples over the sphere'],
        ['MMD weight', t.w_mmd, 'aligns generated and real latent distributions'],
        ['Batch size', t.batch_size, 'windows per update'],
        ['Learning rates', t.lr_g + ' / ' + t.lr_d, 'generator / critic, Adam'],
        ['EMA decay', t.ema_decay, 'weights used for sampling']
      ]);

    if (state.lastCtx && state.lastCtx.varWeights) {
      const w = state.lastCtx.varWeights;
      const names = spec.columns.features;
      const avg = new Array(names.length).fill(0);
      for (let t2 = 0; t2 < w.rows; t2++) {
        for (let j = 0; j < names.length; j++) avg[j] += w.data[t2 * w.cols + j] / w.rows;
      }
      Charts.bars($('chart-vsn'), {
        labels: names, values: avg, tipLabel: 'mean weight',
        format: function (v) { return (v * 100).toFixed(1) + '%'; }
      });
    } else {
      $('chart-vsn').innerHTML = '<p class="empty">Generate a scenario set to inspect the selection weights.</p>';
    }
  }


  function initTheme() {
    const stored = (function () {
      try { return localStorage.getItem('gafs-theme'); } catch (e) { return null; }
    })();
    if (stored) document.documentElement.setAttribute('data-theme', stored);
    $('theme-toggle').addEventListener('click', function () {
      const current = document.documentElement.getAttribute('data-theme');
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const next = current ? (current === 'dark' ? 'light' : 'dark') : (prefersDark ? 'light' : 'dark');
      document.documentElement.setAttribute('data-theme', next);
      try { localStorage.setItem('gafs-theme', next); } catch (e) {   }
      rerenderAll();
    });
  }

  function rerenderAll() {
    if (!state.built) return;
    renderTab(activeTab());
  }

  function renderTab(name) {
    if (!state.built) return;
    if (name === 'data') renderDataTab();
    else if (name === 'pipeline') renderPipelineTab();
    else if (name === 'generate') { if (state.lastRun) renderGeneration(); }
    else if (name === 'validate') { if (state.validation) renderValidation(); }
    else if (name === 'model') renderModelTab();
  }

  function activeTab() {
    const btn = document.querySelector('nav.tabs button[aria-selected="true"]');
    return btn ? btn.dataset.panel : 'data';
  }

  function initTabs() {
    const buttons = document.querySelectorAll('nav.tabs button');
    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        buttons.forEach(function (b) { b.setAttribute('aria-selected', 'false'); });
        btn.setAttribute('aria-selected', 'true');
        document.querySelectorAll('.panel').forEach(function (p) { p.classList.remove('active'); });
        $('panel-' + btn.dataset.panel).classList.add('active');
        window.scrollTo({ top: 0, behavior: 'smooth' });

        renderTab(btn.dataset.panel);
      });
    });
    let resizeTimer = null;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { renderTab(activeTab()); }, 180);
    });
  }

  function initControls() {
    $('dataset').addEventListener('change', function (ev) {
      setStatus('Loading panel...', 'busy');
      loadDataset(parseInt(ev.target.value, 10)).then(function () {
        setStatus(state.parity
          ? 'Model live - matches PyTorch to ' + state.parity.maxAbs.toExponential(1)
          : 'Model live', 'ok');
      });
    });
    $('csv-upload').addEventListener('change', function (ev) {
      const file = ev.target.files && ev.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function () {
        try {
          const panel = Pipeline.parseCSV(String(reader.result));
          const wantAssets = state.spec.columns.assets;
          const wantMacro = state.spec.columns.macro;
          if (panel.assets.length !== wantAssets.length) {
            throw new Error('this model takes exactly ' + wantAssets.length +
              ' *_close price columns, the file has ' + panel.assets.length + '.');
          }
          const missing = wantMacro.filter(function (m) { return panel.macro.indexOf(m) < 0; });
          if (missing.length) {
            throw new Error('missing macro column' + (missing.length > 1 ? 's ' : ' ') +
              missing.join(', ') + '. Expected ' + wantMacro.join(', ') + '.');
          }
          const need = state.spec.arch.lookback + state.spec.arch.horizon +
            state.spec.preprocess.vol_window + 5;
          if (panel.length < need) {
            throw new Error('at least ' + need + ' rows are needed for one full ' +
              'window, the file has ' + panel.length + '.');
          }


          const mapped = { dates: panel.dates, columns: {}, names: [],
            assets: wantAssets.slice(), macro: wantMacro.slice(),
            length: panel.length };
          panel.assets.forEach(function (sourceCol, i) {
            const target = wantAssets[i] + '_close';
            mapped.columns[target] = panel.columns[sourceCol];
            mapped.names.push(target);
          });
          wantMacro.forEach(function (m) {
            mapped.columns[m] = panel.columns[m];
            mapped.names.push(m);
          });

          adoptPanel(mapped, file.name);
          const pairs = panel.assets.map(function (sourceCol, i) {
            return sourceCol + ' as ' + wantAssets[i];
          }).join(', ');
          setStatus('Loaded ' + file.name + ' - ' + pairs, 'ok');
        } catch (err) {
          setStatus('CSV rejected: ' + err.message, 'err');
        }
      };
      reader.readAsText(file);
    });
    $('range-start').addEventListener('input', function (ev) {
      state.windowStart = parseInt(ev.target.value, 10);
      renderDataTab();
    });
    $('pipe-asset').addEventListener('change', renderPipelineTab);
    $('mad-threshold').addEventListener('input', renderPipelineTab);
    $('inject').addEventListener('change', renderPipelineTab);
    $('fracdiff-d').addEventListener('input', function () {
      renderFracdiff($('pipe-asset').value || state.built.assets[0]);
    });
    $('n-scenarios').addEventListener('input', function (ev) {
      $('n-scenarios-label').textContent = ev.target.value;
    });
    $('n-anchors').addEventListener('input', function (ev) {
      $('n-anchors-label').textContent = ev.target.value;
    });
    $('per-anchor').addEventListener('input', function (ev) {
      $('per-anchor-label').textContent = ev.target.value;
    });
    $('run-generate').addEventListener('click', runGeneration);
    $('run-validate').addEventListener('click', runValidation);
    $('reset-shocks').addEventListener('click', function () {
      state.shocks = {};
      document.querySelectorAll('#shock-fields input[type=range]').forEach(function (input) {
        input.value = 0;
        $('shock-out-' + input.dataset.macro).textContent = '0';
      });
    });
    $('store-baseline').addEventListener('click', function () {
      if (!state.lastRun) return;
      state.baseline = state.lastRun;
      renderStress();
    });
    $('clear-baseline').addEventListener('click', function () {
      state.baseline = null;
      renderStress();
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    initTheme();
    initTabs();
    initControls();
    boot();
  });
})();
