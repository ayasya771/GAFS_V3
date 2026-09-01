
(function (global) {
  'use strict';

  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    const header = lines[0].split(',');
    const names = header.slice(1);
    const dates = new Array(lines.length - 1);
    const cols = {};
    for (let j = 0; j < names.length; j++) cols[names[j]] = new Float64Array(lines.length - 1);
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(',');
      dates[i - 1] = parts[0];
      for (let j = 0; j < names.length; j++) {
        const v = parseFloat(parts[j + 1]);
        cols[names[j]][i - 1] = Number.isFinite(v) ? v : NaN;
      }
    }
    return {
      dates: dates,
      columns: cols,
      names: names,
      assets: names.filter(function (n) { return /_close$/.test(n); }),
      macro: names.filter(function (n) { return !/_close$/.test(n); }),
      length: dates.length
    };
  }


  function ffillLimited(series, limit) {
    const out = Float64Array.from(series);
    let run = 0;
    for (let i = 1; i < out.length; i++) {
      if (Number.isNaN(out[i])) {
        if (run < limit && !Number.isNaN(out[i - 1])) { out[i] = out[i - 1]; run++; }
      } else {
        run = 0;
      }
    }
    return out;
  }

  function median(values) {
    const a = Array.prototype.slice.call(values).filter(Number.isFinite).sort(function (x, y) { return x - y; });
    if (!a.length) return NaN;
    const mid = a.length >> 1;
    return a.length % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
  }


  function madClean(prices, window, nSigmas, reversalTol) {
    const n = prices.length;
    const w = window % 2 === 0 ? window + 1 : window;
    const half = (w - 1) >> 1;
    const minP = Math.max(5, Math.floor(w / 4));
    const logp = new Float64Array(n);
    for (let i = 0; i < n; i++) logp[i] = Math.log(prices[i]);

    const med = new Float64Array(n).fill(NaN);
    const mad = new Float64Array(n).fill(NaN);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - half), hi = Math.min(n, i + half + 1);
      if (hi - lo < minP) continue;
      const slice = logp.subarray(lo, hi);
      const m = median(slice);
      med[i] = m;
      const dev = new Float64Array(hi - lo);
      for (let j = 0; j < dev.length; j++) dev[j] = Math.abs(slice[j] - m);
      mad[i] = median(dev);
    }

    const clean = Float64Array.from(prices);
    const flagged = [];
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(med[i]) || !Number.isFinite(mad[i]) || mad[i] === 0) continue;
      const dev = Math.abs(logp[i] - med[i]);
      const z = dev / (1.4826 * mad[i]);
      if (z <= nSigmas) continue;
      if (i + 1 >= n) continue;
      const nextDev = Math.abs(logp[i + 1] - med[i]);
      if (nextDev <= reversalTol * dev) {
        clean[i] = Math.exp(med[i]);
        flagged.push(i);
      }
    }
    return { clean: clean, flagged: flagged };
  }

  function logReturns(prices) {
    const out = new Float64Array(prices.length - 1);
    for (let i = 1; i < prices.length; i++) out[i - 1] = Math.log(prices[i] / prices[i - 1]);
    return out;
  }


  function rollingStd(series, window) {
    const n = series.length;
    const out = new Float64Array(n).fill(NaN);
    let sum = 0, sumSq = 0;
    for (let i = 0; i < n; i++) {
      sum += series[i]; sumSq += series[i] * series[i];
      if (i >= window) {
        sum -= series[i - window];
        sumSq -= series[i - window] * series[i - window];
      }
      if (i >= window - 1) {
        const mean = sum / window;
        const varr = (sumSq - window * mean * mean) / (window - 1);
        out[i] = Math.sqrt(Math.max(varr, 0));
      }
    }
    return out;
  }


  function ffdWeights(d, threshold) {
    const w = [1.0];
    let k = 1;
    while (k < 10000) {
      const next = -w[w.length - 1] * (d - k + 1) / k;
      if (Math.abs(next) < threshold) break;
      w.push(next);
      k++;
    }
    return w.reverse();
  }

  function fracDiff(series, d, threshold) {
    const w = ffdWeights(d, threshold);
    const width = w.length;
    const out = new Float64Array(series.length).fill(NaN);
    for (let i = width - 1; i < series.length; i++) {
      let acc = 0, ok = true;
      for (let j = 0; j < width; j++) {
        const v = series[i - width + 1 + j];
        if (!Number.isFinite(v)) { ok = false; break; }
        acc += w[j] * v;
      }
      if (ok) out[i] = acc;
    }
    return out;
  }


  function buildFeatures(panel, spec, options) {
    const opts = options || {};
    const assets = spec.columns.assets;
    const macro = spec.columns.macro;
    const pre = spec.preprocess;
    const madCfg = {
      window: opts.madWindow || pre.mad_window,
      threshold: opts.madThreshold === undefined ? pre.mad_threshold : opts.madThreshold,
      reversalTol: pre.mad_reversal_tol
    };
    const volWindow = pre.vol_window, eps = pre.eps;

    const cleaned = {}, flagged = {};
    for (let a = 0; a < assets.length; a++) {
      const raw = ffillLimited(panel.columns[assets[a] + '_close'], pre.ffill_limit);
      const res = madClean(raw, madCfg.window, madCfg.threshold, madCfg.reversalTol);
      cleaned[assets[a]] = res.clean;
      flagged[assets[a]] = res.flagged;
    }

    const rets = {}, vol = {}, scaled = {};
    for (let a = 0; a < assets.length; a++) {
      const name = assets[a];
      rets[name] = logReturns(cleaned[name]);
      vol[name] = rollingStd(rets[name], volWindow);
      const sc = new Float64Array(rets[name].length).fill(NaN);
      if (pre.scaling === 'vol') {
        for (let i = 1; i < sc.length; i++) {
          const div = vol[name][i - 1];
          if (Number.isFinite(div)) sc[i] = rets[name][i] / (div + eps);
        }
      } else {
        const st = pre.asset_stats[name];
        for (let i = 0; i < sc.length; i++) sc[i] = (rets[name][i] - st.mean) / (st.std + eps);
      }
      scaled[name] = sc;
    }


    const macroScaled = {};
    for (let m = 0; m < macro.length; m++) {
      const name = macro[m];
      const st = pre.macro_stats[name];
      const src = panel.columns[name];
      const out = new Float64Array(src.length - 1);
      for (let i = 0; i < out.length; i++) out[i] = (src[i + 1] - st.mean) / (st.std + eps);
      macroScaled[name] = out;
    }

    const total = panel.length - 1;
    const keep = [];
    for (let i = 0; i < total; i++) {
      let ok = true;
      for (let a = 0; a < assets.length && ok; a++) ok = Number.isFinite(scaled[assets[a]][i]);
      for (let m = 0; m < macro.length && ok; m++) ok = Number.isFinite(macroScaled[macro[m]][i]);
      if (ok) keep.push(i);
    }
    if (!keep.length) throw new Error('No usable rows after preprocessing.');

    const featureCols = spec.columns.features;
    const F = featureCols.length;
    const features = new Float32Array(keep.length * F);
    const auxVol = new Float64Array(keep.length * assets.length);
    const auxClose = new Float64Array(keep.length * assets.length);
    const auxRet = new Float64Array(keep.length * assets.length);
    const dates = new Array(keep.length);

    for (let r = 0; r < keep.length; r++) {
      const i = keep[r];
      dates[r] = panel.dates[i + 1];
      for (let f = 0; f < F; f++) {
        const name = featureCols[f];
        features[r * F + f] = assets.indexOf(name) >= 0 ? scaled[name][i] : macroScaled[name][i];
      }
      for (let a = 0; a < assets.length; a++) {
        const name = assets[a];
        auxVol[r * assets.length + a] = vol[name][i];
        auxRet[r * assets.length + a] = rets[name][i];
        auxClose[r * assets.length + a] = cleaned[name][i + 1];
      }
    }

    return {
      dates: dates,
      features: features,
      nFeatures: F,
      nRows: keep.length,
      assets: assets,
      macro: macro,
      aux: { vol: auxVol, close: auxClose, ret: auxRet },
      raw: { cleaned: cleaned, flagged: flagged, returns: rets, vol: vol },
      rowIndex: keep
    };
  }


  function contextAt(built, spec, row) {
    const k = spec.arch.lookback, F = built.nFeatures;
    if (row < k - 1 || row >= built.nRows) {
      throw new Error('Anchor row ' + row + ' has less than ' + k + ' rows of history.');
    }
    const xHist = new Float32Array(k * F);
    xHist.set(built.features.subarray((row - k + 1) * F, (row + 1) * F));
    const macroNames = spec.columns.macro;
    const cond = new Float32Array(macroNames.length);
    for (let m = 0; m < macroNames.length; m++) {
      cond[m] = built.features[row * F + spec.columns.features.indexOf(macroNames[m])];
    }
    const nA = built.assets.length;
    return {
      xHist: xHist,
      cond: cond,
      lastVol: built.aux.vol.slice(row * nA, (row + 1) * nA),
      lastClose: built.aux.close.slice(row * nA, (row + 1) * nA),
      date: built.dates[row],
      row: row
    };
  }


  function applyShocks(cond, spec, shocks) {
    const out = Float32Array.from(cond);
    const names = spec.columns.macro;
    for (const key in shocks) {
      const idx = names.indexOf(key);
      if (idx < 0) throw new Error('Unknown macro series: ' + key);
      const std = spec.preprocess.macro_stats[key].std || 1;
      out[idx] += shocks[key] / std;
    }
    return out;
  }


  function toPricePaths(scaledOut, spec, ctx, horizon, nAssets) {
    const B = scaledOut.rows;
    const eps = spec.preprocess.eps;
    const returns = new Float64Array(B * horizon * nAssets);
    const prices = new Float64Array(B * (horizon + 1) * nAssets);
    const volMode = spec.preprocess.scaling === 'vol';
    for (let b = 0; b < B; b++) {
      for (let a = 0; a < nAssets; a++) {
        prices[(b * (horizon + 1)) * nAssets + a] = ctx.lastClose[a];
      }
      let cum = new Float64Array(nAssets);
      for (let t = 0; t < horizon; t++) {
        for (let a = 0; a < nAssets; a++) {
          const s = scaledOut.data[b * scaledOut.cols + t * nAssets + a];
          const r = volMode
            ? s * (ctx.lastVol[a] + eps)
            : s * (spec.preprocess.asset_stats[spec.columns.assets[a]].std + eps) +
              spec.preprocess.asset_stats[spec.columns.assets[a]].mean;
          returns[(b * horizon + t) * nAssets + a] = r;
          cum[a] += r;
          prices[(b * (horizon + 1) + t + 1) * nAssets + a] = ctx.lastClose[a] * Math.exp(cum[a]);
        }
      }
    }
    return { returns: returns, prices: prices, nScenarios: B, horizon: horizon, nAssets: nAssets };
  }

  global.Pipeline = {
    parseCSV: parseCSV, ffillLimited: ffillLimited, madClean: madClean,
    logReturns: logReturns, rollingStd: rollingStd, ffdWeights: ffdWeights,
    fracDiff: fracDiff, buildFeatures: buildFeatures, contextAt: contextAt,
    applyShocks: applyShocks, toPricePaths: toPricePaths, median: median
  };
})(typeof window !== 'undefined' ? window : globalThis);
