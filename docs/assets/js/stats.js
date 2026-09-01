
(function (global) {
  'use strict';

  function finite(values) {
    const out = [];
    for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) out.push(values[i]);
    return out;
  }

  function mean(v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i];
    return s / v.length;
  }

  function moments(values) {
    const v = finite(values);
    const n = v.length;
    if (n < 4) return { n: n, mean: NaN, std: NaN, skew: NaN, kurtosis: NaN, min: NaN, max: NaN, annVol: NaN };
    const m = mean(v);
    let m2 = 0, m3 = 0, m4 = 0, min = Infinity, max = -Infinity;
    for (let i = 0; i < n; i++) {
      const d = v[i] - m;
      const d2 = d * d;
      m2 += d2; m3 += d2 * d; m4 += d2 * d2;
      if (v[i] < min) min = v[i];
      if (v[i] > max) max = v[i];
    }
    m2 /= n; m3 /= n; m4 /= n;
    const sd = Math.sqrt(m2);
    return {
      n: n,
      mean: m,
      std: Math.sqrt(m2 * n / (n - 1)),
      annVol: Math.sqrt(m2 * n / (n - 1)) * Math.sqrt(252),
      skew: m3 / Math.pow(m2, 1.5),
      kurtosis: m4 / (m2 * m2) - 3,
      min: min,
      max: max,
      sd: sd
    };
  }


  function windowACF(data, nWindows, windowLen, maxLag, absolute) {
    const lags = Math.min(maxLag, windowLen - 2);
    const out = new Float64Array(lags);
    const counts = new Float64Array(lags);
    const buf = new Float64Array(windowLen);
    for (let w = 0; w < nWindows; w++) {
      const off = w * windowLen;
      let m = 0;
      for (let i = 0; i < windowLen; i++) {
        buf[i] = absolute ? Math.abs(data[off + i]) : data[off + i];
        m += buf[i];
      }
      m /= windowLen;
      let denom = 0;
      for (let i = 0; i < windowLen; i++) {
        buf[i] -= m;
        denom += buf[i] * buf[i];
      }
      if (denom < 1e-12) continue;
      for (let lag = 1; lag <= lags; lag++) {
        let num = 0;
        for (let i = 0; i + lag < windowLen; i++) num += buf[i] * buf[i + lag];
        out[lag - 1] += num / denom;
        counts[lag - 1] += 1;
      }
    }
    for (let i = 0; i < lags; i++) out[i] = counts[i] ? out[i] / counts[i] : NaN;
    return Array.from(out);
  }


  function leverage(data, nWindows, windowLen, lags) {
    const use = lags || [1, 2, 3, 4, 5];
    const vals = [];
    for (let li = 0; li < use.length; li++) {
      const lag = use[li];
      if (windowLen <= lag + 1) continue;
      const xs = [], ys = [];
      for (let w = 0; w < nWindows; w++) {
        const off = w * windowLen;
        for (let i = 0; i + lag < windowLen; i++) {
          xs.push(data[off + i]);
          ys.push(Math.abs(data[off + i + lag]));
        }
      }
      const c = correlation(xs, ys);
      if (Number.isFinite(c)) vals.push(c);
    }
    return vals.length ? mean(vals) : NaN;
  }

  function correlation(a, b) {
    const n = Math.min(a.length, b.length);
    if (n < 2) return NaN;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) {
      const x = a[i] - ma, y = b[i] - mb;
      num += x * y; da += x * x; db += y * y;
    }
    const denom = Math.sqrt(da * db);
    return denom < 1e-15 ? NaN : num / denom;
  }


  function hillAlpha(values, tailFrac) {
    const a = [];
    for (let i = 0; i < values.length; i++) {
      const v = Math.abs(values[i]);
      if (Number.isFinite(v) && v > 0) a.push(v);
    }
    if (a.length < 50) return NaN;
    a.sort(function (x, y) { return y - x; });
    let k = Math.max(10, Math.floor((tailFrac || 0.05) * a.length));
    k = Math.min(k, a.length - 1);
    let sum = 0;
    for (let i = 0; i < k; i++) sum += Math.log(a[i] / a[k]);
    return sum > 0 ? k / sum : NaN;
  }

  function quantile(sorted, p) {
    if (!sorted.length) return NaN;
    const idx = (sorted.length - 1) * p;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }


  function wasserstein(a, b) {
    const x = finite(a).slice().sort(function (p, q) { return p - q; });
    const y = finite(b).slice().sort(function (p, q) { return p - q; });
    if (!x.length || !y.length) return NaN;
    const merged = x.concat(y).sort(function (p, q) { return p - q; });
    let total = 0;
    for (let i = 0; i < merged.length - 1; i++) {
      const width = merged[i + 1] - merged[i];
      if (width === 0) continue;
      total += width * Math.abs(cdf(x, merged[i]) - cdf(y, merged[i]));
    }
    return total;
  }

  function cdf(sorted, v) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo / sorted.length;
  }

  function correlationMatrix(series) {
    const n = series.length;
    const out = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) row.push(i === j ? 1 : correlation(series[i], series[j]));
      out.push(row);
    }
    return out;
  }

  function matrixDistance(a, b) {
    let fro = 0, absSum = 0, count = 0;
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < a.length; j++) {
        const d = a[i][j] - b[i][j];
        fro += d * d; absSum += Math.abs(d); count++;
      }
    }
    return { frobenius: Math.sqrt(fro), meanAbs: absSum / count };
  }


  function riskSummary(paths, weights) {
    const B = paths.nScenarios, h = paths.horizon, A = paths.nAssets;
    const w = weights || new Array(A).fill(1 / A);
    const rows = [];
    const series = [];

    for (let a = 0; a <= A; a++) {
      const horizonRet = new Float64Array(B);
      const drawdown = new Float64Array(B);
      for (let b = 0; b < B; b++) {
        let peak = 1, worst = 0, last = 1;
        for (let t = 0; t <= h; t++) {
          let value = 0;
          if (a < A) {
            value = paths.prices[(b * (h + 1) + t) * A + a] / paths.prices[b * (h + 1) * A + a];
          } else {
            for (let j = 0; j < A; j++) {
              value += w[j] * paths.prices[(b * (h + 1) + t) * A + j] /
                       paths.prices[b * (h + 1) * A + j];
            }
          }
          if (value > peak) peak = value;
          const dd = value / peak - 1;
          if (dd < worst) worst = dd;
          last = value;
        }
        horizonRet[b] = last - 1;
        drawdown[b] = worst;
      }
      const sortedRet = Array.from(horizonRet).sort(function (p, q) { return p - q; });
      const sortedDD = Array.from(drawdown).sort(function (p, q) { return p - q; });
      const q05 = quantile(sortedRet, 0.05), q01 = quantile(sortedRet, 0.01);
      const tail05 = sortedRet.filter(function (v) { return v <= q05; });
      const tail01 = sortedRet.filter(function (v) { return v <= q01; });
      rows.push({
        name: a < A ? null : 'Portfolio',
        index: a,
        meanRet: mean(sortedRet),
        p05: q05,
        p50: quantile(sortedRet, 0.5),
        p95: quantile(sortedRet, 0.95),
        var95: -q05,
        var99: -q01,
        es95: tail05.length ? -mean(tail05) : NaN,
        es99: tail01.length ? -mean(tail01) : NaN,
        medianDrawdown: quantile(sortedDD, 0.5),
        tailDrawdown: quantile(sortedDD, 0.05),
        horizonReturns: horizonRet
      });
      series.push(horizonRet);
    }
    return rows;
  }


  function toWindows(series, windowLen, stride) {
    const step = stride || Math.max(1, Math.floor(windowLen / 2));
    const count = Math.floor((series.length - windowLen) / step) + 1;
    const out = new Float64Array(Math.max(count, 0) * windowLen);
    for (let w = 0; w < count; w++) {
      for (let i = 0; i < windowLen; i++) out[w * windowLen + i] = series[w * step + i];
    }
    return { data: out, count: Math.max(count, 0), windowLen: windowLen };
  }

  global.Stats = {
    moments: moments, windowACF: windowACF, leverage: leverage,
    correlation: correlation, correlationMatrix: correlationMatrix,
    matrixDistance: matrixDistance, hillAlpha: hillAlpha, quantile: quantile,
    wasserstein: wasserstein, riskSummary: riskSummary, toWindows: toWindows,
    mean: mean, finite: finite
  };
})(typeof window !== 'undefined' ? window : globalThis);
