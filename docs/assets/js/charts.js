
(function (global) {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const PAD = { top: 18, right: 16, bottom: 34, left: 54 };

  function css(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function palette() {
    return {
      surface: css('--surface-1', '#fcfcfb'),
      ink: css('--text-primary', '#0b0b0b'),
      secondary: css('--text-secondary', '#52514e'),
      muted: css('--text-muted', '#898781'),
      grid: css('--grid', '#e1e0d9'),
      axis: css('--axis', '#c3c2b7'),
      series: [css('--series-1', '#2a78d6'), css('--series-2', '#eb6834'), css('--series-3', '#1baf7a')],
      ramp: [css('--seq-1', '#cde2fb'), css('--seq-2', '#9ec5f4'), css('--seq-3', '#5598e7'),
             css('--seq-4', '#256abf'), css('--seq-5', '#104281')],
      divLow: css('--div-low', '#2a78d6'),
      divMid: css('--div-mid', '#f0efec'),
      divHigh: css('--div-high', '#d03b3b')
    };
  }

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function niceTicks(min, max, count) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
    if (min === max) { min -= 0.5; max += 0.5; }
    const span = max - min;
    const raw = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
      ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    }
    return ticks;
  }

  function fmt(v, digits) {
    if (!Number.isFinite(v)) return '-';
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(1);
    return v.toFixed(digits === undefined ? (a >= 100 ? 0 : a >= 1 ? 2 : 4) : digits);
  }


  function tickDecimals(ticks) {
    let step = Infinity;
    for (let i = 1; i < ticks.length; i++) {
      const d = Math.abs(ticks[i] - ticks[i - 1]);
      if (d > 0 && d < step) step = d;
    }
    if (!Number.isFinite(step)) step = Math.abs(ticks[0]) || 1;
    if (step >= 1e6 || step < 1e-4) return null;
    return Math.max(0, Math.min(6, Math.ceil(-Math.log10(step)) + 1));
  }

  function axisFormatter(ticks, override) {
    if (override) return override;
    const dec = tickDecimals(ticks);
    if (dec === null) return function (v) { return fmt(v); };
    return function (v) { return (Object.is(v, -0) ? 0 : v).toFixed(dec); };
  }

  function tickObjects(labels, n, maxTicks) {
    const count = Math.max(2, Math.min(maxTicks || 6, labels.length));
    const out = [];
    const seen = {};
    for (let i = 0; i < count; i++) {
      const idx = Math.round(i * (n - 1) / Math.max(1, count - 1));
      if (seen[idx]) continue;
      seen[idx] = true;
      out.push({ value: idx, label: labels[idx] });
    }
    return out;
  }


  function frame(container, opts) {
    container.innerHTML = '';
    const p = palette();

    const measured = Math.round(container.clientWidth || 0);
    const width = opts.width || (measured > 260 ? measured : 720);
    const height = opts.height || 300;
    const pad = Object.assign({}, PAD, opts.pad || {});
    const svg = el('svg', {
      viewBox: '0 0 ' + width + ' ' + height,
      preserveAspectRatio: 'xMidYMid meet',
      class: 'chart-svg',
      role: 'img'
    }, container);
    if (opts.title) el('title', {}, svg).textContent = opts.title;

    const plot = {
      svg: svg, p: p, width: width, height: height, pad: pad,
      w: width - pad.left - pad.right,
      h: height - pad.top - pad.bottom,
      xMin: opts.xMin, xMax: opts.xMax, yMin: opts.yMin, yMax: opts.yMax
    };
    plot.sx = function (v) {
      return pad.left + (v - plot.xMin) / (plot.xMax - plot.xMin || 1) * plot.w;
    };
    plot.sy = function (v) {
      return pad.top + plot.h - (v - plot.yMin) / (plot.yMax - plot.yMin || 1) * plot.h;
    };

    const yTicks = opts.yTicks || niceTicks(plot.yMin, plot.yMax, 5);
    const yFmt = axisFormatter(yTicks, opts.yFormat);
    for (let i = 0; i < yTicks.length; i++) {
      const y = plot.sy(yTicks[i]);
      if (y < pad.top - 1 || y > pad.top + plot.h + 1) continue;
      el('line', { x1: pad.left, x2: pad.left + plot.w, y1: y, y2: y,
        stroke: p.grid, 'stroke-width': 0.7 }, svg);
      const t = el('text', { x: pad.left - 8, y: y + 3.5, 'text-anchor': 'end',
        fill: p.muted, 'font-size': 10 }, svg);
      t.textContent = yFmt(yTicks[i]);
    }

    const xTicks = opts.xTicks || (opts.xTickLabels
      ? tickObjects(opts.xTickLabels, opts.xCount || opts.xTickLabels.length,
                    Math.max(2, Math.min(6, Math.floor(plot.w / 96))))
      : niceTicks(plot.xMin, plot.xMax, Math.max(3, Math.min(6, Math.floor(plot.w / 90)))));
    const numericX = xTicks.length && typeof xTicks[0] !== 'object';
    const xFmt = numericX ? axisFormatter(xTicks, opts.xFormat) : null;
    for (let i = 0; i < xTicks.length; i++) {
      const value = typeof xTicks[i] === 'object' ? xTicks[i].value : xTicks[i];
      const label = typeof xTicks[i] === 'object' ? xTicks[i].label : xFmt(value);
      const x = plot.sx(value);
      if (x < pad.left - 1 || x > pad.left + plot.w + 1) continue;
      el('line', { x1: x, x2: x, y1: pad.top, y2: pad.top + plot.h,
        stroke: p.grid, 'stroke-width': 0.7 }, svg);

      const anchor = x < pad.left + 22 ? 'start' : (x > pad.left + plot.w - 22 ? 'end' : 'middle');
      const t = el('text', { x: x, y: pad.top + plot.h + 15, 'text-anchor': anchor,
        fill: p.muted, 'font-size': 10 }, svg);
      t.textContent = label;
    }
    el('line', { x1: pad.left, x2: pad.left + plot.w, y1: pad.top + plot.h,
      y2: pad.top + plot.h, stroke: p.axis, 'stroke-width': 1 }, svg);
    if (opts.yLabel) {
      const t = el('text', { x: 12, y: pad.top + plot.h / 2, fill: p.muted,
        'font-size': 10, 'text-anchor': 'middle',
        transform: 'rotate(-90 12 ' + (pad.top + plot.h / 2) + ')' }, svg);
      t.textContent = opts.yLabel;
    }
    if (opts.xLabel) {
      const t = el('text', { x: pad.left + plot.w / 2, y: height - 4,
        fill: p.muted, 'font-size': 10, 'text-anchor': 'middle' }, svg);
      t.textContent = opts.xLabel;
    }
    return plot;
  }

  function legend(container, entries) {
    const box = document.createElement('div');
    box.className = 'chart-legend';
    for (let i = 0; i < entries.length; i++) {
      const item = document.createElement('span');
      item.className = 'legend-item';
      const swatch = document.createElement('i');
      swatch.style.background = entries[i].color;
      if (entries[i].dashed) swatch.classList.add('dashed');
      item.appendChild(swatch);
      item.appendChild(document.createTextNode(entries[i].name));
      box.appendChild(item);
    }
    container.appendChild(box);
  }

  function tooltip(container) {
    let node = container.querySelector('.chart-tip');
    if (!node) {
      node = document.createElement('div');
      node.className = 'chart-tip';
      node.hidden = true;
      container.appendChild(node);
    }
    return node;
  }

  function path(points, plot) {
    let d = '';
    let pen = false;
    for (let i = 0; i < points.length; i++) {
      const y = points[i][1];
      if (!Number.isFinite(y)) { pen = false; continue; }
      const cmd = pen ? 'L' : 'M';
      d += cmd + plot.sx(points[i][0]).toFixed(2) + ' ' + plot.sy(y).toFixed(2) + ' ';
      pen = true;
    }
    return d;
  }

  function extent(values, existing) {
    let min = existing ? existing[0] : Infinity;
    let max = existing ? existing[1] : -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return [min, max];
  }

  function padRange(range, frac) {
    const span = (range[1] - range[0]) || Math.abs(range[0]) || 1;
    const pad = span * (frac === undefined ? 0.06 : frac);
    return [range[0] - pad, range[1] + pad];
  }


  function lineChart(container, opts) {
    const p = palette();
    let yRange = null;
    for (let i = 0; i < opts.series.length; i++) {
      yRange = extent(opts.series[i].values, yRange);
    }
    yRange = opts.yRange || padRange(yRange);
    const n = opts.series[0].values.length;
    const plot = frame(container, Object.assign({}, opts, {
      xMin: 0, xMax: Math.max(1, n - 1), yMin: yRange[0], yMax: yRange[1],
      xCount: n
    }));

    for (let i = 0; i < opts.series.length; i++) {
      const s = opts.series[i];
      const pts = [];
      for (let j = 0; j < s.values.length; j++) pts.push([j, s.values[j]]);
      el('path', {
        d: path(pts, plot), fill: 'none',
        stroke: s.color || p.series[i % p.series.length],
        'stroke-width': s.width || 2,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round',
        'stroke-dasharray': s.dashed ? '5 4' : null
      }, plot.svg);
    }
    if (opts.legend !== false && (opts.series.length > 1 || opts.forceLegend)) {
      legend(container, opts.series.map(function (s, i) {
        return { name: s.name, color: s.color || p.series[i % p.series.length], dashed: s.dashed };
      }));
    }

    const tipOpts = opts.tipSeries
      ? Object.assign({}, opts, { series: opts.tipSeries })
      : opts;
    attachCrosshair(container, plot, tipOpts, n);
    return plot;
  }

  function attachCrosshair(container, plot, opts, n) {
    const tip = tooltip(container);
    const p = plot.p;
    const line = el('line', { y1: plot.pad.top, y2: plot.pad.top + plot.h,
      stroke: p.axis, 'stroke-width': 1, 'stroke-dasharray': '3 3',
      visibility: 'hidden' }, plot.svg);
    const dots = [];
    for (let i = 0; i < opts.series.length; i++) {
      dots.push(el('circle', { r: 3.5, fill: opts.series[i].color || p.series[i % p.series.length],
        stroke: p.surface, 'stroke-width': 2, visibility: 'hidden' }, plot.svg));
    }
    const overlay = el('rect', { x: plot.pad.left, y: plot.pad.top, width: plot.w,
      height: plot.h, fill: 'transparent' }, plot.svg);

    function hide() {
      line.setAttribute('visibility', 'hidden');
      for (let i = 0; i < dots.length; i++) dots[i].setAttribute('visibility', 'hidden');
      tip.hidden = true;
    }
    overlay.addEventListener('mouseleave', hide);
    overlay.addEventListener('mousemove', function (ev) {
      const rect = plot.svg.getBoundingClientRect();
      const scale = plot.width / rect.width;
      const px = (ev.clientX - rect.left) * scale;
      let idx = Math.round((px - plot.pad.left) / plot.w * (n - 1));
      idx = Math.max(0, Math.min(n - 1, idx));
      const x = plot.sx(idx);
      line.setAttribute('x1', x); line.setAttribute('x2', x);
      line.setAttribute('visibility', 'visible');
      let html = '<strong>' + (opts.xTickLabels ? opts.xTickLabels[idx] : (opts.xLabel || 'x') + ' ' + idx) + '</strong>';
      for (let i = 0; i < opts.series.length; i++) {
        const v = opts.series[i].values[idx];
        if (Number.isFinite(v)) {
          dots[i].setAttribute('cx', x);
          dots[i].setAttribute('cy', plot.sy(v));
          dots[i].setAttribute('visibility', 'visible');
        } else {
          dots[i].setAttribute('visibility', 'hidden');
        }
        html += '<span><i style="background:' +
          (opts.series[i].color || plot.p.series[i % plot.p.series.length]) + '"></i>' +
          opts.series[i].name + '<b>' + (opts.tipFormat ? opts.tipFormat(v) : fmt(v)) + '</b></span>';
      }
      tip.innerHTML = html;
      tip.hidden = false;
      const left = Math.min(Math.max(0, (x / plot.width) * rect.width - 60), rect.width - 130);
      tip.style.left = left + 'px';
      tip.style.top = '4px';
    });
  }


  function fanChart(container, opts) {
    const p = palette();
    let yRange = extent(opts.bands[0].hi, extent(opts.bands[0].lo, null));
    if (opts.overlay) yRange = extent(opts.overlay.values, yRange);
    yRange = padRange(yRange);
    const n = opts.median.length;
    const plot = frame(container, Object.assign({}, opts, {
      xMin: 0, xMax: n - 1, yMin: yRange[0], yMax: yRange[1]
    }));

    for (let b = 0; b < opts.bands.length; b++) {
      const band = opts.bands[b];
      let d = '';
      for (let i = 0; i < n; i++) {
        d += (i ? 'L' : 'M') + plot.sx(i).toFixed(2) + ' ' + plot.sy(band.hi[i]).toFixed(2) + ' ';
      }
      for (let i = n - 1; i >= 0; i--) {
        d += 'L' + plot.sx(i).toFixed(2) + ' ' + plot.sy(band.lo[i]).toFixed(2) + ' ';
      }
      el('path', { d: d + 'Z', fill: p.ramp[b], stroke: 'none' }, plot.svg);
    }
    const medianPts = [];
    for (let i = 0; i < n; i++) medianPts.push([i, opts.median[i]]);
    el('path', { d: path(medianPts, plot), fill: 'none', stroke: p.ramp[4],
      'stroke-width': 2, 'stroke-linejoin': 'round' }, plot.svg);

    const entries = opts.bands.map(function (b, i) { return { name: b.name, color: p.ramp[i] }; });
    entries.push({ name: 'Median', color: p.ramp[4] });
    if (opts.overlay) {
      const pts = [];
      for (let i = 0; i < opts.overlay.values.length; i++) pts.push([i, opts.overlay.values[i]]);
      el('path', { d: path(pts, plot), fill: 'none', stroke: p.ink, 'stroke-width': 1.6,
        'stroke-dasharray': '5 4' }, plot.svg);
      entries.push({ name: opts.overlay.name, color: p.ink, dashed: true });
    }
    legend(container, entries);

    const series = [{ name: 'Median', values: opts.median, color: p.ramp[4] },
                    { name: opts.bands[0].name, values: opts.bands[0].lo, color: p.ramp[0] },
                    { name: opts.bands[0].name + ' upper', values: opts.bands[0].hi, color: p.ramp[0] }];
    if (opts.overlay) series.push({ name: opts.overlay.name, values: opts.overlay.values, color: p.ink });
    attachCrosshair(container, plot, Object.assign({}, opts, { series: series }), n);
    return plot;
  }


  function histogram(container, opts) {
    const p = palette();
    let all = [];
    for (let i = 0; i < opts.series.length; i++) {
      all = all.concat(Array.prototype.slice.call(opts.series[i].values));
    }
    all = all.filter(Number.isFinite).sort(function (a, b) { return a - b; });
    if (!all.length) { container.innerHTML = '<p class="empty">No data.</p>'; return null; }
    const lo = all[Math.floor(all.length * 0.001)];
    const hi = all[Math.min(all.length - 1, Math.floor(all.length * 0.999))];
    const bins = opts.bins || 56;
    const width = (hi - lo) / bins || 1;
    const counts = [];
    for (let s = 0; s < opts.series.length; s++) {
      const c = new Float64Array(bins);
      const vals = opts.series[s].values;
      let total = 0;
      for (let i = 0; i < vals.length; i++) {
        if (!Number.isFinite(vals[i])) continue;
        total++;
        const b = Math.floor((vals[i] - lo) / width);
        if (b >= 0 && b < bins) c[b]++;
      }
      for (let b = 0; b < bins; b++) c[b] = total ? c[b] / (total * width) : 0;
      counts.push(c);
    }
    const logY = opts.logY !== false;
    let yMax = 0, yMin = Infinity;
    for (let s = 0; s < counts.length; s++) {
      for (let b = 0; b < bins; b++) {
        if (counts[s][b] > yMax) yMax = counts[s][b];
        if (counts[s][b] > 0 && counts[s][b] < yMin) yMin = counts[s][b];
      }
    }
    const tf = logY ? function (v) { return v > 0 ? Math.log10(v) : NaN; } : function (v) { return v; };
    const yLo = logY ? Math.floor(tf(yMin)) : 0;
    const yHi = logY ? Math.ceil(tf(yMax)) : yMax * 1.08;
    const yTicks = [];
    if (logY) { for (let v = yLo; v <= yHi; v++) yTicks.push(v); }

    const plot = frame(container, Object.assign({}, opts, {
      xMin: lo, xMax: hi, yMin: yLo, yMax: yHi,
      yTicks: logY ? yTicks : undefined,
      yFormat: logY ? function (v) { return '1e' + v; } : undefined,
      xTicks: niceTicks(lo, hi, 6)
    }));

    for (let s = 0; s < counts.length; s++) {
      let d = '';
      for (let b = 0; b < bins; b++) {
        const x0 = plot.sx(lo + b * width), x1 = plot.sx(lo + (b + 1) * width);
        const v = tf(counts[s][b]);
        const y = Number.isFinite(v) ? plot.sy(Math.max(v, yLo)) : plot.sy(yLo);
        d += (b ? 'L' : 'M') + x0.toFixed(2) + ' ' + y.toFixed(2) + ' L' + x1.toFixed(2) + ' ' + y.toFixed(2) + ' ';
      }
      el('path', { d: d, fill: 'none', stroke: opts.series[s].color || p.series[s % p.series.length],
        'stroke-width': 2, 'stroke-linejoin': 'round' }, plot.svg);
    }
    legend(container, opts.series.map(function (s, i) {
      return { name: s.name, color: s.color || p.series[i % p.series.length] };
    }));
    return plot;
  }


  function acfChart(container, opts) {
    const p = palette();
    let yRange = null;
    for (let i = 0; i < opts.series.length; i++) yRange = extent(opts.series[i].values, yRange);
    yRange = padRange(yRange, 0.15);
    const n = opts.series[0].values.length;
    const plot = frame(container, Object.assign({}, opts, {
      xMin: 1, xMax: n, yMin: Math.min(yRange[0], 0), yMax: Math.max(yRange[1], 0),
      xTicks: niceTicks(1, n, 5)
    }));
    el('line', { x1: plot.pad.left, x2: plot.pad.left + plot.w,
      y1: plot.sy(0), y2: plot.sy(0), stroke: p.axis, 'stroke-width': 1 }, plot.svg);
    for (let i = 0; i < opts.series.length; i++) {
      const s = opts.series[i];
      const color = s.color || p.series[i % p.series.length];
      const pts = [];
      for (let j = 0; j < s.values.length; j++) pts.push([j + 1, s.values[j]]);
      el('path', { d: path(pts, plot), fill: 'none', stroke: color, 'stroke-width': 2 }, plot.svg);
      for (let j = 0; j < s.values.length; j++) {
        if (!Number.isFinite(s.values[j])) continue;
        el('circle', { cx: plot.sx(j + 1), cy: plot.sy(s.values[j]), r: 3,
          fill: color, stroke: p.surface, 'stroke-width': 1.5 }, plot.svg);
      }
    }
    legend(container, opts.series.map(function (s, i) {
      return { name: s.name, color: s.color || p.series[i % p.series.length] };
    }));
    return plot;
  }


  function heatmaps(container, opts) {
    container.innerHTML = '';
    const p = palette();
    const labels = opts.labels;
    const n = labels.length;
    const cell = 54, labelW = 88, gap = 26;
    const width = opts.matrices.length * (labelW + n * cell) + gap * (opts.matrices.length - 1) + 20;
    const height = n * cell + 46;
    const svg = el('svg', { viewBox: '0 0 ' + width + ' ' + height,
      preserveAspectRatio: 'xMidYMid meet', class: 'chart-svg' }, container);
    svg.style.maxWidth = width + 'px';
    svg.style.margin = '0 auto';
    const tip = tooltip(container);

    function colorFor(v) {
      const t = Math.max(-1, Math.min(1, v));
      const mix = function (a, b, f) {
        const pa = [parseInt(a.slice(1, 3), 16), parseInt(a.slice(3, 5), 16), parseInt(a.slice(5, 7), 16)];
        const pb = [parseInt(b.slice(1, 3), 16), parseInt(b.slice(3, 5), 16), parseInt(b.slice(5, 7), 16)];
        return 'rgb(' + pa.map(function (c, i) {
          return Math.round(c + (pb[i] - c) * f);
        }).join(',') + ')';
      };
      return t >= 0 ? mix(p.divMid, p.divHigh, t) : mix(p.divMid, p.divLow, -t);
    }

    for (let m = 0; m < opts.matrices.length; m++) {
      const ox = m * (labelW + n * cell + gap) + labelW;
      const t = el('text', { x: ox, y: 14, fill: p.ink, 'font-size': 12,
        'font-weight': 600 }, svg);
      t.textContent = opts.matrices[m].title;
      for (let i = 0; i < n; i++) {
        const rl = el('text', { x: ox - 8, y: 30 + i * cell + cell / 2 + 4,
          'text-anchor': 'end', fill: p.muted, 'font-size': 10 }, svg);
        rl.textContent = labels[i];
        for (let j = 0; j < n; j++) {
          const v = opts.matrices[m].values[i][j];
          const rect = el('rect', { x: ox + j * cell + 1, y: 30 + i * cell + 1,
            width: cell - 2, height: cell - 2, rx: 3, fill: colorFor(v) }, svg);
          rect.style.cursor = 'crosshair';
          const label = el('text', { x: ox + j * cell + cell / 2,
            y: 30 + i * cell + cell / 2 + 4, 'text-anchor': 'middle',
            fill: p.ink, 'font-size': 10.5 }, svg);
          label.textContent = v.toFixed(2);
          (function (i, j, v, title) {
            rect.addEventListener('mouseenter', function () {
              tip.innerHTML = '<strong>' + title + '</strong><span>' + labels[i] + ' / ' +
                labels[j] + '<b>' + v.toFixed(3) + '</b></span>';
              tip.hidden = false;
              tip.style.left = '8px';
              tip.style.top = '4px';
            });
            rect.addEventListener('mouseleave', function () { tip.hidden = true; });
          })(i, j, v, opts.matrices[m].title);
        }
      }
      for (let j = 0; j < n; j++) {
        const cl = el('text', { x: ox + j * cell + cell / 2, y: height - 8,
          'text-anchor': 'middle', fill: p.muted, 'font-size': 10 }, svg);
        cl.textContent = labels[j];
      }
    }
    return svg;
  }


  function barChart(container, opts) {
    container.innerHTML = '';
    const p = palette();
    const tip = tooltip(container);
    const n = opts.labels.length;
    const rowH = 26, labelW = 108;
    const width = 460, height = n * rowH + 12;
    const svg = el('svg', { viewBox: '0 0 ' + width + ' ' + height,
      preserveAspectRatio: 'xMidYMid meet', class: 'chart-svg' }, container);
    svg.style.maxWidth = width + 'px';
    const max = Math.max.apply(null, opts.values.concat([1e-9]));
    const trackW = width - labelW - 62;
    for (let i = 0; i < n; i++) {
      const y = i * rowH + 6;
      const t = el('text', { x: labelW - 10, y: y + 13, 'text-anchor': 'end',
        fill: p.secondary, 'font-size': 11 }, svg);
      t.textContent = opts.labels[i];
      el('rect', { x: labelW, y: y + 3, width: trackW, height: 12, rx: 4,
        fill: p.grid, opacity: 0.55 }, svg);
      const w = Math.max(2, trackW * opts.values[i] / max);
      const bar = el('rect', { x: labelW, y: y + 3, width: w, height: 12, rx: 4,
        fill: opts.color || p.series[0] }, svg);
      bar.style.cursor = 'crosshair';
      const v = el('text', { x: labelW + trackW + 8, y: y + 13, fill: p.muted,
        'font-size': 10.5 }, svg);
      v.textContent = (opts.format || function (x) { return x.toFixed(3); })(opts.values[i]);
      (function (i) {
        bar.addEventListener('mouseenter', function () {
          tip.innerHTML = '<strong>' + opts.labels[i] + '</strong><span>' +
            (opts.tipLabel || 'weight') + '<b>' + opts.values[i].toFixed(4) + '</b></span>';
          tip.hidden = false; tip.style.left = '8px'; tip.style.top = '4px';
        });
        bar.addEventListener('mouseleave', function () { tip.hidden = true; });
      })(i);
    }
    return svg;
  }

  global.Charts = {
    line: lineChart, fan: fanChart, histogram: histogram, acf: acfChart,
    heatmaps: heatmaps, bars: barChart, palette: palette, fmt: fmt,
    niceTicks: niceTicks, tickDecimals: tickDecimals
  };
})(typeof window !== 'undefined' ? window : globalThis);
