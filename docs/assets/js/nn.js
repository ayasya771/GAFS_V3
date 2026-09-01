
(function (global) {
  'use strict';

  function mat(rows, cols, data) {
    return { rows: rows, cols: cols, data: data || new Float32Array(rows * cols) };
  }

  function like(m) {
    return mat(m.rows, m.cols, new Float32Array(m.rows * m.cols));
  }

  function clone(m) {
    return mat(m.rows, m.cols, m.data.slice());
  }


  function linear(A, B, bias, out) {
    const n = A.rows, k = A.cols, m = B.rows;
    if (B.cols !== k) throw new Error('linear: inner dims ' + k + ' vs ' + B.cols);
    const C = out || mat(n, m);
    const a = A.data, b = B.data, c = C.data;
    for (let i = 0; i < n; i++) {
      const ao = i * k, co = i * m;
      for (let j = 0; j < m; j++) {
        const bo = j * k;
        let sum = bias ? bias[j] : 0;
        for (let p = 0; p < k; p++) sum += a[ao + p] * b[bo + p];
        c[co + j] = sum;
      }
    }
    return C;
  }


  function matmul(A, B, out) {
    const n = A.rows, k = A.cols, m = B.cols;
    if (B.rows !== k) throw new Error('matmul: inner dims ' + k + ' vs ' + B.rows);
    const C = out || mat(n, m);
    const a = A.data, b = B.data, c = C.data;
    c.fill(0);
    for (let i = 0; i < n; i++) {
      const ao = i * k, co = i * m;
      for (let p = 0; p < k; p++) {
        const av = a[ao + p];
        if (av === 0) continue;
        const bo = p * m;
        for (let j = 0; j < m; j++) c[co + j] += av * b[bo + j];
      }
    }
    return C;
  }

  function addRowVector(m, vec) {
    const d = m.data, cols = m.cols;
    for (let i = 0; i < m.rows; i++) {
      const o = i * cols;
      for (let j = 0; j < cols; j++) d[o + j] += vec[j];
    }
    return m;
  }

  function addInto(target, other) {
    const a = target.data, b = other.data;
    for (let i = 0; i < a.length; i++) a[i] += b[i];
    return target;
  }

  function mulInto(target, other) {
    const a = target.data, b = other.data;
    for (let i = 0; i < a.length; i++) a[i] *= b[i];
    return target;
  }

  function elu(m) {
    const d = m.data;
    for (let i = 0; i < d.length; i++) if (d[i] <= 0) d[i] = Math.exp(d[i]) - 1;
    return m;
  }

  function sigmoidArr(d, from, to) {
    for (let i = from; i < to; i++) d[i] = 1 / (1 + Math.exp(-d[i]));
  }

  function tanhArr(d, from, to) {
    for (let i = from; i < to; i++) d[i] = Math.tanh(d[i]);
  }


  function layerNorm(m, weight, bias, eps) {
    const e = eps === undefined ? 1e-5 : eps;
    const d = m.data, cols = m.cols;
    for (let i = 0; i < m.rows; i++) {
      const o = i * cols;
      let mean = 0;
      for (let j = 0; j < cols; j++) mean += d[o + j];
      mean /= cols;
      let varr = 0;
      for (let j = 0; j < cols; j++) {
        const v = d[o + j] - mean;
        varr += v * v;
      }
      varr /= cols;
      const inv = 1 / Math.sqrt(varr + e);
      for (let j = 0; j < cols; j++) {
        d[o + j] = (d[o + j] - mean) * inv * weight[j] + bias[j];
      }
    }
    return m;
  }


  function softmaxRows(m) {
    const d = m.data, cols = m.cols;
    for (let i = 0; i < m.rows; i++) {
      const o = i * cols;
      let max = -Infinity;
      for (let j = 0; j < cols; j++) if (d[o + j] > max) max = d[o + j];
      let sum = 0;
      for (let j = 0; j < cols; j++) {
        const v = Math.exp(d[o + j] - max);
        d[o + j] = v;
        sum += v;
      }
      const inv = 1 / sum;
      for (let j = 0; j < cols; j++) d[o + j] *= inv;
    }
    return m;
  }


  function hcat(parts) {
    const rows = parts[0].rows;
    let cols = 0;
    for (let p = 0; p < parts.length; p++) cols += parts[p].cols;
    const out = mat(rows, cols);
    let offset = 0;
    for (let p = 0; p < parts.length; p++) {
      const src = parts[p], sc = src.cols;
      for (let i = 0; i < rows; i++) {
        out.data.set(src.data.subarray(i * sc, i * sc + sc), i * cols + offset);
      }
      offset += sc;
    }
    return out;
  }


  function colSlice(m, from, to) {
    const cols = to - from;
    const out = mat(m.rows, cols);
    for (let i = 0; i < m.rows; i++) {
      out.data.set(m.data.subarray(i * m.cols + from, i * m.cols + to), i * cols);
    }
    return out;
  }


  function broadcastRow(row, n) {
    const cols = row.cols;
    const out = mat(n, cols);
    for (let i = 0; i < n; i++) out.data.set(row.data, i * cols);
    return out;
  }


  function rng(seed) {
    let s = seed >>> 0;
    const uniform = function () {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    return {
      uniform: uniform,
      normal: function () {
        if (spare !== null) { const v = spare; spare = null; return v; }
        let u = 0, v = 0, s2 = 0;
        do {
          u = uniform() * 2 - 1;
          v = uniform() * 2 - 1;
          s2 = u * u + v * v;
        } while (s2 >= 1 || s2 === 0);
        const f = Math.sqrt(-2 * Math.log(s2) / s2);
        spare = v * f;
        return u * f;
      }
    };
  }

  global.NN = {
    mat: mat, like: like, clone: clone, linear: linear, matmul: matmul,
    addRowVector: addRowVector, addInto: addInto, mulInto: mulInto,
    elu: elu, sigmoidArr: sigmoidArr, tanhArr: tanhArr,
    layerNorm: layerNorm, softmaxRows: softmaxRows,
    hcat: hcat, colSlice: colSlice, broadcastRow: broadcastRow, rng: rng
  };
})(typeof window !== 'undefined' ? window : globalThis);
