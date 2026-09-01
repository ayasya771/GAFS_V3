
(function (global) {
  'use strict';

  const NN = global.NN;

  function tensorReader(manifest, buffer) {
    const all = new Float32Array(buffer);
    return function (name) {
      const spec = manifest.tensors[name];
      if (!spec) throw new Error('missing tensor: ' + name);
      const flat = all.subarray(spec.offset, spec.offset + spec.size);
      if (spec.shape.length === 1) return flat;
      return NN.mat(spec.shape[0], spec.shape[1], flat);
    };
  }


  function buildGLU(get, prefix) {
    const W = get(prefix + '.fc.weight');
    const b = get(prefix + '.fc.bias');
    const out = W.rows / 2;
    return function (x) {
      const proj = NN.linear(x, W, b);
      const res = NN.mat(x.rows, out);
      const p = proj.data, r = res.data;
      for (let i = 0; i < x.rows; i++) {
        const po = i * proj.cols, ro = i * out;
        for (let j = 0; j < out; j++) {
          r[ro + j] = p[po + j] / (1 + Math.exp(-p[po + out + j]));
        }
      }
      return res;
    };
  }

  function buildGRN(get, prefix, hasSkip, hasCtx) {
    const fc1W = get(prefix + '.fc1.weight'), fc1b = get(prefix + '.fc1.bias');
    const fc2W = get(prefix + '.fc2.weight'), fc2b = get(prefix + '.fc2.bias');
    const gate = buildGLU(get, prefix + '.gate');
    const normW = get(prefix + '.norm.weight'), normB = get(prefix + '.norm.bias');
    const skipW = hasSkip ? get(prefix + '.skip.weight') : null;
    const skipb = hasSkip ? get(prefix + '.skip.bias') : null;
    const ctxW = hasCtx ? get(prefix + '.ctx.weight') : null;


    return function (x, ctxRow) {
      let h = NN.linear(x, fc1W, fc1b);
      if (ctxW && ctxRow) NN.addRowVector(h, ctxRow);
      NN.elu(h);
      h = NN.linear(h, fc2W, fc2b);
      const skip = skipW ? NN.linear(x, skipW, skipb) : NN.clone(x);
      NN.addInto(skip, gate(h));
      return NN.layerNorm(skip, normW, normB);
    };
  }

  function projectContext(get, prefix) {
    const W = get(prefix + '.ctx.weight');
    return function (ctx) {
      return NN.linear(ctx, W, null).data;
    };
  }

  function buildAddGateNorm(get, prefix) {
    const glu = buildGLU(get, prefix + '.glu');
    const normW = get(prefix + '.norm.weight'), normB = get(prefix + '.norm.bias');
    return function (x, residual) {
      const out = NN.clone(residual);
      NN.addInto(out, glu(x));
      return NN.layerNorm(out, normW, normB);
    };
  }

  function buildLSTM(get, prefix) {
    const Wih = get(prefix + '.weight_ih_l0'), Whh = get(prefix + '.weight_hh_l0');
    const bih = get(prefix + '.bias_ih_l0'), bhh = get(prefix + '.bias_hh_l0');
    const hidden = Whh.cols;
    const bias = new Float32Array(bih.length);
    for (let i = 0; i < bias.length; i++) bias[i] = bih[i] + bhh[i];


    return {
      hidden: hidden,
      step: function (x, state) {
        const gates = NN.linear(x, Wih, bias);
        const hg = NN.linear(state.h, Whh, null);
        NN.addInto(gates, hg);
        const g = gates.data, B = x.rows;
        const hData = state.h.data, cData = state.c.data;
        for (let b = 0; b < B; b++) {
          const o = b * 4 * hidden, ho = b * hidden;
          NN.sigmoidArr(g, o, o + hidden);
          NN.sigmoidArr(g, o + hidden, o + 2 * hidden);
          NN.tanhArr(g, o + 2 * hidden, o + 3 * hidden);
          NN.sigmoidArr(g, o + 3 * hidden, o + 4 * hidden);
          for (let j = 0; j < hidden; j++) {
            const c = g[o + hidden + j] * cData[ho + j] +
                      g[o + j] * g[o + 2 * hidden + j];
            cData[ho + j] = c;
            hData[ho + j] = g[o + 3 * hidden + j] * Math.tanh(c);
          }
        }
        return state;
      }
    };
  }


  function build(manifest, buffer) {
    const get = tensorReader(manifest, buffer);
    const A = manifest.arch;
    const nVars = A.n_features, hidden = A.hidden, heads = A.heads;
    const headDim = hidden / heads;
    const scale = 1 / Math.sqrt(headDim);

    const condEncoder = buildGRN(get, 'cond_encoder', true, false);
    const ctxSelect = buildGRN(get, 'ctx_select', false, false);
    const ctxH = buildGRN(get, 'ctx_h', false, false);
    const ctxC = buildGRN(get, 'ctx_c', false, false);
    const ctxEnrich = buildGRN(get, 'ctx_enrich', false, false);

    const embedW = [], embedB = [], varGRN = [];
    for (let i = 0; i < nVars; i++) {
      embedW.push(get('vsn.embed.' + i + '.weight'));
      embedB.push(get('vsn.embed.' + i + '.bias'));
      varGRN.push(buildGRN(get, 'vsn.var_grns.' + i, false, false));
    }
    const weightGRN = buildGRN(get, 'vsn.weight_grn', true, true);
    const weightGRNCtx = projectContext(get, 'vsn.weight_grn');

    const encLSTM = buildLSTM(get, 'enc_lstm');
    const decLSTM = buildLSTM(get, 'dec_lstm');
    const decInputW = get('dec_input.weight'), decInputB = get('dec_input.bias');
    const gateEnc = buildAddGateNorm(get, 'gate_enc');
    const gateDec = buildAddGateNorm(get, 'gate_dec');
    const enrich = buildGRN(get, 'enrich', false, true);
    const enrichCtx = projectContext(get, 'enrich');
    const gateAttn = buildAddGateNorm(get, 'gate_attn');
    const posFF = buildGRN(get, 'pos_ff', false, false);
    const gateFinal = buildAddGateNorm(get, 'gate_final');
    const headW = get('head.weight'), headB = get('head.bias');

    const inW = get('attn.in_proj_weight'), inB = get('attn.in_proj_bias');
    const Wq = NN.mat(hidden, hidden, inW.data.subarray(0, hidden * hidden));
    const Wk = NN.mat(hidden, hidden, inW.data.subarray(hidden * hidden, 2 * hidden * hidden));
    const Wv = NN.mat(hidden, hidden, inW.data.subarray(2 * hidden * hidden, 3 * hidden * hidden));
    const bq = inB.subarray(0, hidden);
    const bk = inB.subarray(hidden, 2 * hidden);
    const bv = inB.subarray(2 * hidden, 3 * hidden);
    const outW = get('attn.out_proj.weight'), outB = get('attn.out_proj.bias');


    function variableSelection(x, ctxRow) {
      const T = x.rows;
      const embedded = [];
      for (let v = 0; v < nVars; v++) {
        const col = NN.mat(T, 1);
        for (let t = 0; t < T; t++) col.data[t] = x.data[t * nVars + v];
        embedded.push(NN.linear(col, embedW[v], embedB[v]));
      }
      const flat = NN.hcat(embedded);
      const weights = NN.softmaxRows(weightGRN(flat, ctxRow));
      const processed = [];
      for (let v = 0; v < nVars; v++) processed.push(varGRN[v](embedded[v]));
      const fused = NN.mat(T, hidden);
      for (let v = 0; v < nVars; v++) {
        const p = processed[v].data, f = fused.data, w = weights.data;
        for (let t = 0; t < T; t++) {
          const wv = w[t * nVars + v], o = t * hidden;
          for (let j = 0; j < hidden; j++) f[o + j] += wv * p[o + j];
        }
      }
      return { fused: fused, weights: weights };
    }


    function encode(xHist, cond) {
      const condRow = (cond && cond.cols) ? cond : NN.mat(1, 1);
      const c = condEncoder(condRow, null);
      const selection = variableSelection(xHist, weightGRNCtx(ctxSelect(c, null)));

      const state = {
        h: NN.clone(ctxH(c, null)),
        c: NN.clone(ctxC(c, null))
      };
      const T = xHist.rows;
      const encOut = NN.mat(T, hidden);
      for (let t = 0; t < T; t++) {
        const xt = NN.mat(1, hidden, selection.fused.data.subarray(t * hidden, (t + 1) * hidden));
        encLSTM.step(xt, state);
        encOut.data.set(state.h.data, t * hidden);
      }
      const encState = { h: NN.clone(state.h), c: NN.clone(state.c) };
      const gatedEnc = gateEnc(encOut, selection.fused);
      const enrichRow = enrichCtx(ctxEnrich(c, null));
      const enrichedHist = enrich(gatedEnc, enrichRow);

      return {
        c: c,
        encState: encState,
        enrichedHist: enrichedHist,
        kHist: NN.linear(enrichedHist, Wk, bk),
        vHist: NN.linear(enrichedHist, Wv, bv),
        enrichRow: enrichRow,
        varWeights: selection.weights,
        lookback: T
      };
    }


    function decode(ctx, Z) {
      const B = Z[0].rows, h = A.horizon, k = ctx.lookback, zDim = A.z_dim;
      const nAssets = A.n_assets;
      const cRow = ctx.c.data;
      const state = {
        h: NN.broadcastRow(ctx.encState.h, B),
        c: NN.broadcastRow(ctx.encState.c, B)
      };

      const decIn = [], decOut = [], enrichedDec = [];
      const inp = NN.mat(B, zDim + hidden);
      for (let b = 0; b < B; b++) inp.data.set(cRow, b * inp.cols + zDim);
      for (let t = 0; t < h; t++) {
        for (let b = 0; b < B; b++) {
          inp.data.set(Z[t].data.subarray(b * zDim, (b + 1) * zDim), b * inp.cols);
        }
        const step = NN.linear(inp, decInputW, decInputB);
        decLSTM.step(step, state);
        decIn.push(step);
        decOut.push(NN.clone(state.h));
      }
      for (let t = 0; t < h; t++) {
        decOut[t] = gateDec(decOut[t], decIn[t]);
        enrichedDec.push(enrich(decOut[t], ctx.enrichRow));
      }


      const q = [], kDec = [], vDec = [], att = [];
      for (let t = 0; t < h; t++) {
        q.push(NN.linear(enrichedDec[t], Wq, bq));
        kDec.push(NN.linear(enrichedDec[t], Wk, bk));
        vDec.push(NN.linear(enrichedDec[t], Wv, bv));
        att.push(NN.mat(B, hidden));
      }

      const scores = new Float32Array(k + h);
      const kHist = ctx.kHist.data, vHist = ctx.vHist.data;
      for (let b = 0; b < B; b++) {
        const bo = b * hidden;
        for (let g = 0; g < heads; g++) {
          const off = g * headDim;
          for (let i = 0; i < h; i++) {
            const qd = q[i].data, qo = bo + off;
            let max = -Infinity;
            const limit = k + i + 1;
            for (let j = 0; j < k; j++) {
              const so = j * hidden + off;
              let dot = 0;
              for (let d = 0; d < headDim; d++) dot += qd[qo + d] * kHist[so + d];
              dot *= scale;
              scores[j] = dot;
              if (dot > max) max = dot;
            }
            for (let j = k; j < limit; j++) {
              const src = kDec[j - k].data;
              let dot = 0;
              for (let d = 0; d < headDim; d++) dot += qd[qo + d] * src[bo + off + d];
              dot *= scale;
              scores[j] = dot;
              if (dot > max) max = dot;
            }
            let sum = 0;
            for (let j = 0; j < limit; j++) {
              scores[j] = Math.exp(scores[j] - max);
              sum += scores[j];
            }
            const inv = 1 / sum;
            const ad = att[i].data, ao = bo + off;
            for (let j = 0; j < k; j++) {
              const w = scores[j] * inv;
              const so = j * hidden + off;
              for (let d = 0; d < headDim; d++) ad[ao + d] += w * vHist[so + d];
            }
            for (let j = k; j < limit; j++) {
              const w = scores[j] * inv;
              const src = vDec[j - k].data;
              for (let d = 0; d < headDim; d++) ad[ao + d] += w * src[bo + off + d];
            }
          }
        }
      }


      const out = NN.mat(B, h * nAssets);
      const withNoise = NN.mat(B, hidden + zDim);
      for (let t = 0; t < h; t++) {
        const projected = NN.linear(att[t], outW, outB);
        const gated = gateAttn(projected, enrichedDec[t]);
        const fused = gateFinal(posFF(gated, null), decOut[t]);
        for (let b = 0; b < B; b++) {
          withNoise.data.set(fused.data.subarray(b * hidden, (b + 1) * hidden),
                             b * withNoise.cols);
          withNoise.data.set(Z[t].data.subarray(b * zDim, (b + 1) * zDim),
                             b * withNoise.cols + hidden);
        }
        const y = NN.linear(withNoise, headW, headB);
        for (let b = 0; b < B; b++) {
          for (let aI = 0; aI < nAssets; aI++) {
            out.data[b * out.cols + t * nAssets + aI] = y.data[b * nAssets + aI];
          }
        }
      }
      return out;
    }

    function sampleNoise(B, random) {
      const Z = [];
      for (let t = 0; t < A.horizon; t++) {
        const m = NN.mat(B, A.z_dim);
        for (let i = 0; i < m.data.length; i++) m.data[i] = random.normal();
        Z.push(m);
      }
      return Z;
    }

    return {
      arch: A,
      manifest: manifest,
      encode: encode,
      decode: decode,
      sampleNoise: sampleNoise,

      forward: function (xHist, cond, Z) {
        return decode(encode(xHist, cond), Z);
      }
    };
  }


  function verifyParity(model, parity) {
    const shp = parity.shapes;
    const x = NN.mat(shp.x_hist[0], shp.x_hist[1], Float32Array.from(parity.x_hist));
    const cond = parity.cond.length
      ? NN.mat(1, parity.cond.length, Float32Array.from(parity.cond))
      : null;
    const zDim = shp.z[1];
    const Z = [];
    for (let t = 0; t < shp.z[0]; t++) {
      Z.push(NN.mat(1, zDim, Float32Array.from(parity.z.slice(t * zDim, (t + 1) * zDim))));
    }
    const got = model.forward(x, cond, Z).data;
    let maxAbs = 0, maxRel = 0;
    for (let i = 0; i < parity.y.length; i++) {
      const diff = Math.abs(got[i] - parity.y[i]);
      if (diff > maxAbs) maxAbs = diff;
      const denom = Math.max(1e-6, Math.abs(parity.y[i]));
      if (diff / denom > maxRel) maxRel = diff / denom;
    }
    return { maxAbs: maxAbs, maxRel: maxRel, n: parity.y.length };
  }

  async function load(base) {
    const manifest = await fetch(base + '/generator.json').then(function (r) {
      if (!r.ok) throw new Error('generator.json: HTTP ' + r.status);
      return r.json();
    });
    const buffer = await fetch(base + '/' + manifest.weights).then(function (r) {
      if (!r.ok) throw new Error(manifest.weights + ': HTTP ' + r.status);
      return r.arrayBuffer();
    });
    const parity = await fetch(base + '/parity.json').then(function (r) {
      return r.ok ? r.json() : null;
    });
    const model = build(manifest, buffer);
    return { model: model, parity: parity ? verifyParity(model, parity) : null };
  }

  global.Generator = { build: build, load: load, verifyParity: verifyParity };
})(typeof window !== 'undefined' ? window : globalThis);
