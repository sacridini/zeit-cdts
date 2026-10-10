
// ---------------------------------------------------------------------- embeddings
// zeit.plot's views of a cube of embeddings. This file is appended to viewer.js into one
// module (see _widget.viewer_bundle), so `Viewer`, `h`, `VS`, `niceTicks` and `fmtTick` are
// in scope.
//
// Each year arrives as int8 vectors, one scale per dimension, laid out (layers, h, w, 4):
// layer k holds dimensions 4k..4k+3 and uploads as one layer of an RGBA8I texture array.
// One shader draws three views from them:
// - components: three of the first six principal components as red, green and blue
//   (fitted on every year, or on the visible area);
// - similarity: the cosine similarity of every pixel to a reference vector, the one under
//   the cursor (live) or a pinned pixel's, in the year shown or in a fixed year;
// - change: 1 - the cosine similarity of each pixel's vectors in the year shown and the
//   previous year (or a fixed year).
// The per-dimension rows the shader needs (the reference and the components, both times the
// scale, and the scale squared) travel in a small float texture.

const EMB_FS = `#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;
uniform highp isampler2DArray A;   // the year shown
uniform highp isampler2DArray B;   // the year it is compared with (change)
uniform sampler2D rows;            // L x 5: reference*s, 3 components*s, s*s
uniform sampler2D luts;            // 256 x 2: similarity, change
uniform int mode, layers, raw;
uniform vec3 offset, span;
uniform vec2 range;
uniform float refNorm;
in vec2 uv; out vec4 color;
void main() {
  ivec2 size = textureSize(A, 0).xy;
  ivec2 at = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
  ivec4 first = texelFetch(A, ivec3(at, 0), 0);
  if (first.r == -128) discard;
  if (mode == 0) {
    vec3 acc = vec3(0.0);
    for (int k = 0; k < layers; k++) {
      vec4 q = vec4(texelFetch(A, ivec3(at, k), 0));
      acc += vec3(dot(q, texelFetch(rows, ivec2(k, 1), 0)), dot(q, texelFetch(rows, ivec2(k, 2), 0)),
                  dot(q, texelFetch(rows, ivec2(k, 3), 0)));
    }
    color = vec4(clamp((acc - offset) / span, 0.0, 1.0), 1.0);
    return;
  }
  float value;
  if (mode == 1) {
    float d = 0.0, n = 0.0;
    for (int k = 0; k < layers; k++) {
      vec4 q = vec4(texelFetch(A, ivec3(at, k), 0));
      d += dot(q, texelFetch(rows, ivec2(k, 0), 0));
      n += dot(q * q, texelFetch(rows, ivec2(k, 4), 0));
    }
    if (n <= 0.0 || refNorm <= 0.0) discard;
    value = d / (sqrt(n) * refNorm);
  } else {
    if (texelFetch(B, ivec3(at, 0), 0).r == -128) discard;
    float d = 0.0, na = 0.0, nb = 0.0;
    for (int k = 0; k < layers; k++) {
      vec4 q = vec4(texelFetch(A, ivec3(at, k), 0)), p = vec4(texelFetch(B, ivec3(at, k), 0));
      vec4 s2 = texelFetch(rows, ivec2(k, 4), 0);
      d += dot(q * p, s2); na += dot(q * q, s2); nb += dot(p * p, s2);
    }
    if (na <= 0.0 || nb <= 0.0) discard;
    value = 1.0 - d / sqrt(na * nb);
  }
  float t = clamp((value - range.x) / max(range.y - range.x, 1e-9), 0.0, 1.0);
  if (raw == 1) {   // the value itself, in 16 bits (tests read it back)
    float x = floor(t * 65535.0 + 0.5);
    color = vec4(floor(x / 256.0) / 255.0, mod(x, 256.0) / 255.0, 0.0, 1.0);
    return;
  }
  color = texelFetch(luts, ivec2(1 + int(round(t * 254.0)), mode - 1), 0);
}`;

const EMB_VIEWS = [["components", "Components (RGB)"], ["similarity", "Similarity"], ["change", "Change"]];
const EMB_MODES = { components: 0, similarity: 1, change: 2 };
const PIN_COLOR = "#d62728";

function percentile(values, q) {
  if (!values.length) return NaN;
  const s = Float64Array.from(values).sort();
  return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))];
}

class Embeddings {
  constructor(viewer, meta, buffers) {
    this.v = viewer;
    this.e = meta.embedding;
    this.D = this.e.dims; this.L = this.e.layers;
    this.scale = Float32Array.from(this.e.scale);
    const lut = (b) => new Uint8Array(b.buffer || b, b.byteOffset || 0, 1024);
    this.lutBytes = new Uint8Array(2048);
    this.lutBytes.set(lut(buffers[1]), 0); this.lutBytes.set(lut(buffers[2]), 1024);
    this.mode = "components";
    this.scope = "all";
    this.pca = this.e.pca;
    this.comps = [0, 1, 2];
    this.cursor = null;          // full-resolution cell under the mouse
    this.pinned = null;          // [cx, cy] of the pinned reference
    this.refYear = null;         // frame index the reference is taken from, when fixed
    this.compare = "previous";   // change: "previous" or a frame index
    this.uploaded = { A: -1, B: -1 };
    this.changeRanges = new Map();
    this.raw = false;
    this.build();
  }

  destroy() {
    this.panel.remove();
    const gl = this.v.gl;
    if (gl) for (const t of [this.texA, this.texB, this.texDA, this.texDB, this.rowsTex, this.lutTex]) if (t) gl.deleteTexture(t);
    if (this.v.pin) this.v.pin = null;
  }

  // ------------------------------------------------------------------ panel
  build() {
    const v = this.v;
    const select = (options, value, onchange) => {
      const s = h("select", { class: "zv-var", onchange: () => onchange(s.value) });
      for (const [k, text] of options) { const o = h("option", { value: k }, text); if (String(k) === String(value)) o.selected = true; s.append(o); }
      return s;
    };
    this.viewSelect = select(EMB_VIEWS, this.mode, (value) => this.setMode(value));
    this.scopeSelect = select([["all", "every year"], ["visible", "visible area"]], this.scope, (value) => {
      this.scope = value;
      if (value === "all") { this.pca = this.e.pca; this.pcaWindow = null; this.changed(); } else this.fitVisible();
    });
    const pcs = [0, 1, 2, 3, 4, 5].filter((k) => k < this.pca.weights.length).map((k) => [k, `PC${k + 1}`]);
    this.compSelects = ["R", "G", "B"].map((c, i) => h("label", {}, c + " ",
      select(pcs, this.comps[i], (value) => { this.comps[i] = +value; this.changed(); })));
    this.explained = h("div", { class: "zv-emb-note" });
    this.refText = h("div", { class: "zv-emb-note" });
    this.fixBox = h("input", { type: "checkbox" });
    this.fixBox.addEventListener("change", () => {
      this.refYear = this.fixBox.checked ? v.index : null;
      this.changed();
    });
    this.unpinBtn = h("button", { class: "zv-play zv-emb-btn", onclick: () => this.unpin() }, "unpin");
    const years = v.meta.labels.map((t, i) => [i, t]);
    this.compareSelect = select([["previous", "the previous year"], ...years], this.compare, (value) => {
      this.compare = value === "previous" ? "previous" : +value; this.changed(); v.scheduleDetail();
    });
    this.groups = {
      components: h("div", { class: "zv-emb-group" }, h("label", {}, "fitted on ", this.scopeSelect),
        h("div", { class: "zv-emb-row" }, ...this.compSelects), this.explained),
      similarity: h("div", { class: "zv-emb-group" }, this.refText,
        h("label", {}, this.fixBox, " keep the reference's year"), this.unpinBtn),
      change: h("div", { class: "zv-emb-group" }, h("label", {}, "from ", this.compareSelect)),
    };
    this.profile = h("div", { class: "zv-emb-profile" });
    this.panel = h("div", { class: "zv-emb" },
      h("div", { class: "zv-emb-title" }, "Embeddings"),
      h("div", { class: "zv-emb-note" }, `${this.e.source ? this.e.source + " · " : ""}${this.D} dimensions`),
      h("label", {}, "view ", this.viewSelect), ...Object.values(this.groups), this.profile);
    v.side.append(this.panel);
    v.on("frame", () => { if (this.v.emb === this) { this.changed(false); this.updateChart(); } });
    this.setMode(this.mode, false);
  }

  setMode(mode, redraw = true) {
    this.mode = mode;
    this.viewSelect.value = mode;
    for (const [k, g] of Object.entries(this.groups)) g.style.display = k === mode ? "" : "none";
    if (redraw) { this.changed(); this.v.scheduleDetail(); }
  }

  updatePanel() {
    const ex = this.pca.explained || [];
    this.explained.textContent = this.comps.map((k) => `PC${k + 1} ${ex[k] !== undefined ? (100 * ex[k]).toFixed(1) : "?"} %`).join(" · ")
      + (this.scope === "visible" ? " (visible area)" : "");
    const labels = this.v.meta.labels;
    const year = this.refYear !== null ? ` in ${labels[this.refYear]}` : "";
    this.refText.textContent = this.pinned ? `reference: the pinned pixel${year} (click it again or Esc to unpin)`
      : `reference: the pixel under the cursor${year}; click to pin it`;
    this.unpinBtn.style.display = this.pinned ? "" : "none";
  }

  // ------------------------------------------------------------------ vectors in the browser
  vectorIn(bytes, w, h, col, row) {
    if (!bytes || col < 0 || row < 0 || col >= w || row >= h) return null;
    const q = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), plane = w * h, at = row * w + col;
    if (q[at * 4] === -128) return null;
    const out = new Float32Array(this.D);
    for (let d = 0; d < this.D; d++) out[d] = q[((d >> 2) * plane + at) * 4 + (d & 3)] * this.scale[d];
    return out;
  }

  vectorAt(i, cx, cy) {   // frame i's vector at a full-resolution cell (the finer data when there is some)
    const v = this.v, m = v.meta, d = v.detail;
    if (i < 0 || i >= m.n) return null;
    if (d && cx >= d.x0 && cx < d.x1 && cy >= d.y0 && cy < d.y1) {
      const bytes = d.index === i ? d.bytes : d.other === i ? d.otherBytes : null;
      if (bytes) return this.vectorIn(bytes, d.width, d.height, Math.floor((cx - d.x0) / d.step), Math.floor((cy - d.y0) / d.step));
    }
    return this.vectorIn(v.frames[i], m.width, m.height, Math.floor(cx / m.step), Math.floor(cy / m.step));
  }

  cosine(a, b) {
    if (!a || !b) return null;
    let d = 0, na = 0, nb = 0;
    for (let k = 0; k < a.length; k++) { d += a[k] * b[k]; na += a[k] * a[k]; nb += b[k] * b[k]; }
    return na > 0 && nb > 0 ? d / Math.sqrt(na * nb) : null;
  }

  sampleCells(i, n = 4096) {   // up to ~n vectors of frame i on a regular grid of the preview
    const m = this.v.meta, f = this.v.frames[i];
    if (!f) return [];
    const stride = Math.max(1, Math.ceil(Math.sqrt((m.width * m.height) / n))), out = [];
    for (let r = 0; r < m.height; r += stride) for (let c = 0; c < m.width; c += stride) {
      const vec = this.vectorIn(f, m.width, m.height, c, r);
      if (vec) out.push([c, r, vec]);
    }
    return out;
  }

  reference() {   // the reference vector of the similarity: the pinned pixel's, else the cursor's
    const cell = this.pinned || this.cursor;
    if (!cell) return null;
    return this.vectorAt(this.refYear !== null ? this.refYear : this.v.index, cell[0], cell[1]);
  }

  otherIndex() {
    const i = this.v.index;
    return this.compare === "previous" ? i - 1 : this.compare;
  }

  value(cx, cy) {   // what the view shows at a cell, computed on the CPU
    const vec = this.vectorAt(this.v.index, cx, cy);
    if (!vec) return null;
    if (this.mode === "similarity") return this.cosine(vec, this.reference());
    if (this.mode === "change") { const c = this.cosine(vec, this.vectorAt(this.otherIndex(), cx, cy)); return c === null ? null : 1 - c; }
    return this.comps.map((k) => { let p = 0; const w = this.pca.weights[k], mean = this.pca.mean;
      for (let d = 0; d < this.D; d++) p += (vec[d] - mean[d]) * w[d]; return p; });
  }

  // ------------------------------------------------------------------ reacting
  hover(cx, cy) {
    this.cursor = [cx, cy];
    if (this.mode === "similarity" && !this.pinned) this.changed(false);
    this.scheduleSide();
    const value = this.value(cx, cy);
    if (value === null) return this.v.frames[this.v.index] ? "no embedding" : "";
    if (this.mode === "components") return value.map((p, k) => `PC${this.comps[k] + 1} ${p.toFixed(3)}`).join("  ");
    if (this.mode === "similarity") return `similarity ${value.toFixed(3)}`;
    return `change ${value.toFixed(3)} from ${this.v.meta.labels[this.otherIndex()]}`;
  }

  click(cx, cy) {
    const cell = [Math.floor(cx) + 0.5, Math.floor(cy) + 0.5];
    if (this.pinned && Math.floor(this.pinned[0]) === Math.floor(cx) && Math.floor(this.pinned[1]) === Math.floor(cy)) { this.unpin(); return; }
    this.pinned = cell;
    this.v.pin = cell;
    this.changed();
    this.updateChart();
  }

  unpin() {
    this.pinned = null; this.v.pin = null;
    this.changed(); this.updateChart();
  }

  key(e) {
    if (e.key === "Escape" && this.pinned) { this.unpin(); return true; }
    return false;
  }

  scheduleSide() {   // the profile and the chart follow the cursor, once per animation frame
    if (this.sideRaf) return;
    this.sideRaf = requestAnimationFrame(() => { this.sideRaf = null; this.drawProfile(); this.updateChart(); });
  }

  changed(legend = true) {   // the reference, the components or the year changed
    this.rowsDirty = true;
    this.updateRange();
    this.updatePanel();
    if (legend) this.drawLegend();
    this.v.render();
  }

  updateRange() {
    const v = this.v;
    if (this.mode === "similarity") {
      const ref = this.reference(), values = [];
      if (ref) for (const [, , vec] of this.sampleCells(v.index)) { const c = this.cosine(vec, ref); if (c !== null) values.push(c); }
      const lo = values.length ? percentile(values, 0.02) : 0;
      const next = [Math.min(lo, 0.999), 1];
      const moved = !this.range || Math.abs(next[0] - this.range[0]) > 1e-3;
      this.range = next;
      if (moved && this.mode === "similarity") this.drawLegend();
    } else if (this.mode === "change") {
      const i = v.index, j = this.otherIndex(), key = `${i},${j}`;
      if (!this.changeRanges.has(key) && v.frames[i] && v.frames[j]) {
        const m = v.meta, values = [];
        for (const [c, r, vec] of this.sampleCells(i)) {
          const cos = this.cosine(vec, this.vectorIn(v.frames[j], m.width, m.height, c, r));
          if (cos !== null) values.push(1 - cos);
        }
        if (values.length) this.changeRanges.set(key, Math.max(percentile(values, 0.98), 1e-3));
      }
      const hi = this.changeRanges.get(key);
      const next = [0, hi || 0.5];
      const moved = !this.range || Math.abs(next[1] - this.range[1]) > 1e-6;
      this.range = next;
      if (moved) this.drawLegend();
    }
  }

  fitVisible() {   // the components of what is on screen
    clearTimeout(this.pcaTimer);
    this.pcaTimer = setTimeout(async () => {
      const v = this.v, m = v.meta;
      if (this.scope !== "visible" || !v.cssSize) return;
      const [x0, y0] = v.toCell(0, 0), [x1, y1] = v.toCell(...v.cssSize);
      const win = [Math.max(0, x0), Math.max(0, y0), Math.min(m.full_width, x1), Math.min(m.full_height, y1)].map(Math.round);
      if (this.pcaWindow && win.every((x, k) => x === this.pcaWindow[k])) return;
      this.pcaWindow = win;
      try {
        const { content } = await v.transport.request({ type: "pca", window: win });
        if (this.scope !== "visible" || v.emb !== this) return;
        this.pca = content;
        this.changed();
      } catch (err) { v.readout.textContent = String(err.message || err); }
    }, 150);
  }

  viewChanged() { if (this.scope === "visible") this.fitVisible(); }

  detailRequest() {
    const j = this.otherIndex();
    return this.mode === "change" && j >= 0 && j < this.v.meta.n ? { other: j } : {};
  }

  // ------------------------------------------------------------------ WebGL
  setupGL(gl) {
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    if (!this.v.embProgram) {
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, EMB_FS));
      gl.bindAttribLocation(p, 0, "corner");
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      const u = {};
      for (const n of ["rect", "view", "canvas", "A", "B", "rows", "luts", "mode", "layers", "raw", "offset", "span", "range", "refNorm"])
        u[n] = gl.getUniformLocation(p, n);
      const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      this.v.embProgram = { p, u, buf };
    }
    this.gp = this.v.embProgram;
    const tex = (unit, target) => { const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(target, t);
      for (const [k, val] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(target, k, val);
      return t; };
    const m = this.v.meta, A = gl.TEXTURE_2D_ARRAY;
    this.texA = tex(3, A); gl.texImage3D(A, 0, gl.RGBA8I, m.width, m.height, this.L, 0, gl.RGBA_INTEGER, gl.BYTE, null);
    this.texB = tex(4, A); gl.texImage3D(A, 0, gl.RGBA8I, m.width, m.height, this.L, 0, gl.RGBA_INTEGER, gl.BYTE, null);
    this.texDA = tex(5, A); this.texDB = tex(6, A);
    this.rowsTex = tex(7, gl.TEXTURE_2D);
    this.lutTex = tex(8, gl.TEXTURE_2D);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, this.lutBytes);
    this.uploaded = { A: -1, B: -1 };
    this.rowsDirty = true;
  }

  updateRows(gl) {
    const D = this.D, W = this.L * 4, rows = new Float32Array(W * 5), s = this.scale;
    const ref = this.mode === "similarity" ? this.reference() : null;
    let norm = 0;
    if (ref) for (let d = 0; d < D; d++) { rows[d] = ref[d] * s[d]; norm += ref[d] * ref[d]; }
    const offset = [0, 0, 0], span = [1, 1, 1];
    this.comps.forEach((k, c) => {
      const w = this.pca.weights[k];
      let mw = 0;
      for (let d = 0; d < D; d++) { rows[(c + 1) * W + d] = w[d] * s[d]; mw += this.pca.mean[d] * w[d]; }
      offset[c] = mw + this.pca.lo[k];
      span[c] = Math.max(this.pca.hi[k] - this.pca.lo[k], 1e-9);
    });
    for (let d = 0; d < D; d++) rows[4 * W + d] = s[d] * s[d];
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, this.rowsTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, this.L, 5, 0, gl.RGBA, gl.FLOAT, rows);
    this.refNorm = Math.sqrt(norm);
    this.offset = offset; this.span = span;
    this.rowsDirty = false;
  }

  drawGL(gl) {
    const v = this.v, m = v.meta, i = v.index, f = v.frames[i];
    if (!f || !this.gp) return;
    let mode = EMB_MODES[this.mode], j = -1;
    if (mode === 2) {
      j = this.otherIndex();
      if (j < 0 || j >= m.n || !v.frames[j]) return;
    }
    const { p, u, buf } = this.gp;
    gl.useProgram(p);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    if (this.rowsDirty) this.updateRows(gl);
    if (mode === 1 && !(this.refNorm > 0)) mode = 0;   // no reference yet: the components
    const upload = (unit, texture, slot, index) => {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
      if (this.uploaded[slot] !== index) {
        const b = v.frames[index];
        gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, 0, m.width, m.height, this.L, gl.RGBA_INTEGER, gl.BYTE,
          new Int8Array(b.buffer, b.byteOffset, b.byteLength));
        this.uploaded[slot] = index;
      }
    };
    upload(3, this.texA, "A", i);
    if (mode === 2) upload(4, this.texB, "B", j);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, this.rowsTex);
    gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.uniform3f(u.view, v.view.ox, v.view.oy, v.view.s);
    gl.uniform2f(u.canvas, ...v.cssSize);
    gl.uniform1i(u.rows, 7); gl.uniform1i(u.luts, 8);
    gl.uniform1i(u.mode, mode); gl.uniform1i(u.layers, this.L); gl.uniform1i(u.raw, this.raw ? 1 : 0);
    gl.uniform3f(u.offset, ...this.offset); gl.uniform3f(u.span, ...this.span);
    const range = this.raw ? [-1, 2] : (this.range || [0, 1]);
    gl.uniform2f(u.range, range[0], range[1]);
    gl.uniform1f(u.refNorm, this.refNorm || 0);
    gl.uniform1i(u.A, 3); gl.uniform1i(u.B, 4);
    gl.uniform4f(u.rect, 0, 0, m.width * m.step, m.height * m.step);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    const d = v.detail;
    if (d && d.index === i && d.dataA && (mode !== 2 || (d.other === j && d.dataB))) {
      gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texDA);
      gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texDB);
      gl.uniform1i(u.A, 5); gl.uniform1i(u.B, 6);
      gl.uniform4f(u.rect, d.x0, d.y0, d.x1, d.y1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }

  setDetail(d) {   // a zoomed-in window of the year shown (and of the year it is compared with)
    const gl = this.v.gl, A = gl.TEXTURE_2D_ARRAY;
    const put = (unit, texture, bytes) => {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(A, texture);
      gl.texImage3D(A, 0, gl.RGBA8I, d.width, d.height, this.L, 0, gl.RGBA_INTEGER, gl.BYTE,
        new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    };
    put(5, this.texDA, d.bytes); d.dataA = true;
    if (d.otherBytes) { put(6, this.texDB, d.otherBytes); d.dataB = true; }
    if (this.mode === "similarity") this.rowsDirty = true;   // the reference may now come from finer cells
  }

  probe(cx, cy) {   // the value the shader computes at a cell (for tests): drawn in 16 bits and read back
    const v = this.v, gl = v.gl, dpr = window.devicePixelRatio || 1;
    this.raw = true;
    try {
      v.draw();
      const px = Math.floor((cx - v.view.ox) * v.view.s * dpr), py = Math.floor((cy - v.view.oy) * v.view.s * dpr);
      const out = new Uint8Array(4);
      gl.readPixels(px, v.canvas.height - 1 - py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);
      if (!out[3]) return null;
      return -1 + 3 * ((out[0] * 256 + out[1]) / 65535);
    } finally { this.raw = false; v.render(); }
  }

  // ------------------------------------------------------------------ legend, profile, chart
  drawLegend() {
    const v = this.v, legend = v.legend;
    legend.replaceChildren();
    if (this.mode === "components" || (this.mode === "similarity" && !this.reference())) {
      const ex = this.pca.explained || [];
      ["#e53935", "#43a047", "#1e88e5"].forEach((color, c) => legend.append(h("span", { class: "zv-chip" },
        h("i", { style: `background:${color}` }), `PC${this.comps[c] + 1}` + (ex[this.comps[c]] !== undefined
          ? ` (${(100 * ex[this.comps[c]]).toFixed(1)} %)` : ""))));
      if (this.mode === "similarity") legend.append(h("span", { class: "zv-unit" }, "move the mouse over the map"));
      v.addOpacity();
      return;
    }
    const [lo, hi] = this.range || [0, 1], row = this.mode === "similarity" ? 0 : 1;
    const ticks = h("div", { class: "zv-ticks" });
    for (const t of niceTicks(lo, hi, 5)) {
      const pos = (100 * (t - lo)) / ((hi - lo) || 1);
      if (pos < -0.5 || pos > 100.5) continue;
      const shift = pos < 6 ? "0" : pos > 94 ? "-100%" : "-50%";
      ticks.append(h("span", { style: `left:${pos}%;transform:translateX(${shift})` }, fmtTick(t, hi - lo)));
    }
    const ramp = h("canvas", { class: "zv-ramp", width: 255, height: 1 });
    const ctx = ramp.getContext("2d"), img = ctx.createImageData(255, 1);
    img.data.set(this.lutBytes.subarray(row * 1024 + 4, row * 1024 + 1024));
    ctx.putImageData(img, 0, 0);
    const unit = this.mode === "similarity" ? "cosine similarity to the reference"
      : `change (1 − cosine) from ${this.compare === "previous" ? "the previous year" : v.meta.labels[this.compare]}`;
    legend.append(h("div", { class: "zv-rampwrap" }, ramp, ticks), h("span", { class: "zv-unit" }, unit));
    v.addOpacity();
  }

  drawProfile() {
    const vecs = [];
    if (this.cursor) vecs.push(["cursor", "#1f77b4", this.vectorAt(this.v.index, ...this.cursor)]);
    if (this.pinned) vecs.push(["pin", PIN_COLOR, this.vectorAt(this.refYear !== null ? this.refYear : this.v.index, ...this.pinned)]);
    const shown = vecs.filter(([, , vec]) => vec);
    if (!shown.length) { this.profile.replaceChildren(); return; }
    const W = 220, H = 96, M = 4, NS = "http://www.w3.org/2000/svg";
    let lo = Infinity, hi = -Infinity;
    for (const [, , vec] of shown) for (const x of vec) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
    if (hi === lo) { hi += 1; lo -= 1; }
    const sx = (d) => M + (d / Math.max(1, this.D - 1)) * (W - 2 * M), sy = (x) => H - M - ((x - lo) / (hi - lo)) * (H - 2 * M);
    const svg = document.createElementNS(NS, "svg");
    for (const [k, val] of Object.entries({ width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "zv-svg" })) svg.setAttribute(k, val);
    const zero = document.createElementNS(NS, "line");
    for (const [k, val] of Object.entries({ x1: M, x2: W - M, y1: sy(0), y2: sy(0), class: "zv-grid" })) zero.setAttribute(k, val);
    svg.append(zero);
    for (const [, color, vec] of shown) {
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", Array.from(vec, (x, d) => `${d ? "L" : "M"}${sx(d).toFixed(1)},${sy(x).toFixed(1)}`).join(""));
      path.setAttribute("fill", "none"); path.setAttribute("stroke", color); path.setAttribute("stroke-width", 1.2);
      svg.append(path);
    }
    this.profile.replaceChildren(h("div", { class: "zv-emb-note" },
      "latent profile: " + shown.map(([name]) => name).join(", ") + ` (${this.D} dimensions)`), svg);
  }

  updateChart() {   // the similarity of the cursor's (and the pin's) pixel over the years to the reference
    const v = this.v, m = v.meta, cell = this.cursor || this.pinned, ref = this.reference();
    if (!m || !cell || !ref || m.n < 2) return;
    const over = (c) => m.labels.map((_, i) => { const s = this.cosine(this.vectorAt(i, c[0], c[1]), ref); return s; });
    const series = [];
    if (this.cursor) series.push({ name: "cursor", y: over(this.cursor) });
    if (this.pinned) series.push({ name: "pin", y: over(this.pinned) });
    v.pixel = { x: this.e.x, series, overlays: [], labels: m.labels, is_time: this.e.is_time,
      world: v.toWorld(cell[0], cell[1]) };
    v.drawChart();
  }
}
