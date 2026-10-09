// zeit.plot viewer: dense raster time series at the browser's frame rate.
//
// Frames arrive as colour indices (uint8; 0 = NoData) and are painted by a WebGL2 shader
// through a 256-colour lookup table, so the kernel never renders and, once a frame is in
// the browser, paging through time does not talk to the kernel at all.
//
// `mount(el, transport)` builds the viewer; `transport.request(req)` resolves to
// {content, buffers}. The notebook widget and the standalone window give different
// transports to the same viewer.

const VS = `#version 300 es
in vec2 corner;                  // 0..1 quad corner
uniform vec4 rect;               // x0, y0, x1, y1 of the texture, in full-resolution cells
uniform vec3 view;               // ox, oy (cell at the canvas origin), scale (css px per cell)
uniform vec2 canvas;             // canvas size in css px
out vec2 uv;
void main() {
  vec2 cell = mix(rect.xy, rect.zw, corner);
  vec2 px = (cell - view.xy) * view.z;
  vec2 clip = px / canvas * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  uv = corner;
}`;

const FS = `#version 300 es
precision highp float;
precision highp usampler2D;
uniform highp usampler2D data;   // colour indices (R8UI) or RGB (RGB8UI)
uniform sampler2D lut;
uniform int rgb;
in vec2 uv; out vec4 color;
void main() {
  ivec2 size = textureSize(data, 0);
  ivec2 at = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
  uvec4 v = texelFetch(data, at, 0);
  if (rgb == 1) {
    if (v.r == 0u || v.g == 0u || v.b == 0u) discard;
    color = vec4((vec3(v.rgb) - 1.0) / 254.0, 1.0);
  } else {
    if (v.r == 0u) discard;
    color = texelFetch(lut, ivec2(int(v.r), 0), 0);
  }
}`;

const CSS_ID = "zeit-viewer-css";
const MEMORY_BUDGET = 400e6;   // bytes of frames kept in the browser

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

async function inflate(buffer, compressed) {
  const bytes = new Uint8Array(buffer.buffer || buffer, buffer.byteOffset || 0, buffer.byteLength);
  if (!compressed) return bytes.slice();
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function formatValue(style, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "NoData";
  if (style.kind === "categorical") {
    const c = style.classes.find((k) => k[0] === value);
    return c ? c[1] : String(value);
  }
  if (style.kind === "years") return String(Math.round(value));
  const span = Math.abs(style.vmax - style.vmin) || 1;
  const digits = Math.max(0, Math.min(6, 2 - Math.floor(Math.log10(span))));
  return "≈ " + value.toFixed(digits);
}

function niceTicks(lo, hi, n) {
  const span = hi - lo || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((st) => span / st <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

function fmtTick(v, span) {
  const digits = Math.max(0, Math.min(6, 1 - Math.floor(Math.log10(Math.abs(span) || 1))));
  return Math.abs(v) >= 1e5 ? v.toExponential(1) : v.toFixed(digits);
}

function yearTicks(t0, t1, max) {
  const y0 = new Date(t0).getUTCFullYear(), y1 = new Date(t1).getUTCFullYear() + 1;
  const step = [1, 2, 5, 10, 20, 50].find((st) => (y1 - y0) / st <= Math.max(2, max)) || 100;
  const out = [];
  for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
    const t = Date.UTC(y, 0, 1);
    if (t >= t0 && t <= t1) out.push([t, String(y)]);
  }
  if (out.length < 2) {   // short series: every third month
    const d = new Date(t0);
    for (let k = 0; k < 36; k++) {
      const t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + k, 1);
      if (t > t1) break;
      if (t >= t0 && k % 3 === 0) out.push([t, new Date(t).toISOString().slice(0, 7)]);
    }
  }
  return out;
}

// Bilinear interpolation in a regular control grid {nx, ny, x0, x1, y0, y1, a, b}: (x, y) -> [a, b].
// Beyond the grid it extrapolates from the edge cells.
function interp(g, x, y) {
  const fx = ((x - g.x0) / (g.x1 - g.x0)) * (g.nx - 1), fy = ((y - g.y0) / (g.y1 - g.y0)) * (g.ny - 1);
  const i = Math.min(Math.max(Math.floor(fx), 0), g.nx - 2), j = Math.min(Math.max(Math.floor(fy), 0), g.ny - 2);
  const tx = fx - i, ty = fy - j, k = j * g.nx + i;
  const at = (arr) => (arr[k] * (1 - tx) + arr[k + 1] * tx) * (1 - ty) + (arr[k + g.nx] * (1 - tx) + arr[k + g.nx + 1] * tx) * ty;
  return [at(g.a), at(g.b)];
}

const tileLon = (x, z) => (x / 2 ** z) * 360 - 180;
const tileLat = (y, z) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / 2 ** z))) * 180) / Math.PI;
const lonTile = (lon, z) => ((lon + 180) / 360) * 2 ** z;
const latTile = (lat, z) => {
  const r = (Math.max(Math.min(lat, 85.0511), -85.0511) * Math.PI) / 180;
  return ((1 - Math.asinh(Math.tan(r)) / Math.PI) / 2) * 2 ** z;
};

export class Viewer {
  constructor(el, transport, options = {}) {
    this.el = el;
    this.transport = transport;
    this.options = options;
    this.frames = [];
    this.lru = [];
    this.index = 0;
    this.playing = false;
    this.fps = options.fps || 8;
    this.detail = null;
    this.listeners = [];
    this.layers = [];   // extra drawing (basemap, vectors, overlays): {draw2d(ctx, viewer), below}
    this.build();
  }

  // ------------------------------------------------------------------ DOM
  build() {
    if (!document.getElementById(CSS_ID) && this.options.css) {
      document.head.append(h("style", { id: CSS_ID }, this.options.css));
    }
    this.titleEl = h("div", { class: "zv-title" });
    this.varSelect = h("select", { class: "zv-var", onchange: () => this.select(this.varSelect.value) });
    this.below = h("canvas", { class: "zv-below" });
    this.canvas = h("canvas", { class: "zv-map" });
    this.above = h("canvas", { class: "zv-above" });
    this.attribution = h("div", { class: "zv-attribution" });
    this.stage = h("div", { class: "zv-stage" }, this.below, this.canvas, this.above, this.attribution);
    this.opacity = h("input", { class: "zv-opacity", type: "range", min: 0, max: 1, step: 0.05, value: 1,
      title: "Opacity of the data over the basemap" });
    this.opacity.addEventListener("input", () => { this.canvas.style.opacity = this.opacity.value; });
    this.opacityBox = h("label", { class: "zv-opacitybox" }, "opacity ", this.opacity);
    this.readout = h("div", { class: "zv-readout" }, " ");
    this.playBtn = h("button", { class: "zv-play", title: "Play / pause (space)", onclick: () => this.toggle() }, "▶");
    this.slider = h("input", { class: "zv-slider", type: "range", min: 0, max: 0, value: 0 });
    this.slider.addEventListener("input", () => this.show(+this.slider.value));
    this.label = h("span", { class: "zv-label" });
    this.speed = h("select", { class: "zv-speed", title: "Frames per second" });
    for (const f of [1, 2, 4, 8, 12, 24, 60]) {
      const o = h("option", { value: f }, `${f} fps`); if (f === this.fps) o.selected = true; this.speed.append(o);
    }
    this.speed.addEventListener("change", () => { this.fps = +this.speed.value; });
    this.progress = h("div", { class: "zv-progress" }, h("div", { class: "zv-bar" }));
    this.controls = h("div", { class: "zv-controls" }, this.playBtn, this.slider, this.label, this.speed);
    this.tiles = new Map();
    this.legend = h("div", { class: "zv-legend" });
    this.side = h("div", { class: "zv-side" });
    this.root = h("div", { class: "zv-root", tabindex: 0 },
      h("div", { class: "zv-head" }, this.titleEl, this.varSelect),
      h("div", { class: "zv-body" }, h("div", { class: "zv-main" }, this.stage, this.progress, this.controls,
        this.legend, this.readout, this.chart = h("div", { class: "zv-chart" })), this.side));
    this.el.append(this.root);
    this.root.addEventListener("keydown", (e) => this.key(e));
    this.bindPointer();
    new ResizeObserver(() => this.resize()).observe(this.stage);
    this.on("click", (cell) => this.inspect(cell));
    this.on("frame", () => this.drawChart());
    this.layers.push({ draw2d: (ctx, v) => v.drawPin(ctx) });
    this.layers.push({ below: true, draw2d: (ctx, v) => v.drawBasemap(ctx) });
    this.layers.push({ draw2d: (ctx, v) => v.drawVectors(ctx) });
  }

  // ------------------------------------------------------------------ pixel inspector
  async inspect([cx, cy]) {
    const m = this.meta;
    if (!m || cx < 0 || cy < 0 || cx >= m.full_width || cy >= m.full_height) return;
    this.pin = [Math.floor(cx) + 0.5, Math.floor(cy) + 0.5];
    this.render();
    this.chart.classList.add("zv-busy");
    try {
      const { content } = await this.transport.request({ type: "pixel", x: Math.floor(cx), y: Math.floor(cy) });
      this.pixel = content;
      this.drawChart();
    } catch (err) {
      this.chart.textContent = String(err.message || err);
    } finally { this.chart.classList.remove("zv-busy"); }
  }

  drawPin(ctx) {
    if (!this.pin) return;
    const x = (this.pin[0] - this.view.ox) * this.view.s, y = (this.pin[1] - this.view.oy) * this.view.s;
    ctx.save();
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,0.6)";
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff";
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI); ctx.stroke();
    ctx.restore();
  }

  drawChart() {
    const data = this.pixel;
    if (!data) return;
    const W = Math.max(240, this.chart.clientWidth || 600), H = 200, M = { l: 56, r: 12, t: 24, b: 26 };
    const NS = "http://www.w3.org/2000/svg";
    const el = (tag, attrs = {}, text) => { const n = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (text !== undefined) n.textContent = text; return n; };
    const xs = [...data.x], ys = [];
    for (const s of data.series) for (const v of s.y) if (v !== null) ys.push(v);
    for (const o of data.overlays) {
      if (o.kind === "line") { xs.push(...o.x); for (const v of o.y) if (v !== null) ys.push(v); }
      else if (o.kind === "vline") xs.push(o.x); else if (o.kind === "span") xs.push(o.x0, o.x1);
    }
    if (!ys.length) { this.chart.textContent = "no data at this pixel"; return; }
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    const pad = (y1 - y0) * 0.08 || 1; y0 -= pad; y1 += pad;
    const sx = (x) => M.l + ((x - x0) / (x1 - x0)) * (W - M.l - M.r);
    const sy = (y) => H - M.b - ((y - y0) / (y1 - y0)) * (H - M.t - M.b);
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "zv-svg" });
    for (const v of niceTicks(y0, y1, 5)) {
      svg.append(el("line", { x1: M.l, x2: W - M.r, y1: sy(v), y2: sy(v), class: "zv-grid" }));
      svg.append(el("text", { x: M.l - 6, y: sy(v) + 4, "text-anchor": "end", class: "zv-tick" }, fmtTick(v, y1 - y0)));
    }
    const xt = data.is_time ? yearTicks(x0, x1, Math.floor((W - M.l) / 60))
      : niceTicks(x0, x1, 8).map((v) => [v, data.labels[Math.round(v)] || ""]);
    for (const [v, t] of xt) svg.append(el("text", { x: sx(v), y: H - 8, "text-anchor": "middle", class: "zv-tick" }, t));
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: "zv-axis" }));
    const legend = [];
    for (const o of data.overlays) if (o.kind === "span") {
      svg.append(el("rect", { x: sx(o.x0), y: M.t, width: Math.max(2, sx(o.x1) - sx(o.x0)), height: H - M.t - M.b,
        fill: o.color, "fill-opacity": 0.15 }));
      if (o.label) legend.push([o.label, o.color]);
    }
    const colors = ["#1f77b4", "#2ca02c", "#9467bd"];
    const path = (xsArr, ysArr) => { let d = "", pen = false;
      xsArr.forEach((x, i) => { const v = ysArr[i]; if (v === null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(v).toFixed(1)}`; pen = true; }); return d; };
    data.series.forEach((s, k) => {
      const c = colors[k % colors.length];
      svg.append(el("path", { d: path(data.x, s.y), fill: "none", stroke: c, "stroke-width": 1.2, "stroke-opacity": 0.85 }));
      s.y.forEach((v, i) => { if (v !== null) svg.append(el("circle", { cx: sx(data.x[i]), cy: sy(v), r: 2.2, fill: c })); });
      legend.push([s.name, c]);
    });
    for (const o of data.overlays) {
      if (o.kind === "line") {
        svg.append(el("path", { d: path(o.x, o.y), fill: "none", stroke: o.color, "stroke-width": 2,
          "stroke-dasharray": o.dashed ? "5 4" : "none" }));
        if (o.markers) o.x.forEach((x, i) => { if (o.y[i] !== null) svg.append(el("circle", { cx: sx(x), cy: sy(o.y[i]), r: 3.5, fill: o.color })); });
      } else if (o.kind === "vline") {
        svg.append(el("line", { x1: sx(o.x), x2: sx(o.x), y1: M.t, y2: H - M.b, stroke: o.color, "stroke-width": 1.5,
          "stroke-dasharray": o.dashed ? "4 3" : "none" }));
      }
      if (o.label && o.kind !== "span") legend.push([o.label, o.color]);
    }
    const cur = data.x[this.index];
    if (cur !== undefined) svg.append(el("line", { x1: sx(cur), x2: sx(cur), y1: M.t, y2: H - M.b, class: "zv-now" }));
    const [wx, wy] = data.world;
    svg.append(el("text", { x: M.l, y: 14, class: "zv-ctitle" }, `pixel  x ${wx.toFixed(5)}  y ${wy.toFixed(5)}`));
    let lx = W - M.r;
    for (const [text, color] of legend.slice().reverse()) {
      lx -= 24 + text.length * 6.2;
      svg.append(el("rect", { x: lx, y: 5, width: 10, height: 10, fill: color, "fill-opacity": 0.85 }));
      svg.append(el("text", { x: lx + 14, y: 14, class: "zv-tick" }, text));
    }
    svg.addEventListener("click", (e) => {   // go to the frame nearest to the click
      const r = svg.getBoundingClientRect(), x = x0 + ((e.clientX - r.left - M.l) / (W - M.l - M.r)) * (x1 - x0);
      let best = 0;
      data.x.forEach((v, i) => { if (Math.abs(v - x) < Math.abs(data.x[best] - x)) best = i; });
      this.show(best);
    });
    this.chart.replaceChildren(svg);
  }

  // ------------------------------------------------------------------ data
  async start() {
    const { content, buffers } = await this.transport.request({ type: "meta" });
    this.setMeta(content, buffers);
  }

  async select(variable) {
    this.stop();
    const { content, buffers } = await this.transport.request({ type: "select", var: variable });
    this.setMeta(content, buffers);
  }

  setMeta(meta, buffers) {
    this.meta = meta;
    this.generation = (this.generation || 0) + 1;
    this.frames = new Array(meta.n);
    this.lru = [];
    this.detail = null;
    this.index = Math.min(this.index, meta.n - 1);
    this.capacity = Math.max(3, Math.floor(MEMORY_BUDGET / Math.max(1, meta.frame_bytes)));
    this.titleEl.textContent = meta.variables.length > 1 && meta.title === meta.var ? "" : (meta.title || "");
    this.varSelect.replaceChildren(...meta.variables.map((v) => {
      const o = h("option", { value: v }, v); if (v === meta.var) o.selected = true; return o; }));
    this.varSelect.style.display = meta.variables.length > 1 ? "" : "none";
    this.slider.max = meta.n - 1;
    this.slider.value = this.index;
    this.controls.style.display = meta.n > 1 ? "" : "none";
    this.setupGL(new Uint8Array(buffers[0].buffer || buffers[0], buffers[0].byteOffset || 0, 1024));
    this.drawLegend();
    this.view = null;
    this.pin = null; this.pixel = null; this.chart.replaceChildren();
    this.attribution.textContent = meta.basemap ? meta.basemap.attribution : "";
    this.attribution.style.display = meta.basemap && meta.basemap.attribution ? "" : "none";
    this.opacity.value = meta.opacity ?? 1;
    this.canvas.style.opacity = this.opacity.value;
    this.vectors = null;
    if (meta.has_vectors) this.loadVectors(this.generation);
    this.resize();          // the first ResizeObserver call may have come before the metadata
    this.label.textContent = meta.labels[this.index] || "";
    this.emit("meta", meta);
    this.load(this.generation);
  }

  // Preload: the current frame first, then outwards from it; at most two requests in flight;
  // beyond the memory budget the frames farthest from the current one are dropped (LRU).
  async load(generation) {
    if (this.loading === generation) return;   // one loader at a time; it follows this.index
    this.loading = generation;
    const n = this.meta.n;
    const batch = Math.max(1, Math.min(16, Math.floor(8e6 / Math.max(1, this.meta.frame_bytes))));
    const missing = () => {
      const order = [], reach = Math.min(n, this.capacity);
      for (let d = 0; order.length < reach && d < n; d++)
        for (const i of d ? [this.index + d, this.index - d] : [this.index]) if (i >= 0 && i < n) order.push(i);
      return order.filter((i) => !this.frames[i]);
    };
    const pending = new Map();
    const fetchFrom = async (start) => {
      const { content, buffers } = await this.transport.request({ type: "frames", start, count: Math.min(batch, n - start) });
      if (generation !== this.generation) return;
      for (let k = 0; k < buffers.length; k++) this.store(content.start + k, await inflate(buffers[k], this.meta.compressed));
      this.updateProgress();
      if (this.index >= content.start && this.index < content.start + content.count) this.render();
    };
    try {
      while (generation === this.generation) {
        const todo = missing();
        if (!todo.length && !pending.size) break;
        for (const i of todo) {
          if (pending.size >= 2) break;
          const start = Math.floor(i / batch) * batch;
          if (!pending.has(start)) pending.set(start, fetchFrom(start).finally(() => pending.delete(start)));
        }
        if (pending.size) await Promise.race(pending.values()); else break;
      }
    } finally {
      if (this.loading === generation) this.loading = null;
      if (this.meta) this.updateProgress();
    }
  }

  store(i, bytes) {
    if (!this.frames[i]) this.lru.push(i);
    this.frames[i] = bytes;
    while (this.lru.length > this.capacity) {
      // drop the stored frame farthest from the current one
      let far = 0;
      for (let k = 1; k < this.lru.length; k++)
        if (Math.abs(this.lru[k] - this.index) > Math.abs(this.lru[far] - this.index)) far = k;
      const [gone] = this.lru.splice(far, 1);
      this.frames[gone] = undefined;
    }
  }

  updateProgress() {
    const have = this.frames.reduce((a, f) => a + (f ? 1 : 0), 0);
    const want = Math.min(this.meta.n, this.capacity);
    const bar = this.progress.firstChild;
    bar.style.width = `${Math.min(100, (100 * have) / want)}%`;
    this.progress.style.visibility = have >= want ? "hidden" : "visible";
  }

  // ------------------------------------------------------------------ WebGL
  setupGL(lutBytes) {
    const gl = this.gl || (this.gl = this.canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false }));
    if (!gl) { this.readout.textContent = "WebGL2 is not available in this browser"; return; }
    if (!this.program) {
      const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(p); this.program = p; gl.useProgram(p);
      const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(p, "corner"); gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      this.u = {};
      for (const n of ["rect", "view", "canvas", "data", "lut", "rgb"]) this.u[n] = gl.getUniformLocation(p, n);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    }
    const tex = (unit) => { const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST],
        [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
      return t; };
    for (const t of [this.lutTex, this.baseTex, this.detailTex]) if (t) gl.deleteTexture(t);
    this.lutTex = tex(2);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lutBytes);
    const rgb = this.meta.rgb;
    this.texFormat = rgb ? [gl.RGB8UI, gl.RGB_INTEGER] : [gl.R8UI, gl.RED_INTEGER];
    this.baseTex = tex(0);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.texFormat[0], this.meta.width, this.meta.height, 0, this.texFormat[1],
      gl.UNSIGNED_BYTE, null);
    this.detailTex = tex(1);
    gl.useProgram(this.program);
    gl.uniform1i(this.u.lut, 2);
    gl.uniform1i(this.u.rgb, rgb ? 1 : 0);
    this.uploaded = -1;
  }

  // ------------------------------------------------------------------ view
  resize() {
    const r = this.stage.getBoundingClientRect();
    if (!r.width || !this.meta) return;
    const dpr = window.devicePixelRatio || 1;
    for (const c of [this.below, this.canvas, this.above]) {
      c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr);
    }
    this.cssSize = [r.width, r.height];
    if (!this.view) this.fit();
    this.render();
  }

  fit() {
    if (!this.meta) return;
    const r = this.stage.getBoundingClientRect();
    const W = this.meta.full_width, H = this.meta.full_height;
    const s = Math.min(r.width / W, r.height / H) || 1;
    this.view = { ox: -(r.width / s - W) / 2, oy: -(r.height / s - H) / 2, s };
    this.render();
  }

  toCell(px, py) { return [px / this.view.s + this.view.ox, py / this.view.s + this.view.oy]; }

  toWorld(cx, cy) {   // full-resolution cell -> map coordinates (x, y)
    const [l, r, b, t] = this.meta.extent;
    return [l + (cx / this.meta.full_width) * (r - l), t - (cy / this.meta.full_height) * (t - b)];
  }

  toPixel(wx, wy) {   // map coordinates -> css px on the canvas
    const [l, r, b, t] = this.meta.extent;
    const cx = ((wx - l) / (r - l)) * this.meta.full_width, cy = ((t - wy) / (t - b)) * this.meta.full_height;
    return [(cx - this.view.ox) * this.view.s, (cy - this.view.oy) * this.view.s];
  }

  bindPointer() {
    let drag = null;
    this.above.addEventListener("pointerdown", (e) => {
      drag = { x: e.offsetX, y: e.offsetY, ox: this.view.ox, oy: this.view.oy, moved: false };
      this.above.setPointerCapture(e.pointerId);
    });
    this.above.addEventListener("pointermove", (e) => {
      if (drag) {
        const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        this.view.ox = drag.ox - dx / this.view.s; this.view.oy = drag.oy - dy / this.view.s;
        this.render(); this.scheduleDetail();
      }
      this.hover(e.offsetX, e.offsetY);
    });
    this.above.addEventListener("pointerup", (e) => {
      if (drag && !drag.moved) this.emit("click", this.toCell(e.offsetX, e.offsetY));
      drag = null;
    });
    this.above.addEventListener("pointerleave", () => { this.readout.textContent = " "; });
    this.above.addEventListener("dblclick", () => this.fit());
    this.above.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [cx, cy] = this.toCell(e.offsetX, e.offsetY);
      const s = Math.min(Math.max(this.view.s * Math.exp(-e.deltaY * 0.0015), 1e-3), 64);
      this.view = { s, ox: cx - e.offsetX / s, oy: cy - e.offsetY / s };
      this.render(); this.scheduleDetail();
    }, { passive: false });
  }

  key(e) {
    if (e.key === " ") { e.preventDefault(); this.toggle(); }
    else if (e.key === "ArrowRight") this.show(Math.min(this.meta.n - 1, this.index + 1));
    else if (e.key === "ArrowLeft") this.show(Math.max(0, this.index - 1));
    else if (e.key === "Home") this.show(0);
    else if (e.key === "End") this.show(this.meta.n - 1);
  }

  hover(px, py) {
    if (!this.meta) return;
    const [cx, cy] = this.toCell(px, py);
    const m = this.meta;
    if (cx < 0 || cy < 0 || cx >= m.full_width || cy >= m.full_height) { this.readout.textContent = " "; return; }
    const [wx, wy] = this.toWorld(cx, cy);
    let text = `${m.labels[this.index]} · x ${wx.toFixed(5)}  y ${wy.toFixed(5)}`;
    const value = this.valueAt(cx, cy);
    if (value !== undefined) text += ` · ${formatValue(m.style, value)}`;
    this.readout.textContent = text;
    this.emit("hover", { cell: [cx, cy], world: [wx, wy], value });
  }

  valueAt(cx, cy) {   // value from the frames in the browser (detail when available)
    const m = this.meta, d = this.detail;
    let code;
    if (d && d.index === this.index && cx >= d.x0 && cx < d.x1 && cy >= d.y0 && cy < d.y1) {
      const col = Math.floor((cx - d.x0) / d.step), row = Math.floor((cy - d.y0) / d.step);
      code = d.bytes[(row * d.width + col) * (m.rgb ? 3 : 1)];
    } else {
      const f = this.frames[this.index]; if (!f) return undefined;
      const col = Math.floor(cx / m.step), row = Math.floor(cy / m.step);
      code = f[(row * m.width + col) * (m.rgb ? 3 : 1)];
    }
    if (m.rgb) return undefined;
    if (!code) return null;
    const s = m.style;
    if (s.kind === "categorical") return s.classes[code - 1] ? s.classes[code - 1][0] : null;
    return s.vmin + ((code - 1) * (s.vmax - s.vmin)) / 254;
  }

  // ------------------------------------------------------------------ time
  show(i) {
    this.index = i;
    this.slider.value = i;
    this.label.textContent = this.meta.labels[i] || "";
    if (this.frames[i]) this.render(); else this.load(this.generation);
    if (!this.playing) this.scheduleDetail();
    this.emit("frame", i);
  }

  toggle() { if (this.playing) this.stop(); else this.play(); }

  play() {
    if (this.meta.n < 2) return;
    this.playing = true; this.playBtn.textContent = "❚❚";
    let last = 0;
    const tick = (t) => {
      if (!this.playing) return;
      if (t - last >= 1000 / this.fps) {
        last = t;
        let next = (this.index + 1) % this.meta.n;
        if (!this.frames[next]) { this.load(this.generation); requestAnimationFrame(tick); return; }   // wait for data
        this.show(next);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop() { this.playing = false; this.playBtn.textContent = "▶"; this.scheduleDetail(); }

  // ------------------------------------------------------------------ detail on zoom
  scheduleDetail() {
    clearTimeout(this.detailTimer);
    this.detailTimer = setTimeout(() => this.fetchDetail(), 180);
  }

  async fetchDetail() {
    if (!this.meta || this.playing || !this.cssSize) return;
    const m = this.meta;
    if (this.view.s * m.step <= 1.2) { this.detail = null; this.render(); return; }   // preview is fine enough
    const [x0, y0] = this.toCell(0, 0), [x1, y1] = this.toCell(...this.cssSize);
    const dpr = window.devicePixelRatio || 1;
    const index = this.index, generation = this.generation;
    const { content, buffers } = await this.transport.request({ type: "detail", index,
      x0: Math.max(0, x0), y0: Math.max(0, y0), x1: Math.min(m.full_width, x1), y1: Math.min(m.full_height, y1),
      max_px: Math.round(Math.max(...this.cssSize) * dpr) });
    if (content.empty || generation !== this.generation || index !== this.index) return;
    const bytes = await inflate(buffers[0], m.compressed);
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.detailTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.texFormat[0], content.width, content.height, 0, this.texFormat[1],
      gl.UNSIGNED_BYTE, bytes);
    this.detail = { ...content, bytes };
    this.render();
  }

  // ------------------------------------------------------------------ drawing
  render() {
    if (!this.meta || !this.gl || !this.cssSize) return;
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(() => this.draw());
  }

  draw() {
    const gl = this.gl, m = this.meta, [cw, ch] = this.cssSize;
    this.label.textContent = m.labels[this.index] || "";
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.uniform3f(this.u.view, this.view.ox, this.view.oy, this.view.s);
    gl.uniform2f(this.u.canvas, cw, ch);
    const frame = this.frames[this.index];
    if (frame) {
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.baseTex);
      if (this.uploaded !== this.index) {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, m.width, m.height, this.texFormat[1], gl.UNSIGNED_BYTE, frame);
        this.uploaded = this.index;
      }
      gl.uniform1i(this.u.data, 0);
      gl.uniform4f(this.u.rect, 0, 0, m.width * m.step, m.height * m.step);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    const d = this.detail;
    if (d && d.index === this.index) {
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.detailTex);
      gl.uniform1i(this.u.data, 1);
      gl.uniform4f(this.u.rect, d.x0, d.y0, d.x1, d.y1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    for (const [canvas, below] of [[this.below, true], [this.above, false]]) {
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cw, ch);
      for (const layer of this.layers) if (!!layer.below === below && layer.draw2d) layer.draw2d(ctx, this);
    }
  }

  drawLegend() {
    const s = this.meta.style;
    this.legend.replaceChildren();
    this.rampCanvas = null;
    if (s.kind === "rgb") { this.addOpacity(); return; }
    if (s.kind === "categorical") {
      for (const [, text, color] of s.classes)
        this.legend.append(h("span", { class: "zv-chip" }, h("i", { style: `background:${color}` }), text));
      this.addOpacity();
      return;
    }
    const ticks = h("div", { class: "zv-ticks" });
    for (const [v, t] of s.ticks) {
      const pos = (100 * (v - s.vmin)) / ((s.vmax - s.vmin) || 1);
      const shift = pos < 6 ? "0" : pos > 94 ? "-100%" : "-50%";   // keep the end labels inside
      ticks.append(h("span", { style: `left:${pos}%;transform:translateX(${shift})` }, t));
    }
    this.rampCanvas = h("canvas", { class: "zv-ramp", width: 255, height: 1 });
    this.legend.append(h("div", { class: "zv-rampwrap" }, this.rampCanvas, ticks), h("span", { class: "zv-unit" }, s.label || ""));
    this.paintRamp();
    this.addOpacity();
  }

  addOpacity() { if (this.meta.basemap) this.legend.append(this.opacityBox); }

  // ------------------------------------------------------------------ basemap
  lonlatOf(px, py) { return interp(this.meta.geo.forward, ...this.toCell(px, py)); }

  cellOf(lon, lat) { return interp(this.meta.geo.inverse, lon, lat); }

  drawBasemap(ctx) {
    const m = this.meta;
    if (!m || !m.basemap || !m.geo) return;
    const [cw, ch] = this.cssSize;
    const pts = [[0, 0], [cw, 0], [0, ch], [cw, ch], [cw / 2, 0], [cw / 2, ch], [0, ch / 2], [cw, ch / 2], [cw / 2, ch / 2]];
    const ll = pts.map(([x, y]) => this.lonlatOf(x, y));
    let lon0 = Math.min(...ll.map((p) => p[0])), lon1 = Math.max(...ll.map((p) => p[0]));
    let lat0 = Math.min(...ll.map((p) => p[1])), lat1 = Math.max(...ll.map((p) => p[1]));
    lon0 = Math.max(lon0, -180); lon1 = Math.min(lon1, 180); lat0 = Math.max(lat0, -85); lat1 = Math.min(lat1, 85);
    if (!(lon1 > lon0) || !(lat1 > lat0)) return;
    let z = Math.round(Math.log2((360 * cw) / ((lon1 - lon0) * 256)));
    z = Math.min(Math.max(z, 0), m.basemap.max_zoom || 19);
    let tx0, tx1, ty0, ty1;
    for (;;) {
      tx0 = Math.floor(lonTile(lon0, z)); tx1 = Math.floor(lonTile(lon1, z));
      ty0 = Math.floor(latTile(lat1, z)); ty1 = Math.floor(latTile(lat0, z));
      if ((tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= 120 || z === 0) break;
      z -= 1;
    }
    const toPx = (lon, lat) => { const [cx, cy] = this.cellOf(lon, lat);
      return [(cx - this.view.ox) * this.view.s, (cy - this.view.oy) * this.view.s]; };
    for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
      const img = this.tile(z, tx, ty);
      if (!img.complete || !img.naturalWidth) continue;
      const nw = toPx(tileLon(tx, z), tileLat(ty, z)), ne = toPx(tileLon(tx + 1, z), tileLat(ty, z));
      const sw = toPx(tileLon(tx, z), tileLat(ty + 1, z));
      ctx.save();
      ctx.transform((ne[0] - nw[0]) / 256, (ne[1] - nw[1]) / 256, (sw[0] - nw[0]) / 256, (sw[1] - nw[1]) / 256, nw[0], nw[1]);
      ctx.drawImage(img, -0.6, -0.6, 257.2, 257.2);   // a hair larger: no seams between tiles
      ctx.restore();
    }
  }

  tile(z, x, y) {
    const n = 2 ** z, key = `${z}/${x}/${y}`;
    let img = this.tiles.get(key);
    if (img) return img;
    img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => this.render();
    const xx = ((x % n) + n) % n;
    img.src = this.meta.basemap.url.replace("{z}", z).replace("{x}", xx).replace("{y}", y)
      .replace("{s}", "abc"[(x + y) % 3]).replace("{r}", "");
    this.tiles.set(key, img);
    if (this.tiles.size > 600) this.tiles.delete(this.tiles.keys().next().value);
    return img;
  }

  // ------------------------------------------------------------------ vectors
  async loadVectors(generation) {
    try {
      const { content, buffers } = await this.transport.request({ type: "vectors" });
      if (generation !== this.generation || !buffers.length) return;
      const b = buffers[0];
      const bytes = new Uint8Array(b.buffer || b, b.byteOffset || 0, b.byteLength).slice();
      this.vectors = { paths: content.paths, xy: new Float32Array(bytes.buffer) };
      this.render();
    } catch (err) { this.readout.textContent = `vectors: ${err.message || err}`; }
  }

  drawVectors(ctx) {
    const v = this.vectors, m = this.meta;
    if (!v) return;
    const { ox, oy, s } = this.view;
    ctx.save();
    ctx.lineJoin = "round";
    const color = m.vector_color || "#ffd400";
    for (const pass of [0, 1]) {   // a dark halo under the colour keeps lines visible on any background
      ctx.strokeStyle = pass ? color : "rgba(0,0,0,0.55)";
      ctx.fillStyle = color;
      ctx.lineWidth = pass ? (m.vector_width || 1.5) : (m.vector_width || 1.5) + 2;
      ctx.beginPath();
      const dots = new Path2D();
      for (const [off, len, closed] of v.paths) {
        const x0 = (v.xy[2 * off] - ox) * s, y0 = (v.xy[2 * off + 1] - oy) * s;
        if (len === 1) { dots.moveTo(x0 + 3.5, y0); dots.arc(x0, y0, 3.5, 0, 2 * Math.PI); continue; }
        ctx.moveTo(x0, y0);
        for (let k = 1; k < len; k++) ctx.lineTo((v.xy[2 * (off + k)] - ox) * s, (v.xy[2 * (off + k) + 1] - oy) * s);
        if (closed) ctx.closePath();
      }
      ctx.stroke();
      ctx.stroke(dots);
      if (pass) ctx.fill(dots);
    }
    ctx.restore();
  }

  paintRamp() {
    if (!this.rampCanvas || !this.lutRGBA) return;
    const ctx = this.rampCanvas.getContext("2d"), img = ctx.createImageData(255, 1);
    img.data.set(this.lutRGBA.subarray(4, 256 * 4));
    ctx.putImageData(img, 0, 0);
  }

  // ------------------------------------------------------------------ events
  on(name, fn) { this.listeners.push([name, fn]); }
  emit(name, value) { for (const [n, fn] of this.listeners) if (n === name) fn(value, this); }
}

// Keep the LUT bytes for the legend ramp as well as the texture.
const setMeta = Viewer.prototype.setMeta;
Viewer.prototype.setMeta = function (meta, buffers) {
  const b = buffers[0];
  this.lutRGBA = new Uint8Array(b.buffer || b, b.byteOffset || 0, 1024).slice();
  setMeta.call(this, meta, buffers);
};

export function mount(el, transport, options = {}) {
  const viewer = new Viewer(el, transport, options);
  viewer.start();
  return viewer;
}

// ---------------------------------------------------------------------- anywidget
function widgetTransport(model) {
  let next = 0;
  const waiting = new Map();
  model.on("msg:custom", (msg, buffers) => {
    if (msg.type !== "reply") return;
    const p = waiting.get(msg.id); if (!p) return;
    waiting.delete(msg.id);
    if (msg.error) p.reject(new Error(msg.error)); else p.resolve({ content: msg.content, buffers: buffers || [] });
  });
  return {
    request(req) {
      const id = ++next;
      return new Promise((resolve, reject) => { waiting.set(id, { resolve, reject });
        model.send({ type: "request", id, request: req }); });
    },
  };
}

export default {
  render({ model, el }) {
    const viewer = mount(el, widgetTransport(model), { css: model.get("_css_text"), fps: model.get("fps") });
    el.style.setProperty("--zv-height", `${model.get("height")}px`);
    for (const plugin of (globalThis.__zeitViewerPlugins || [])) plugin(viewer, model);
    return () => { viewer.stop(); };
  },
};
