// zeit.interpret: label reference points by eye, one after the other.
//
// This file is appended to viewer.js into one module (see _widget.interpret_bundle), so
// `Viewer`, `h`, `inflate` and `widgetTransport` are in scope. The Viewer draws the map, the
// time slider and the series chart; the Interpreter adds the queue of points, the form, the
// chips and the review, and talks to the InterpretSession with four more requests: points,
// point, label and review.

const REF_COLOR = "#2e7d32";
const STATUS_COLOR = { todo: "#9e9e9e", done: "#2e7d32", skipped: "#757575", low: "#ef6c00" };
const CONFIDENCE_LEVELS = ["high", "medium", "low"];

function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === "INPUT" && t.type !== "range" && t.type !== "radio" || t.tagName === "TEXTAREA"
    || t.tagName === "SELECT");
}

function msOf(iso) { return iso ? Date.parse(iso + "T00:00:00Z") : null; }

function isoOf(ms) { return new Date(ms).toISOString().slice(0, 10); }

function pct(v) { return v === null || v === undefined ? "–" : (100 * v).toFixed(1) + "%"; }

class Interpreter {
  constructor(viewer, transport) {
    this.viewer = viewer;
    this.transport = transport;
    this.points = [];
    this.byId = new Map();
    this.current = null;
    this.draft = null;
    this.filter = null;
    this.asking = 0;
    this.build();
  }

  // ------------------------------------------------------------------ DOM
  build() {
    const v = this.viewer;
    // the map shows the queue: a click on it does not inspect another pixel
    v.listeners = v.listeners.filter(([name]) => name !== "click");
    v.on("click", (cell) => this.pickNear(cell));
    v.on("chart", (svg) => this.decorateChart(svg));
    v.on("frame", () => this.markChip());
    v.layers.push({ draw2d: (ctx, viewer) => this.drawPoints(ctx, viewer) });

    this.tabPoints = h("button", { class: "zi-tab zi-on", onclick: () => this.tab("points") }, "Points");
    this.tabReview = h("button", { class: "zi-tab", onclick: () => this.tab("review") }, "Review");
    this.progressEl = h("div", { class: "zi-progress" });
    this.filterEl = h("div", { class: "zi-filter" });
    this.list = h("div", { class: "zi-list", role: "listbox" });
    this.form = this.buildForm();
    this.pointsPane = h("div", { class: "zi-pane" }, this.progressEl, this.filterEl, this.list, this.form);
    this.reviewPane = h("div", { class: "zi-pane zi-review", style: "display:none" });
    this.panel = h("div", { class: "zi-panel" }, h("div", { class: "zi-tabs" }, this.tabPoints, this.tabReview),
      this.pointsPane, this.reviewPane);
    v.side.append(this.panel);
    this.chipsEl = h("div", { class: "zi-chips" });
    v.stage.after(this.chipsEl);
    v.root.classList.add("zi-root");
    v.root.addEventListener("keydown", (e) => this.key(e));
    // keep the point in the middle when the map changes size, until the user moves the map
    const resize = v.resize.bind(v);
    v.resize = () => { resize(); if (this.centered && this.current !== null) this.center(this.byId.get(this.current)); };
    for (const name of ["pointerdown", "wheel"]) v.above.addEventListener(name, () => { this.centered = false; });
  }

  buildForm() {
    this.where = h("div", { class: "zi-where" });
    this.mapSays = h("div", { class: "zi-map" });
    this.classBox = h("div", { class: "zi-classes" });
    this.confBox = h("div", { class: "zi-conf" }, h("span", { class: "zi-k" }, "confidence"));
    this.confButtons = {};
    for (const level of CONFIDENCE_LEVELS) {
      const b = h("button", { class: "zi-chipbtn", title: `${level} (${level[0]})`,
        onclick: () => this.setConfidence(level) }, level);
      this.confButtons[level] = b;
      this.confBox.append(b);
    }
    this.dateEl = h("span", { class: "zi-date" });
    this.dateBox = h("div", { class: "zi-datebox" }, h("span", { class: "zi-k" }, "date"), this.dateEl,
      h("button", { class: "zi-link", title: "The date shown on the map (d)", onclick: () => this.dateFromFrame() }, "this image"),
      h("button", { class: "zi-link", title: "No date", onclick: () => this.setDate(null) }, "clear"));
    this.note = h("input", { class: "zi-note", type: "text", placeholder: "note" });
    this.note.addEventListener("input", () => { if (this.draft) this.draft.note = this.note.value; });
    this.note.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); this.save(); } });
    this.saveBtn = h("button", { class: "zi-primary", title: "Save and go to the next point (Enter)",
      onclick: () => this.save() }, "Save ⏎");
    const skip = h("button", { class: "zi-secondary", title: "Skip this point (s)", onclick: () => this.skip() }, "Skip");
    const prev = h("button", { class: "zi-secondary", title: "Previous point (p)", onclick: () => this.step(-1) }, "◀");
    const next = h("button", { class: "zi-secondary", title: "Next point (n)", onclick: () => this.step(1) }, "▶");
    this.message = h("div", { class: "zi-message" });
    return h("div", { class: "zi-form" }, this.where, this.mapSays, this.classBox, this.confBox, this.dateBox,
      this.note, h("div", { class: "zi-actions" }, prev, this.saveBtn, skip, next), this.message,
      h("div", { class: "zi-help" }, "1–9 class · Enter save · s skip · n/p next/previous · d date of the image · "
        + "click the chart to date the change · ←/→ images"));
  }

  // ------------------------------------------------------------------ data
  async start() {
    const { content } = await this.transport.request({ type: "points" });
    this.meta = content;
    this.points = content.points;
    this.byId = new Map(this.points.map((p) => [p.id, p]));
    this.classes = content.classes;
    this.classBox.replaceChildren(...this.classes.map((c, k) => h("button", { class: "zi-class", "data-class": c,
      title: k < 9 ? `${c} (${k + 1})` : c, onclick: () => this.setClass(c) }, k < 9 ? `${k + 1} ${c}` : c)));
    this.tabReview.style.display = content.has_map ? "" : "none";
    this.chipsEl.style.display = content.chips ? "" : "none";
    this.renderList();
    this.renderProgress(content.progress);
    const first = this.points.find((p) => p.status === "todo") || this.points[0];
    if (first) await this.goTo(first.id); else this.message.textContent = "no point falls on the data";
  }

  async goTo(id) {
    const p = this.byId.get(id);
    if (!p) return;
    this.current = id;
    this.draft = { ref: p.ref, ref_date: p.ref_date, confidence: p.confidence || "high", note: p.note || "" };
    this.center(p);
    this.renderList();
    this.renderForm();
    const ask = ++this.asking;
    this.viewer.chart.classList.add("zv-busy");
    try {
      const { content, buffers } = await this.transport.request({ type: "point", id });
      if (this.current !== id || ask !== this.asking) return;
      Object.assign(p, content.state);
      this.series = content.series;
      this.mapText = content.map;
      this.fitHidden = content.fit_hidden;
      this.showChart();
      this.renderForm();
      if (content.chips) this.showChips(content.chips, buffers[0]);
    } catch (err) {
      this.message.textContent = String(err.message || err);
    } finally { this.viewer.chart.classList.remove("zv-busy"); }
  }

  center(p) {
    const v = this.viewer;
    if (!p) return;
    this.centered = true;
    v.pin = [p.col + 0.5, p.top + 0.5];
    if (v.cssSize && v.view) {
      const [cw, ch] = v.cssSize;
      const zoom = Math.max(8, this.meta.zoom || 120);
      const s = Math.min(Math.max(cw / zoom, 1e-3), 64);
      v.view = { s, ox: v.pin[0] - cw / (2 * s), oy: v.pin[1] - ch / (2 * s) };
      v.fitted = false;
      v.scheduleDetail();
    }
    v.render();
  }

  showChart() {
    const v = this.viewer, s = this.series;
    if (!s) return;
    const overlays = [...(s.overlays || [])];
    if (this.draft && this.draft.ref_date && s.is_time)
      overlays.push({ kind: "vline", x: msOf(this.draft.ref_date), label: "reference date", color: REF_COLOR });
    v.pixel = { ...s, overlays };
    v.drawChart();
  }

  decorateChart(svg) {
    const v = this.viewer, s = this.series;
    if (!s || !s.is_time || !v.chartScale) return;
    if (this.fitHidden) {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", v.chartScale.W - v.chartScale.M.r); t.setAttribute("y", v.chartScale.H - 32);
      t.setAttribute("text-anchor", "end"); t.setAttribute("class", "zv-tick");
      t.textContent = "the algorithm's fit shows once the point is labelled";
      svg.append(t);
    }
    svg.addEventListener("click", (e) => {   // the viewer also moves to that image
      const { x0, x1, W, M } = v.chartScale, r = svg.getBoundingClientRect();
      const x = x0 + ((e.clientX - r.left - M.l) / (W - M.l - M.r)) * (x1 - x0);
      let best = 0;
      s.x.forEach((t, i) => { if (Math.abs(t - x) < Math.abs(s.x[best] - x)) best = i; });
      this.setDate(isoOf(s.x[best]));
    });
  }

  showChips(info, buffer) {
    const bytes = new Uint8Array(buffer.buffer || buffer, buffer.byteOffset || 0, buffer.byteLength);
    const { width: w, height: hh } = info, size = w * hh * 4;
    const scale = Math.max(2, Math.floor(96 / Math.max(w, hh, 1)));
    this.chips = info;
    this.chipsEl.replaceChildren(...info.labels.map((label, k) => {
      const canvas = h("canvas", { width: w * scale, height: hh * scale, class: "zi-chipimg" });
      const ctx = canvas.getContext("2d");
      const tmp = document.createElement("canvas"); tmp.width = w; tmp.height = hh;
      const img = new ImageData(new Uint8ClampedArray(bytes.slice(k * size, (k + 1) * size).buffer), w, hh);
      tmp.getContext("2d").putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmp, 0, 0, w * scale, hh * scale);
      const [cx, cy] = info.center;   // the point's pixel, outlined
      ctx.lineWidth = 2; ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.strokeRect(cx * scale - 1, cy * scale - 1, scale + 2, scale + 2);
      ctx.lineWidth = 1; ctx.strokeStyle = "#fff"; ctx.strokeRect(cx * scale - 0.5, cy * scale - 0.5, scale + 1, scale + 1);
      const fig = h("figure", { class: "zi-chip", title: `${label}: show this image`,
        onclick: () => this.viewer.show(info.first[k]) }, canvas, h("figcaption", {}, label));
      return fig;
    }));
    this.markChip();
  }

  markChip() {
    if (!this.chips) return;
    const i = this.viewer.index;
    [...this.chipsEl.children].forEach((fig, k) => fig.classList.toggle("zi-now", this.chips.members[k].includes(i)));
  }

  // ------------------------------------------------------------------ form
  setClass(c) { if (!this.draft) return; this.draft.ref = c; this.renderForm(); }

  setConfidence(level) { if (!this.draft) return; this.draft.confidence = level; this.renderForm(); }

  setDate(iso) { if (!this.draft) return; this.draft.ref_date = iso; this.renderForm(); this.showChart(); }

  dateFromFrame() {
    const s = this.series;
    if (s && s.is_time && s.x[this.viewer.index] !== undefined) this.setDate(isoOf(s.x[this.viewer.index]));
  }

  renderForm() {
    const p = this.byId.get(this.current), d = this.draft;
    if (!p || !d) return;
    const n = this.points.indexOf(p) + 1;
    this.where.textContent = `point ${p.id} · ${n} of ${this.points.length}` + (p.stratum ? ` · stratum ${p.stratum}` : "");
    this.mapSays.textContent = this.mapText ? `map: ${this.mapText}` : "";
    this.mapSays.style.display = this.mapText ? "" : "none";
    for (const b of this.classBox.children) b.classList.toggle("zi-on", b.dataset.class === d.ref);
    for (const [level, b] of Object.entries(this.confButtons)) b.classList.toggle("zi-on", d.confidence === level);
    this.dateEl.textContent = d.ref_date || "–";
    if (document.activeElement !== this.note) this.note.value = d.note || "";
    this.message.textContent = "";
  }

  async save() {
    const d = this.draft, id = this.current;
    if (!d || id === null) return;
    if (!d.ref) { this.message.textContent = "choose a class (1–9) first"; return; }
    await this.send({ id, status: "done", ref: d.ref, ref_date: d.ref_date, confidence: d.confidence, note: d.note });
  }

  async skip() {
    if (this.current === null) return;
    await this.send({ id: this.current, status: "skipped", note: this.draft ? this.draft.note : null });
  }

  async send(label) {
    try {
      const { content } = await this.transport.request({ type: "label", ...label });
      Object.assign(this.byId.get(label.id), content.point);
      this.renderProgress(content.progress);
      this.meta.review_ready = content.review_ready;
      this.renderList();
      this.step(1, true);
    } catch (err) { this.message.textContent = String(err.message || err); }
  }

  step(direction, toNextTodo = false) {
    const queue = this.visible();
    if (!queue.length) return;
    let k = queue.findIndex((p) => p.id === this.current);
    if (toNextTodo) {
      for (let j = 1; j <= queue.length; j++) {
        const p = queue[(k + j * direction + queue.length * j) % queue.length];
        if (p.status === "todo") return this.goTo(p.id);
      }
      this.message.textContent = "every point has a label" + (this.meta.has_map ? ": see the review" : "");
    }
    k = Math.min(Math.max(k + direction, 0), queue.length - 1);
    return this.goTo(queue[k].id);
  }

  pickNear([cx, cy]) {   // a click on the map: the nearest point within a few screen pixels
    const v = this.viewer;
    let best = null, dist = 12 / v.view.s;
    for (const p of this.visible()) {
      const d = Math.hypot(p.col + 0.5 - cx, p.top + 0.5 - cy);
      if (d < dist) { dist = d; best = p; }
    }
    if (best) this.goTo(best.id);
  }

  key(e) {
    if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (k >= "1" && k <= "9") { const c = this.classes[+k - 1]; if (c) { e.preventDefault(); this.setClass(c); } }
    else if (k === "Enter") { e.preventDefault(); this.save(); }
    else if (k === "s") this.skip();
    else if (k === "n" || k === "PageDown") { e.preventDefault(); this.step(1); }
    else if (k === "p" || k === "PageUp") { e.preventDefault(); this.step(-1); }
    else if (k === "d") this.dateFromFrame();
    else if (k === "h" || k === "m" || k === "l") this.setConfidence({ h: "high", m: "medium", l: "low" }[k]);
  }

  // ------------------------------------------------------------------ queue
  visible() { return this.filter ? this.points.filter((p) => this.filter.ids.has(p.id)) : this.points; }

  renderProgress(pr) {
    this.progressEl.textContent = `${pr.done} of ${pr.total} labelled` + (pr.skipped ? ` · ${pr.skipped} skipped` : "")
      + (this.meta && this.meta.saving ? " · saved as you go" : "");
    this.progressEl.title = this.meta && this.meta.saving ? this.meta.saving : "not saved to a file (save=)";
  }

  renderList() {
    const items = this.visible().map((p) => {
      const color = p.status === "done" ? (p.confidence === "low" ? STATUS_COLOR.low : STATUS_COLOR.done) : STATUS_COLOR[p.status];
      const text = p.status === "done" ? `${p.ref}${p.ref_date ? " · " + p.ref_date : ""}` : p.status === "skipped" ? "skipped" : "";
      const item = h("div", { class: "zi-item" + (p.id === this.current ? " zi-current" : ""), role: "option",
        onclick: () => this.goTo(p.id) }, h("i", { style: `background:${color}` }), h("b", {}, `#${p.id}`),
        h("span", {}, text));
      return item;
    });
    this.list.replaceChildren(...items);
    const cur = this.list.querySelector(".zi-current");
    if (cur) cur.scrollIntoView({ block: "nearest" });
    this.filterEl.style.display = this.filter ? "" : "none";
    if (this.filter) this.filterEl.replaceChildren(h("span", {}, this.filter.text + " "),
      h("button", { class: "zi-link", onclick: () => { this.filter = null; this.renderList(); } }, "show all"));
    this.viewer.render();
  }

  drawPoints(ctx, v) {
    if (!this.points.length || !v.view) return;
    ctx.save();
    for (const p of this.visible()) {
      if (p.id === this.current) continue;   // the viewer's pin marks it
      const x = (p.col + 0.5 - v.view.ox) * v.view.s, y = (p.top + 0.5 - v.view.oy) * v.view.s;
      if (x < -10 || y < -10 || x > v.cssSize[0] + 10 || y > v.cssSize[1] + 10) continue;
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = p.status === "done" ? STATUS_COLOR.done : p.status === "skipped" ? STATUS_COLOR.skipped : "#ffffff";
      ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.lineWidth = 1.5;
      ctx.fill(); ctx.stroke();
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------ review
  tab(name) {
    const review = name === "review";
    this.tabPoints.classList.toggle("zi-on", !review);
    this.tabReview.classList.toggle("zi-on", review);
    this.pointsPane.style.display = review ? "none" : "";
    this.reviewPane.style.display = review ? "" : "none";
    if (review) this.loadReview();
  }

  async loadReview() {
    this.reviewPane.replaceChildren(h("div", { class: "zi-message" }, "computing…"));
    let content;
    try { ({ content } = await this.transport.request({ type: "review" })); }
    catch (err) { this.reviewPane.replaceChildren(h("div", { class: "zi-message" }, String(err.message || err))); return; }
    if (!content.available) { this.reviewPane.replaceChildren(h("div", { class: "zi-message" }, content.reason)); return; }
    const out = [];
    const acc = content.accuracy;
    if (acc) {
      const [oa, lo, hi] = acc.overall;
      out.push(h("div", { class: "zi-oa" }, `overall accuracy ${pct(oa)}`,
        h("span", { class: "zi-k" }, ` (${Math.round(acc.confidence * 100)}% CI ${pct(lo)}–${pct(hi)}, ${acc.n} points)`)));
      const table = h("table", { class: "zi-table" }, h("tr", {}, h("th", {}, "class"), h("th", {}, "user's"),
        h("th", {}, "producer's"), h("th", {}, `area (${acc.unit})`)));
      acc.classes.forEach((c, k) => {
        const [u] = acc.users[k], [pa] = acc.producers[k], [, est, ci] = acc.area[k];
        table.append(h("tr", {}, h("td", {}, c), h("td", {}, pct(u)), h("td", {}, pct(pa)),
          h("td", {}, est === null ? "–" : `${Math.round(est).toLocaleString()} ± ${Math.round(ci).toLocaleString()}`)));
      });
      out.push(table);
    }
    if (content.error) out.push(h("div", { class: "zi-message" }, content.error));
    for (const w of content.warnings || []) out.push(h("div", { class: "zi-warn" }, w));
    // the error matrix in points: a click on a cell lists its points
    const classes = content.classes;
    const matrix = h("table", { class: "zi-table zi-matrix" },
      h("tr", {}, h("th", { title: "rows: the map; columns: the reference" }, "map ╲ ref"), ...classes.map((c) => h("th", {}, c))));
    classes.forEach((m, a) => {
      const row = h("tr", {}, h("th", {}, m));
      classes.forEach((r, b) => {
        const n = content.counts[a][b], ids = content.cells[`${a},${b}`] || [];
        const td = h("td", { class: (a === b ? "zi-diag" : n ? "zi-off" : "") + (n ? " zi-click" : "") }, String(n));
        if (n) td.addEventListener("click", () => {
          this.filter = { ids: new Set(ids), text: `${n} points: map ${m}, reference ${r}` };
          this.tab("points"); this.renderList(); this.goTo(ids[0]);
        });
        row.append(td);
      });
      matrix.append(row);
    });
    out.push(h("div", { class: "zi-k" }, "points per cell (rows: map, columns: reference); click a cell to review its points"), matrix);
    this.reviewPane.replaceChildren(...out);
  }
}

export function mountInterpret(el, transport, options = {}) {
  const viewer = new Viewer(el, transport, options);
  const interpreter = new Interpreter(viewer, transport);
  viewer.interpreter = interpreter;
  viewer.start().then(() => interpreter.start()).catch((err) => { interpreter.message.textContent = String(err.message || err); });
  return viewer;
}

export default {
  render({ model, el }) {
    const viewer = mountInterpret(el, widgetTransport(model), { css: model.get("_css_text"), fps: model.get("fps") });
    el.style.setProperty("--zv-height", `${model.get("height")}px`);
    return () => { viewer.stop(); };
  },
};
