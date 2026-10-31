// Hover / focus tooltips for the inline SVG charts (elements with a data-tip).
(function () {
  let tip;
  function el() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "zc-tooltip";
      tip.setAttribute("role", "tooltip");
      document.body.appendChild(tip);
    }
    return tip;
  }
  function place(x, y) {
    const t = el();
    const w = t.offsetWidth, h = t.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top + h > window.innerHeight - 8) top = y - h - 14;
    t.style.left = Math.max(8, left) + "px";
    t.style.top = Math.max(8, top) + "px";
  }
  function show(target, x, y) {
    const t = el();
    t.textContent = target.getAttribute("data-tip");
    t.classList.add("is-on");
    place(x, y);
  }
  function hide() { if (tip) tip.classList.remove("is-on"); }
  document.addEventListener("pointermove", (e) => {
    const target = e.target.closest && e.target.closest("[data-tip]");
    if (target) show(target, e.clientX, e.clientY); else hide();
  });
  document.addEventListener("focusin", (e) => {
    const target = e.target.closest && e.target.closest("[data-tip]");
    if (!target) return;
    const r = target.getBoundingClientRect();
    show(target, r.left + Math.min(r.width, 260), r.top);
  });
  document.addEventListener("focusout", hide);
  window.addEventListener("scroll", hide, { passive: true });
})();
