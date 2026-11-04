// EndoComparer project page — small, dependency-free interactions.
(function () {
  "use strict";

  // ---- theme toggle (system by default; explicit choice remembered per browser) ----
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  function stored() { try { return localStorage.getItem("theme"); } catch (e) { return null; } }
  function apply(t) {
    if (t === "light" || t === "dark") root.setAttribute("data-theme", t);
    else root.removeAttribute("data-theme");
    if (btn) btn.setAttribute("aria-label", "Theme: " + (t || "system") + " (click to change)");
  }
  apply(stored());
  if (btn) {
    btn.addEventListener("click", function () {
      var cur = stored();
      var dark = root.getAttribute("data-theme") === "dark" ||
        (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
      var next = cur ? null : (dark ? "light" : "dark"); // system -> opposite of current; explicit -> back to system
      try { if (next) localStorage.setItem("theme", next); else localStorage.removeItem("theme"); } catch (e) { /* ignore */ }
      apply(next);
    });
  }
})();

// ---- drop-down selectors: category -> template (attribute) -> one visible card ----
(function () {
  "use strict";
  function setup(root) {
    var catSel = root.querySelector("select[data-role=cat]");
    var tplSel = root.querySelector("select[data-role=tpl]");
    var pool = document.querySelector(root.getAttribute("data-cards")) || root.nextElementSibling;
    var cards = pool ? pool.querySelectorAll("[data-tid]") : [];
    if (!catSel || !tplSel) return;
    var all = Array.prototype.map.call(tplSel.options, function (o) {
      return { value: o.value, text: o.textContent, cat: o.getAttribute("data-cat") };
    });
    function fill(cat) {
      var keep = tplSel.value;
      tplSel.innerHTML = "";
      all.forEach(function (o) {
        if (cat !== "all" && o.cat !== cat) return;
        var el = document.createElement("option");
        el.value = o.value; el.textContent = o.text; el.setAttribute("data-cat", o.cat);
        tplSel.appendChild(el);
      });
      if (Array.prototype.some.call(tplSel.options, function (o) { return o.value === keep; })) tplSel.value = keep;
    }
    function render() {
      var tid = tplSel.value;
      cards.forEach(function (k) { k.hidden = k.getAttribute("data-tid") !== tid; });
    }
    catSel.addEventListener("change", function () { fill(catSel.value); render(); });
    tplSel.addEventListener("change", render);
    fill(catSel.value);
    render();
  }
  document.querySelectorAll("[data-selector]").forEach(setup);
})();
