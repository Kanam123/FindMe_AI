/* FINDME AI — frontend interactions (no backend changes) */
(function () {
  "use strict";

  /* ---------------- Theme ---------------- */
  var root = document.documentElement;
  function applyTheme(t) {
    root.setAttribute("data-theme", t);
    try { localStorage.setItem("findme-theme", t); } catch (e) {}
  }
  var stored;
  try { stored = localStorage.getItem("findme-theme"); } catch (e) {}
  applyTheme(stored || "dark");

  function onReady(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  onReady(function () {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }

    /* Theme toggle */
    var themeBtn = document.querySelector("[data-theme-toggle]");
    if (themeBtn) {
      themeBtn.addEventListener("click", function () {
        var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
        applyTheme(next);
      });
    }

    /* Mobile nav */
    var navToggle = document.querySelector("[data-nav-toggle]");
    var header = document.querySelector(".nav");
    if (navToggle && header) {
      navToggle.addEventListener("click", function () {
        var open = header.classList.toggle("nav-open");
        navToggle.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }

    /* Upload: drag & drop + preview + remove */
    var box = document.querySelector("[data-upload-box]");
    if (box) {
      var input = box.querySelector("[data-image-input]");
      var preview = box.querySelector("[data-upload-preview]");
      var previewImg = box.querySelector("[data-preview-image]");
      var fileName = box.querySelector("[data-file-name]");
      var removeBtn = box.querySelector("[data-remove-image]");
      var label = box.querySelector(".upload-label");

      function showFile(file) {
        if (!file) return;
        if (fileName) fileName.textContent = file.name;
        if (previewImg) previewImg.src = URL.createObjectURL(file);
        if (preview) preview.hidden = false;
        if (label) label.style.display = "none";
      }
      function clearFile() {
        if (input) input.value = "";
        if (preview) preview.hidden = true;
        if (label) label.style.display = "";
      }

      if (input) {
        input.addEventListener("change", function () {
          if (input.files && input.files[0]) showFile(input.files[0]);
        });
      }
      if (removeBtn) {
        removeBtn.addEventListener("click", function (e) {
          e.preventDefault();
          clearFile();
        });
      }
      ["dragenter", "dragover"].forEach(function (ev) {
        box.addEventListener(ev, function (e) { e.preventDefault(); box.classList.add("dragover"); });
      });
      ["dragleave", "drop"].forEach(function (ev) {
        box.addEventListener(ev, function (e) { e.preventDefault(); box.classList.remove("dragover"); });
      });
      box.addEventListener("drop", function (e) {
        var files = e.dataTransfer && e.dataTransfer.files;
        if (files && files.length && input) {
          input.files = files;
          showFile(files[0]);
        }
      });
    }

    /* Search form: loading state on submit */
    var forms = document.querySelectorAll("[data-search-form]");
    forms.forEach(function (form) {
      form.addEventListener("submit", function () {
        var loading = form.querySelector("[data-loading-state]");
        if (loading) loading.hidden = false;
        var btn = form.querySelector('button[type="submit"], input[type="submit"]');
        if (btn) {
          btn.disabled = true;
          if (!btn.dataset.label) btn.dataset.label = btn.textContent;
          btn.textContent = "Searching…";
        }
      });
    });

    /* Clear search */
    var clearBtn = document.querySelector("[data-clear-search]");
    if (clearBtn) {
      clearBtn.addEventListener("click", function (e) {
        e.preventDefault();
        var f = clearBtn.closest("form");
        if (f) f.reset();
        var ta = document.querySelector("#query");
        if (ta) ta.value = "";
        var box2 = document.querySelector("[data-upload-box]");
        if (box2) {
          var i = box2.querySelector("[data-image-input]");
          var p = box2.querySelector("[data-upload-preview]");
          var l = box2.querySelector(".upload-label");
          if (i) i.value = "";
          if (p) p.hidden = true;
          if (l) l.style.display = "";
        }
      });
    }

    /* Password visibility */
    document.querySelectorAll("[data-toggle-password]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var target = document.getElementById(btn.getAttribute("data-toggle-password"));
        if (!target) return;
        var reveal = target.type === "password";
        target.type = reveal ? "text" : "password";
        btn.classList.toggle("revealed", reveal);
        btn.setAttribute("aria-label", reveal ? "Hide password" : "Show password");
      });
    });

    /* People directory: filter + sort */
    var filterInput = document.querySelector("[data-people-filter]");
    var sortSelect = document.querySelector("[data-people-sort]");
    var list = document.querySelector("[data-people-list]");
    if (list) {
      var cards = Array.prototype.slice.call(list.querySelectorAll("[data-person-card]"));
      if (filterInput) {
        filterInput.addEventListener("input", function () {
          var q = filterInput.value.trim().toLowerCase();
          cards.forEach(function (c) {
            c.style.display = c.getAttribute("data-search").indexOf(q) !== -1 ? "" : "none";
          });
        });
      }
      if (sortSelect) {
        sortSelect.addEventListener("change", function () {
          var key = sortSelect.value;
          cards.sort(function (a, b) {
            return (a.getAttribute("data-" + key) || "").localeCompare(b.getAttribute("data-" + key) || "");
          });
          cards.forEach(function (c) { list.appendChild(c); });
        });
      }
    }
  });
})();
