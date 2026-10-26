// Router and wiring.
//
// Four views, one origin, no build step. Module boundaries mirror a Next.js
// App Router layout (one module per route, a shared pair component, an api
// client) so a port is mechanical — see PLAN.md section 7 for why there is no
// Next.js build here.

import { api } from "./api.js";
import { initGraph, refreshGraphChoices, setGraphActive } from "./graph.js";
import { loadMetrics } from "./metrics.js";
import { initPlayground } from "./playground.js";
import { initReview, setReviewActive, updateBadge } from "./review.js";
import { $, $$, toast } from "./util.js";

const VIEWS = ["playground", "review", "graph", "metrics"];

function show(view) {
  if (!VIEWS.includes(view)) view = "playground";
  for (const name of VIEWS) {
    $(`#view-${name}`).hidden = name !== view;
  }
  for (const tab of $$(".tab")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === view));
  }
  if (location.hash.slice(1) !== view) history.replaceState(null, "", `#${view}`);

  setReviewActive(view === "review");
  setGraphActive(view === "graph");
  if (view === "metrics") loadMetrics();
}

function initTheme() {
  const stored = (() => {
    try {
      return localStorage.getItem("indic-er-theme");
    } catch {
      return null; // private window, blocked storage — the page still works
    }
  })();
  if (stored) document.documentElement.dataset.theme = stored;

  $("#theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.dataset.theme;
    const next = current === "dark" ? "light" : current === "light" ? "auto" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      if (next === "auto") localStorage.removeItem("indic-er-theme");
      else localStorage.setItem("indic-er-theme", next);
    } catch {
      /* nothing to do */
    }
    toast(`Colour scheme: ${next}`);
  });
}

async function initHealth() {
  const pill = $("#health-pill");
  try {
    const health = await api.health();
    pill.textContent =
      `v${health.version} · ${health.seeded_records.toLocaleString()} records · ` +
      `merge ≥ ${health.thresholds.auto_merge} · llm ${health.llm_mode}`;
    pill.classList.add("ok");
    pill.title =
      `${health.graph.nodes} graph nodes, ${health.graph.edges} edges (${health.graph.backend})`;
    updateBadge(health.review_queue);
  } catch (error) {
    pill.textContent = "api unreachable";
    pill.classList.add("bad");
    pill.title = error.message;
  }
}

function init() {
  initTheme();
  initHealth();

  for (const tab of $$(".tab")) {
    tab.addEventListener("click", () => show(tab.dataset.view));
  }
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));

  initPlayground({
    onResolved: () => {
      initHealth();
      refreshGraphChoices();
    },
  });
  initReview({
    onChange: () => {
      initHealth();
      refreshGraphChoices();
    },
  });
  initGraph();

  show(location.hash.slice(1) || "playground");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
