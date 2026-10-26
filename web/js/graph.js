// Cluster graph (Cytoscape.js, vendored so the demo works offline).
//
// Records are circles, the canonical entity is a square, and `same_as` edges
// are labelled with the pair score. Clicking anything shows its provenance —
// the point of keeping provenance on every fact is that it can be read.

import { api } from "./api.js";
import { $, esc, fixed } from "./util.js";

let cy = null;
let loadedFor = null;

export async function initGraph() {
  $("#graph-refit").addEventListener("click", () => cy?.fit(undefined, 30));
  $("#graph-entity").addEventListener("change", (event) => show(event.target.value));
  await populateSelect();
}

async function populateSelect() {
  const select = $("#graph-entity");
  try {
    const { entities } = await api.entities({ limit: 150, min_size: 2 });
    if (!entities.length) {
      select.innerHTML = `<option value="">no merged entities yet</option>`;
      return;
    }
    select.innerHTML = entities
      .map(
        (entity) =>
          `<option value="${esc(entity.entity_id)}">${esc(entity.canonical_name)} — ${entity.size} records</option>`
      )
      .join("");
  } catch (error) {
    select.innerHTML = `<option value="">${esc(error.message)}</option>`;
  }
}

export async function setGraphActive(isActive) {
  if (!isActive) return;
  const select = $("#graph-entity");
  if (!select.options.length || !select.value) await populateSelect();
  if (select.value && select.value !== loadedFor) show(select.value);
  else cy?.resize();
}

export async function show(entityId) {
  if (!entityId) return;
  loadedFor = entityId;
  const container = $("#cy");
  try {
    const detail = await api.entity(entityId);
    const elements = detail.graph.elements;
    if (cy) cy.destroy();

    const style = getComputedStyle(document.body);
    const token = (name, fallback) =>
      (style.getPropertyValue(name) || fallback).trim();

    cy = cytoscape({
      container,
      elements,
      style: [
        {
          selector: "node",
          style: {
            "background-color": token("--accent", "#1f6feb"),
            label: "data(label)",
            color: token("--text", "#16181d"),
            "font-size": 10,
            "font-family": "system-ui, sans-serif",
            "text-valign": "bottom",
            "text-margin-y": 5,
            "text-wrap": "wrap",
            "text-max-width": 110,
            width: 22,
            height: 22,
            "border-width": 1,
            "border-color": token("--border-strong", "#c3c8d0"),
          },
        },
        {
          selector: 'node[kind = "CanonicalEntity"]',
          style: {
            shape: "round-rectangle",
            "background-color": token("--merge", "#11823b"),
            width: 36,
            height: 28,
            "font-weight": "bold",
            "font-size": 11,
          },
        },
        {
          selector: 'node[kind = "Business"]',
          style: { shape: "diamond", width: 26, height: 26 },
        },
        {
          selector: "edge",
          style: {
            width: "mapData(score, 0.5, 1, 1, 5)",
            "line-color": token("--border-strong", "#c3c8d0"),
            "curve-style": "bezier",
            label: "data(score)",
            "font-size": 8,
            "font-family": "monospace",
            color: token("--text-muted", "#5c6370"),
            "text-background-color": token("--surface", "#fff"),
            "text-background-opacity": 0.85,
            "text-background-padding": 2,
            opacity: 0.85,
          },
        },
        {
          selector: 'edge[decision = "merge"]',
          style: { "line-color": token("--merge", "#11823b") },
        },
        {
          selector: ":selected",
          style: {
            "border-width": 3,
            "border-color": token("--accent", "#1f6feb"),
            "line-color": token("--accent", "#1f6feb"),
          },
        },
      ],
      layout: { name: "cose", animate: false, padding: 30, nodeRepulsion: 9000 },
      wheelSensitivity: 0.25,
    });

    cy.on("tap", "node", (event) => describeNode(event.target, detail));
    cy.on("tap", "edge", (event) => describeEdge(event.target));
    cy.on("tap", (event) => {
      if (event.target === cy) renderSide(summary(detail));
    });
    renderSide(summary(detail));
  } catch (error) {
    container.innerHTML = `<p class="error" style="padding:14px">${esc(error.message)}</p>`;
  }
}

function summary(detail) {
  const entity = detail.entity;
  return `
    <h4>${esc(entity.canonical_name)}</h4>
    <p class="muted">${entity.size} records · cluster score ${fixed(entity.cluster_score, 3)}
       · ${esc(entity.review_state.replace("_", " "))}</p>
    <dl class="kv">
      <dt>entity</dt><dd>${esc(entity.entity_id)}</dd>
      <dt>kind</dt><dd>${esc(entity.kind)}</dd>
      <dt>scripts</dt><dd>${esc(entity.scripts.join(", "))}</dd>
      <dt>sources</dt><dd>${esc(entity.sources.join(", "))}</dd>
      ${entity.canonical_phone ? `<dt>phone</dt><dd>${esc(entity.canonical_phone)}</dd>` : ""}
      ${entity.canonical_address?.pincode ? `<dt>pincode</dt><dd>${esc(entity.canonical_address.pincode)}</dd>` : ""}
    </dl>
    ${entity.split_reason ? `<div class="split-reason">${esc(entity.split_reason)}</div>` : ""}
    ${
      detail.conflicts.length
        ? `<p class="hint" style="margin-top:9px"><strong>${detail.conflicts.length} internal conflict(s):</strong>
           ${detail.conflicts.slice(0, 3).map((c) => esc(c.rule)).join(", ")}</p>`
        : ""
    }
    <p class="hint" style="margin-top:9px">Variants:<br>${entity.variants.map((v) => esc(v)).join("<br>")}</p>
    <p class="hint">Click a node or an edge for its provenance.</p>`;
}

function describeNode(node, detail) {
  const data = node.data();
  const provenance = data.provenance || {};
  const member = detail.members.find((m) => m.record_id === data.id);
  renderSide(`
    <h4>${esc(data.label)}</h4>
    <p class="muted">${esc(data.kind)} · <code>${esc(data.id)}</code></p>
    ${
      member
        ? `<dl class="kv">
            <dt>raw name</dt><dd>${esc(member.raw_name)}</dd>
            <dt>script</dt><dd>${esc(member.name.script)}</dd>
            <dt>latin</dt><dd>${esc(member.name.display)}</dd>
            <dt>key</dt><dd>${esc(member.name.phonetic_key)}</dd>
            ${member.address ? `<dt>address</dt><dd>${esc(member.address.raw)}</dd>` : ""}
            ${member.phone_last10 ? `<dt>phone</dt><dd>${esc(member.phone_last10)}</dd>` : ""}
            <dt>source</dt><dd>${esc(member.source)}</dd>
          </dl>`
        : ""
    }
    <h4 style="margin-top:11px">Provenance</h4>
    <dl class="kv">
      <dt>source</dt><dd>${esc(provenance.source)}</dd>
      <dt>method</dt><dd>${esc(provenance.method)}</dd>
      <dt>confidence</dt><dd>${fixed(provenance.confidence, 2)}</dd>
      <dt>when</dt><dd>${esc((provenance.timestamp || "").slice(0, 19))}</dd>
    </dl>`);
}

function describeEdge(edge) {
  const data = edge.data();
  const provenance = data.provenance || {};
  const evidence = provenance.detail || {};
  renderSide(`
    <h4>same_as</h4>
    <p class="muted"><code>${esc(data.source)}</code> ~ <code>${esc(data.target)}</code></p>
    <dl class="kv">
      <dt>score</dt><dd>${fixed(data.score, 3)}</dd>
      <dt>decision</dt><dd>${esc(data.decision)}</dd>
      <dt>method</dt><dd>${esc(provenance.method)}</dd>
      <dt>source</dt><dd>${esc(provenance.source)}</dd>
      <dt>confidence</dt><dd>${fixed(provenance.confidence, 2)}</dd>
      ${evidence.rule ? `<dt>rule</dt><dd>${esc(evidence.rule)}</dd>` : ""}
      ${evidence.reviewer ? `<dt>reviewer</dt><dd>${esc(evidence.reviewer)}</dd>` : ""}
      <dt>when</dt><dd>${esc((provenance.timestamp || "").slice(0, 19))}</dd>
    </dl>`);
}

function renderSide(html) {
  $("#graph-side").innerHTML = html;
}

export const refreshGraphChoices = populateSelect;
