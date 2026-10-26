// Small shared helpers.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Escape text for innerHTML. Record data is user input; never trust it. */
export function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
export const fixed = (v, n = 3) => (v == null ? "—" : Number(v).toFixed(n));

export function el(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

let toastTimer;
export function toast(message, ms = 2600) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, ms);
}

/**
 * Parse pasted records: a JSON array, or CSV with a header row.
 *
 * CSV is parsed properly rather than split on commas, because every
 * interesting Indian address contains commas and lives inside quotes.
 */
export function parseRecords(text) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("Nothing to resolve — paste some records first.");

  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed);
    const rows = Array.isArray(parsed) ? parsed : parsed.records;
    if (!Array.isArray(rows)) throw new Error("JSON must be an array of records.");
    return rows.map(normaliseRow);
  }

  const rows = parseCsv(trimmed);
  if (rows.length < 2) throw new Error("CSV needs a header row and at least one record.");
  const header = rows[0].map((h) => h.trim().toLowerCase());
  if (!header.includes("name")) throw new Error('CSV header must include a "name" column.');

  return rows.slice(1)
    .filter((cells) => cells.some((c) => c.trim()))
    .map((cells, index) => {
      const row = {};
      header.forEach((key, i) => { row[key] = (cells[i] ?? "").trim(); });
      if (!row.id) row.id = `r${index + 1}`;
      return normaliseRow(row);
    });
}

function normaliseRow(row) {
  const out = {
    id: String(row.id ?? row.record_id ?? "").trim(),
    name: String(row.name ?? "").trim(),
    kind: row.kind === "business" ? "business" : "person",
  };
  if (!out.id) throw new Error("Every record needs an id.");
  if (!out.name) throw new Error(`Record ${out.id} has no name.`);
  for (const key of ["address", "phone", "source", "identifier_scheme", "identifier_masked"]) {
    const value = row[key];
    if (value != null && String(value).trim() !== "") out[key] = String(value).trim();
  }
  return out;
}

/** RFC4180-ish CSV reader: quotes, escaped quotes, embedded commas and newlines. */
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 1; }
        else quoted = false;
      } else field += ch;
      continue;
    }
    if (ch === '"') { quoted = true; continue; }
    if (ch === ",") { row.push(field); field = ""; continue; }
    if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }
    field += ch;
  }
  row.push(field);
  if (row.some((c) => c !== "")) rows.push(row);
  return rows;
}

export function toCsv(records) {
  const cell = (v) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  const header = ["id", "name", "kind", "address", "phone", "source"];
  return [
    header.join(","),
    ...records.map((r) => header.map((k) => cell(r[k])).join(",")),
  ].join("\n");
}

export function download(filename, text, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

/** Wrap matched tokens in <mark>, for the "Why?" view. */
export function highlightName(name, matched) {
  const keys = new Set(Object.keys(matched || {}));
  if (!keys.size) return esc(name);
  return esc(name)
    .split(/(\s+)/)
    .map((word) => {
      const bare = word.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
      return keys.has(bare) ? `<mark>${word}</mark>` : word;
    })
    .join("");
}

export const DECISION_LABEL = {
  merge: "Same entity",
  review: "Needs a human",
  reject: "Different",
};
