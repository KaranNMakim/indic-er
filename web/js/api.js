// Thin wrapper over the API. Served from the same origin as FastAPI, so
// there is no base URL to configure and no CORS to think about.

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) {
        detail = typeof body.detail === "string"
          ? body.detail
          : body.detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ");
      }
    } catch {
      /* keep the status line */
    }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  health: () => request("/health"),
  showcase: () => request("/showcase"),
  seed: () => request("/seed"),
  metrics: () => request("/metrics"),

  resolve: (body) =>
    request("/resolve", { method: "POST", body: JSON.stringify(body) }),

  match: (left, right, extra = {}) =>
    request("/match", {
      method: "POST",
      body: JSON.stringify({ left, right, ...extra }),
    }),

  entities: (params = {}) =>
    request(`/entities?${new URLSearchParams(params)}`),

  entity: (id) => request(`/entities/${encodeURIComponent(id)}`),

  reviewQueue: (params = {}) =>
    request(`/review-queue?${new URLSearchParams(params)}`),

  decide: (pairId, decision, reviewer = "demo") =>
    request(`/review/${encodeURIComponent(pairId)}`, {
      method: "POST",
      body: JSON.stringify({ decision, reviewer }),
    }),

  undo: (pairId) =>
    request(`/review/${encodeURIComponent(pairId)}/undo`, { method: "POST" }),
};
