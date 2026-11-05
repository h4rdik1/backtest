// ─────────────────────────────────────────────────────────────
// src/api/client.ts — single fetch wrapper.
// All network calls go through fetchJson. Never call fetch() directly.
// ─────────────────────────────────────────────────────────────

export class ApiError extends Error {
  status: number;
  raw: unknown;

  constructor(message: string, status: number, raw?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.raw = raw;
  }
}

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;

  try {
    res = await fetch(url, {
      ...init,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json', ...init?.headers },
    });
  } catch (err) {
    throw new ApiError(
      `Network error: ${err instanceof Error ? err.message : String(err)}`,
      0,
      err,
    );
  }

  if (!res.ok) {
    let body: unknown;
    try { body = await res.clone().json(); }
    catch { body = await res.clone().text().catch(() => null); }
    throw new ApiError(`HTTP ${res.status}`, res.status, body);
  }

  return res.json() as Promise<T>;
}
