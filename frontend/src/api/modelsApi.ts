// ─────────────────────────────────────────────────────────────
// src/api/modelsApi.ts — typed API surface.
// Only this file (and test mocks) calls fetchJson.
// ─────────────────────────────────────────────────────────────

import { fetchJson } from './client';
import { adaptCisdStatus, type RawCisdResponse } from './adapters/cisd';
import { adaptLtStatus, type RawLtResponse } from './adapters/liquidityTransfer';
import { MODEL_REGISTRY } from '../config/models';
import type { ChallengeStatus } from '../types';

type AnyRaw = RawCisdResponse | RawLtResponse;

function adapt(modelId: string, raw: AnyRaw): ChallengeStatus {
  if (modelId === 'liquidity_transfer') return adaptLtStatus(raw as RawLtResponse);
  return adaptCisdStatus(raw as RawCisdResponse);
}

export async function fetchModelStatus(modelId: string): Promise<ChallengeStatus> {
  const cfg = MODEL_REGISTRY[modelId];
  if (!cfg) throw new Error(`Unknown model id: ${modelId}`);
  const raw = await fetchJson<AnyRaw>(cfg.apiPath);
  return adapt(modelId, raw);
}

export async function fetchAllChallenges(): Promise<Record<string, ChallengeStatus>> {
  const dict = await fetchJson<Record<string, AnyRaw>>('/api/challenges');
  return Object.fromEntries(
    Object.entries(dict).map(([id, raw]) => [id, adapt(id, raw)])
  );
}
