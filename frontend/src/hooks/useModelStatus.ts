// ─────────────────────────────────────────────────────────────
// src/hooks/useModelStatus.ts
// React Query hooks for challenge status polling (every 30s).
// queryKey always includes modelId so each model caches separately.
// ─────────────────────────────────────────────────────────────

import { useQuery } from '@tanstack/react-query';
import { fetchModelStatus, fetchAllChallenges } from '../api/modelsApi';
import type { ChallengeStatus } from '../types';

const POLL_MS = 30_000;
const STALE_MS = 10_000;

export function useModelStatus(modelId: string, opts?: { retry?: number | boolean }) {
  return useQuery<ChallengeStatus, Error>({
    queryKey: ['challenge', modelId],
    queryFn: () => fetchModelStatus(modelId),
    refetchInterval: POLL_MS,
    staleTime: STALE_MS,
    retry: opts?.retry ?? 2,
  });
}

export function useAllChallenges() {
  return useQuery<Record<string, ChallengeStatus>, Error>({
    queryKey: ['challenges'],
    queryFn: fetchAllChallenges,
    refetchInterval: POLL_MS,
    staleTime: STALE_MS,
    retry: 2,
  });
}
