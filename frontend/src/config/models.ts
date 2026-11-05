// ─────────────────────────────────────────────────────────────
// src/config/models.ts — Model Registry
// Adding a new model = one entry here + one adapter in src/api/adapters/.
// Everything else (hooks, UI) auto-picks it up.
// ─────────────────────────────────────────────────────────────

export interface ModelConfig {
  id: string;
  /** Display name in switcher tabs */
  name: string;
  shortName: string;
  description: string;
  /** Tailwind-compatible hex for accent colour */
  accentColor: string;
  accentGlow: string;
  /** API path for /api/challenges/:id */
  apiPath: string;
  /** Whether the backend is expected to be running */
  enabled: boolean;
  /** Default symbol when no open trade is selected */
  defaultSymbol: string;
}

export const MODEL_REGISTRY: Record<string, ModelConfig> = {
  cisd: {
    id: 'cisd',
    name: 'CISD + FVG Multi-TF',
    shortName: 'CISD',
    description: 'Multi-TF Order Blocks, FVG Displacement, SFP & Trend Alignment',
    accentColor: '#3b82f6',
    accentGlow: 'rgba(59,130,246,0.35)',
    apiPath: '/api/challenges/cisd',
    enabled: true,
    defaultSymbol: 'BTC/USDT',
  },
  liquidity_transfer: {
    id: 'liquidity_transfer',
    name: 'Liquidity Transfer',
    shortName: 'LT Model',
    description: '4H Structural Anchor (SFP/Discount-Premium Tap) → 15M MSS + Retest',
    accentColor: '#a855f7',
    accentGlow: 'rgba(168,85,247,0.35)',
    apiPath: '/api/challenges/liquidity_transfer',
    enabled: true,
    defaultSymbol: 'SOL/USDT',
  },
};

export const DEFAULT_MODEL_ID = 'cisd';

export const ALL_MODELS = Object.values(MODEL_REGISTRY);

export function getModel(id: string): ModelConfig | undefined {
  return MODEL_REGISTRY[id];
}
