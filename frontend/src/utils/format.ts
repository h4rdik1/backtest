export type Trend = 'bullish' | 'bearish' | 'neutral';

export function getTrendClass(trend?: Trend): string {
    if (trend === 'bullish') return 'up';
    if (trend === 'bearish') return 'down';
    return '';
}

const usdFormatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
});

export function formatUSD(amount: number): string {
    return usdFormatter.format(amount);
}

export function formatPercent(value: number): string {
    return `${value.toFixed(1)}%`;
}

export function getBalanceTrend(currentBalance: number, initialBalance: number): Trend {
    if (currentBalance > initialBalance) return 'bullish';
    if (currentBalance < initialBalance) return 'bearish';
    return 'neutral';
}
