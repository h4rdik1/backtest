import { getTrendClass, type Trend } from '../utils/format';

interface StatCardProps {
    label: string;
    value: string;
    trend?: Trend;
}

export function StatCard(props: StatCardProps) {
    const trendClass = getTrendClass(props.trend);

    return (
        <div className="stat-item">
            <div className="stat-label">{props.label}</div>
            <div className={`stat-value ${trendClass}`.trim()}>
                {props.value}
            </div>
        </div>
    );
}
