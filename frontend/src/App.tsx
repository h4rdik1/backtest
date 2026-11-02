
import './App.css';
import { StatCard } from './components/StatCard';

function App() {
  return (
    <div>
      <StatCard label="Balance" value="$5,000.00" trend="bullish" />
      <StatCard label="Total P/L" value="-$120.00" trend="bearish" />
    </div>
  );
}

export default App;
