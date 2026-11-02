
import './App.css'
import { StatCard } from './components/StatCard'
import { useState } from 'react';

function App() {
 const [count,setCount]=useState(0);
 return (
  <div className=''>
    <StatCard label='balance' value='5000' trend='bullish'/>
    <StatCard label='pnl' value='5600' trend='bearish'/>
    
  </div>
 );
}

export default App
