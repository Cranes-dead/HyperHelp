// App.js
import React from 'react';
import Inventory from './ui/Inventory';
import Header from './ui/Header';
import Navbar from './ui/Navbar';

import './css/Header.css';
import './css/Dashboard.css';
import './css/Navbar.css';

function LandingPage() {
  return (
    <div className="app">
      <Header/>
      <Navbar/>
      <Inventory />
    </div>
  );
}

export default LandingPage;