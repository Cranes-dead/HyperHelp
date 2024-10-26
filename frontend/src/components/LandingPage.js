// App.js
import React from 'react';
import Dashboard from './ui/Dashboard';
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
      <Dashboard />
    </div>
  );
}

export default LandingPage;