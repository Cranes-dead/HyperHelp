// LandingPage.js
import React from 'react';
import Dashboard from './ui/Dashboard';
import Header from './ui/Header';
import './css/Dashboard.css';
import './css/Header.css';
import './css/Navbar.css';
import RightNavbar from './ui/Navbar';

function LandingPage() {
  return (
    <div className="app">
      <Header/>
      <RightNavbar/>
      <Dashboard />
    </div>
  );
}

export default LandingPage;
