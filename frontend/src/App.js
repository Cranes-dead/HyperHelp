// App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import InventoryPage from './components/InventoryPage';
import OrderPage from './components/OrderPage';
import ReportsPage from './components/ReportsPage';
import StorePage from './components/StorePage';
import SupplierPage from './components/SupplierPage';
import SettingsPage from './components/ui/Settings';
import LoginPage from './components/ui/Login';
import SignUpPage from './components/ui/Signup';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/inventory" element={<InventoryPage />} />
          <Route path="/orders" element={<OrderPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/store" element={<StorePage />} />
          <Route path="/suppliers" element={<SupplierPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignUpPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
