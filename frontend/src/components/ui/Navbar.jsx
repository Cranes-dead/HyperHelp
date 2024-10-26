// LeftNavbar.js
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FaBars, FaHome, FaClipboardList, FaChartLine, FaTruck, FaBoxes, FaStore, FaCog, FaSignInAlt } from 'react-icons/fa';

const LeftNavbar = () => {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleNavbar = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className={`navbar ${isExpanded ? 'expanded' : 'collapsed'}`}>
      <div className="navbar-header">
        <button className="hamburger" onClick={toggleNavbar}>
          <FaBars />
        </button>
        {isExpanded && <h2 className="logo-text">Company Name</h2>}
      </div>
      <ul className="nav-links">
        <li>
          <Link to="/">
            <FaHome className="icon" />
            {isExpanded && <span>Dashboard</span>}
          </Link>
        </li>
        <li>
          <Link to="/inventory">
            <FaClipboardList className="icon" />
            {isExpanded && <span>Inventory</span>}
          </Link>
        </li>
        <li>
          <Link to="/reports">
            <FaChartLine className="icon" />
            {isExpanded && <span>Reports</span>}
          </Link>
        </li>
        <li>
          <Link to="/suppliers">
            <FaTruck className="icon" />
            {isExpanded && <span>Suppliers</span>}
          </Link>
        </li>
        <li>
          <Link to="/orders">
            <FaBoxes className="icon" />
            {isExpanded && <span>Orders</span>}
          </Link>
        </li>
        <li>
          <Link to="/store">
            <FaStore className="icon" />
            {isExpanded && <span>Manage Store</span>}
          </Link>
        </li>
      </ul>
      <ul className="nav-links bottom">
        <li>
          <Link to="/settings">
            <FaCog className="icon" />
            {isExpanded && <span>Settings</span>}
          </Link>
        </li>
        <li>
          <Link to="/login">
            <FaSignInAlt className="icon" />
            {isExpanded && <span>Log In</span>}
          </Link>
        </li>
      </ul>
    </div>
  );
};

export default LeftNavbar;
