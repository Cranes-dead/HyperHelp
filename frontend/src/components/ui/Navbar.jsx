import React, { useState } from 'react';
import { FaBars, FaHome, FaClipboardList, FaChartLine, FaTruck, FaBoxes, FaStore, FaCog, FaSignOutAlt } from 'react-icons/fa';

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
          <FaHome className="icon" />
          {isExpanded && <span>Dashboard</span>}
        </li>
        <li>
          <FaClipboardList className="icon" />
          {isExpanded && <span>Inventory</span>}
        </li>
        <li>
          <FaChartLine className="icon" />
          {isExpanded && <span>Reports</span>}
        </li>
        <li>
          <FaTruck className="icon" />
          {isExpanded && <span>Suppliers</span>}
        </li>
        <li>
          <FaBoxes className="icon" />
          {isExpanded && <span>Orders</span>}
        </li>
        <li>
          <FaStore className="icon" />
          {isExpanded && <span>Manage Store</span>}
        </li>
      </ul>
      <ul className="nav-links bottom">
        <li>
          <FaCog className="icon" />
          {isExpanded && <span>Settings</span>}
        </li>
        <li>
          <FaSignOutAlt className="icon" />
          {isExpanded && <span>Log Out</span>}
        </li>
      </ul>
    </div>
  );
};

export default LeftNavbar;
