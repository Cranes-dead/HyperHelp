// Header.js
import React, { useState } from 'react';
import NotificationModal from './Modal';

const Header = () => {
  const [isModalOpen, setModalOpen] = useState(false);

  // Sample notifications data
  const notifications = [
    "You have a new message.",
    "Your profile was updated.",
    "New comment on your post.",
  ];

  const handleNotificationClick = () => {
    setModalOpen(true);
  };

  const closeModal = () => {
    setModalOpen(false);
  };

  return (
    <header className="header">
      {/* Center - Logo and Company Name */}
      <div className="logo">
        <span className="company-name">Company Name</span>
      </div>

      {/* Right - Notification and Profile Icons */}
      <div className="icons">
        <button className="icon-button" onClick={handleNotificationClick}>
          <svg
            className="icon"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M15 17h5l-1.405-4.215A2.001 2.001 0 0017 11h-2v-3a5 5 0 00-10 0v3H3a2.001 2.001 0 00-1.595 1.785L1 17h14z"
            />
          </svg>
        </button>

        <div className="profile">
          <img
            src="https://via.placeholder.com/32"
            alt="Profile"
            className="profile-image"
          />
        </div>
      </div>

      {/* Modal for Notifications */}
      {isModalOpen && (
        <NotificationModal
          notifications={notifications}
          onClose={closeModal}
        />
      )}
    </header>
  );
};

export default Header;
