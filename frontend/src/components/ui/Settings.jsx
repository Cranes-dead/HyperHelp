// SettingsPage.js
import React, { useState } from 'react';
import '../css/Settings.css';

const SettingsPage = () => {
  const [profile, setProfile] = useState({
    name: 'John Doe',
    email: 'johndoe@example.com',
  });

  const [notifications, setNotifications] = useState({
    emailNotifications: true,
    pushNotifications: false,
  });

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    setProfile((prevProfile) => ({
      ...prevProfile,
      [name]: value,
    }));
  };

  const handleNotificationChange = (e) => {
    const { name, checked } = e.target;
    setNotifications((prevNotifications) => ({
      ...prevNotifications,
      [name]: checked,
    }));
  };

  const handleSaveProfile = () => {
    alert('Profile information saved!');
    console.log(profile);
  };

  const handleSaveNotifications = () => {
    alert('Notification settings saved!');
    console.log(notifications);
  };

  return (
    <div className="settings-page">
      <h1>Settings</h1>

      <div className="settings-section">
        <h2>Profile Settings</h2>
        <label>
          Name:
          <input
            type="text"
            name="name"
            value={profile.name}
            onChange={handleProfileChange}
          />
        </label>
        <label>
          Email:
          <input
            type="email"
            name="email"
            value={profile.email}
            onChange={handleProfileChange}
          />
        </label>
        <button onClick={handleSaveProfile}>Save Profile</button>
      </div>

      <div className="settings-section">
        <h2>Notification Settings</h2>
        <label>
          <input
            type="checkbox"
            name="emailNotifications"
            checked={notifications.emailNotifications}
            onChange={handleNotificationChange}
          />
          Email Notifications
        </label>
        <label>
          <input
            type="checkbox"
            name="pushNotifications"
            checked={notifications.pushNotifications}
            onChange={handleNotificationChange}
          />
          Push Notifications
        </label>
        <button onClick={handleSaveNotifications}>Save Notifications</button>
      </div>
    </div>
  );
};

export default SettingsPage;
