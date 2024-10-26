// SignUpPage.js
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { db } from '../firebase'; // Import Firestore
import { collection, addDoc } from 'firebase/firestore'; // Firestore functions
import '../css/Signup.css';

const SignUpPage = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  
  const navigate = useNavigate(); // Hook for navigation

  const handleUsernameChange = (e) => setUsername(e.target.value);
  const handleEmailChange = (e) => setEmail(e.target.value);
  const handlePasswordChange = (e) => setPassword(e.target.value);

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!username || !email || !password) {
      setError('Please fill in all fields.');
      return;
    }

    if (!validateEmail(email)) {
      setError('Please enter a valid email address.');
      return;
    }

    if (password.length < 6) {
      setError('Password should be at least 6 characters long.');
      return;
    }

    try {
      setError('');
      setSuccess(false);
      
      // Add user data to Firestore
      await addDoc(collection(db, 'users'), {
        username: username,
        email: email,
        password: password,
      });

      setSuccess(true);
      alert('Sign Up successful');
      
      // Redirect to login page
      navigate('/login');
      
      // Reset form fields
      setUsername('');
      setEmail('');
      setPassword('');
    } catch (err) {
      console.error('Error adding document: ', err);
      setError('An error occurred while signing up. Please try again.');
    }
  };

  return (
    <div className="signup-page">
      <form className="signup-form" onSubmit={handleSubmit}>
        <h2>Sign Up</h2>
        {error && <p className="error">{error}</p>}
        {success && <p className="success">Sign up successful!</p>}

        <label>
          Username:
          <input
            type="text"
            value={username}
            onChange={handleUsernameChange}
            placeholder="Enter your username"
          />
        </label>

        <label>
          Email:
          <input
            type="email"
            value={email}
            onChange={handleEmailChange}
            placeholder="Enter your email"
          />
        </label>

        <label>
          Password:
          <input
            type="password"
            value={password}
            onChange={handlePasswordChange}
            placeholder="Enter your password"
          />
        </label>

        <button type="submit">Sign Up</button>
        <p>Already have an account? <Link to="/login">Log In</Link></p>
      </form>
    </div>
  );
};

export default SignUpPage;
