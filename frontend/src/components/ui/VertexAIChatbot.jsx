// VertexAIChatbot.js
import React, { useEffect } from 'react';

const VertexAIChatbot = () => {
  useEffect(() => {
    // Create script tag
    const script = document.createElement('script');
    script.src = 'https://www.gstatic.com/dialogflow-console/fast/df-messenger/prod/v1/df-messenger.js';
    script.async = true;

    // Handle errors on script load
    script.onload = () => console.log("Chatbot script loaded successfully.");
    script.onerror = () => console.error("Error loading chatbot script.");

    // Append script to head
    document.head.appendChild(script);

    // Cleanup script on component unmount
    return () => {
      document.head.removeChild(script);
    };
  }, []);

  return (
    <>
      <link
        rel="stylesheet"
        href="https://www.gstatic.com/dialogflow-console/fast/df-messenger/prod/v1/themes/df-messenger-default.css"
      />
      <df-messenger
        intent="WELCOME"
        chat-title="HyperHelp"
        agent-id="551f1e50-fba4-4771-8748-48ba08972f7d"
        language-code="en"
        max-query-length="-1"
      ></df-messenger>

      <style>
        {`
          df-messenger {
            z-index: 999;
            position: fixed;
            bottom: 16px;
            right: 16px;
            --df-messenger-font-color: #000;
            --df-messenger-font-family: Google Sans;
            --df-messenger-chat-background: #f3f6fc;
            --df-messenger-message-user-background: #d3e3fd;
            --df-messenger-message-bot-background: #fff;
          }
        `}
      </style>
    </>
  );
};

export default VertexAIChatbot;
