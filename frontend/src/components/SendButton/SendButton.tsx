import React, { useState, useEffect } from 'react';
import './SendButton.css';

export default function SendButton({ isSending, onClick }: { isSending?: boolean, onClick?: () => void }) {
  const [active, setActive] = useState(false);

  useEffect(() => {
    if (isSending) {
      setActive(true);
      const timer = setTimeout(() => setActive(false), 1500); // matches the 1.5s orbit animation
      return () => clearTimeout(timer);
    }
  }, [isSending]);

  return (
    <button 
      type="submit" 
      className={`orbit-send-button ${active ? 'active' : ''}`}
      onClick={(e) => {
        if (!active) {
          if (onClick) onClick();
        } else {
          e.preventDefault();
        }
      }}
      aria-label="Send Message"
    >
      <svg className="plane" viewBox="0 0 28 26">
        <path d="M5.25,15.24,18.42,3.88,7.82,17l0,4.28a.77.77,0,0,0,1.36.49l3-3.68,5.65,2.25a.76.76,0,0,0,1-.58L22,.89A.77.77,0,0,0,20.85.1L.38,11.88a.76.76,0,0,0,.09,1.36Z" />
      </svg>
    </button>
  );
}
