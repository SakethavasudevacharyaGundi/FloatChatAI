'use client';

import { useState } from 'react';
import Link from 'next/link';

export default function Navbar({ isChatPage = false }: { isChatPage?: boolean }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  
  const navItems = isChatPage
    ? [
        { name: 'Home', href: '/' },
        { name: 'Chat', href: '/chat' },
        { name: 'Map', href: '/map' }
      ]
    : [
        { name: 'Map', href: '/map' }
      ];

  const textColor = '#000000';

  return (
    <nav style={{ color: textColor }}>
      <div className="nav-brand" style={{ color: textColor }}>
        <svg className="nav-logo-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" style={{ stroke: textColor }}>
          <path d="M2 12C2 12 5 9 12 12C19 15 22 12 22 12" />
          <path d="M2 17C2 17 5 14 12 17C19 20 22 17 22 17" />
          <path d="M2 7C2 7 5 4 12 7C19 10 22 7 22 7" />
        </svg>
        FloatChat AI
      </div>
      
      <div className="nav-links" onMouseLeave={() => setHoveredIndex(null)}>
        <div 
          className="sliding-indicator" 
          style={{ transform: `translateX(${(hoveredIndex !== null ? hoveredIndex : activeIndex) * 100}%)`, backgroundColor: isChatPage ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.1)' }}
        ></div>
        {navItems.map((item, idx) => (
          <Link 
            key={item.name} 
            href={item.href} 
            className={`nav-link ${activeIndex === idx && hoveredIndex === null ? 'active' : ''}`}
            style={{ color: textColor }}
            onMouseEnter={() => setHoveredIndex(idx)}
            onClick={() => setActiveIndex(idx)}
          >
            {item.name}
          </Link>
        ))}
      </div>

      {!isChatPage && (
        <button className="try-btn">Try Chat Bot</button>
      )}
    </nav>
  );
}
