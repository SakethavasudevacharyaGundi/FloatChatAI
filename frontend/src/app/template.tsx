'use client';

import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

export default function Template({ children }: { children: React.ReactNode }) {
  const barsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Animate the 5 horizontal bars from scaleX 1 to 0 sequentially
      gsap.to('.transition-bar-horizontal', {
        scaleX: 0,
        duration: 0.8,
        stagger: 0.1,
        ease: 'power3.inOut',
        transformOrigin: 'right'
      });
    }, barsRef);

    return () => ctx.revert();
  }, []);

  return (
    <>
      <div 
        ref={barsRef}
        style={{
          position: 'fixed',
          inset: 0,
          display: 'flex',
          flexDirection: 'column', // horizontal bars stack vertically
          zIndex: 9999,
          pointerEvents: 'none'
        }}
      >
        {[...Array(5)].map((_, i) => (
          <div 
            key={i}
            className="transition-bar-horizontal"
            style={{
              flex: 1,
              backgroundColor: '#fcfaf5', // Cream color
              transformOrigin: 'right',
              transform: 'scaleX(1)'
            }}
          />
        ))}
      </div>
      {children}
    </>
  );
}
