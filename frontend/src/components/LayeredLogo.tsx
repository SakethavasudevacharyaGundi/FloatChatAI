'use client';
import { useEffect, useRef } from 'react';
import './LayeredLogo.css';

export default function LayeredLogo({ text = "FloatChat AI" }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      const { clientX, clientY } = e;
      const { innerWidth, innerHeight } = window;
      
      const xPos = (clientX / innerWidth - 0.5) * 2;
      const yPos = (clientY / innerHeight - 0.5) * 2;

      containerRef.current.style.setProperty('--mouseX', xPos.toString());
      containerRef.current.style.setProperty('--mouseY', yPos.toString());
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="layered-logo-container" ref={containerRef}>
      <div className="layered-text">
        <span className="layer layer-1">{text}</span>
        <span className="layer layer-2">{text}</span>
        <span className="layer layer-3">{text}</span>
        <span className="layer layer-4">{text}</span>
        <span className="layer layer-5">{text}</span>
        <span className="layer layer-front">{text}</span>
      </div>
    </div>
  );
}
