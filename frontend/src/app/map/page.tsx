'use client';

import { useEffect, useRef, useState } from 'react';
import './map.css';
import Link from 'next/link';
import WorldMapPaths from './WorldMapPaths';

const ARGO_FLOATS = [
  { id: 'float-49033', name: 'Argo Float #49033', cx: 150, cy: 200 },
  { id: 'float-19234', name: 'Argo Float #19234', cx: 80, cy: 350 },
  { id: 'float-59231', name: 'Argo Float #59231', cx: 200, cy: 450 },
  { id: 'float-69022', name: 'Argo Float #69022', cx: 800, cy: 200 },
  { id: 'float-79011', name: 'Argo Float #79011', cx: 850, cy: 400 },
  { id: 'float-29045', name: 'Argo Float #29045', cx: 350, cy: 200 },
  { id: 'float-39088', name: 'Argo Float #39088', cx: 320, cy: 350 },
  { id: 'float-49012', name: 'Argo Float #49012', cx: 400, cy: 450 },
  { id: 'float-59066', name: 'Argo Float #59066', cx: 600, cy: 300 },
  { id: 'float-19077', name: 'Argo Float #19077', cx: 650, cy: 450 },
  { id: 'float-69033', name: 'Argo Float #69033', cx: 200, cy: 580 },
  { id: 'float-79044', name: 'Argo Float #79044', cx: 500, cy: 560 },
  { id: 'float-29055', name: 'Argo Float #29055', cx: 800, cy: 590 },
  { id: 'float-39066', name: 'Argo Float #39066', cx: 400, cy: 60 },
  { id: 'float-49077', name: 'Argo Float #49077', cx: 600, cy: 80 },
];

export default function MapPage() {
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const mapRef = useRef<SVGSVGElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | null>(null);
  const rectCache = useRef<DOMRect | null>(null);

  // Math for lerping
  const targetTransform = useRef({ x: 0, y: 0, scale: 1 });
  const currentTransform = useRef({ x: 0, y: 0, scale: 1 });

  // Continuous physics loop for buttery smooth interpolation
  useEffect(() => {
    // Set transformOrigin ONCE — it never changes
    if (mapRef.current) {
      mapRef.current.style.transformOrigin = 'center center';
      mapRef.current.style.willChange = 'transform';
    }

    // Cache rect and invalidate on resize
    const updateRect = () => { rectCache.current = null; };
    window.addEventListener('resize', updateRect);

    const animate = () => {
      const c = currentTransform.current;
      const t = targetTransform.current;

      const dx = t.x - c.x;
      const dy = t.y - c.y;
      const dScale = t.scale - c.scale;

      // Only update DOM if we are actually moving
      if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01 || Math.abs(dScale) > 0.001) {
        c.x += dx * 0.08;
        c.y += dy * 0.08;
        c.scale += dScale * 0.08;

        if (mapRef.current) {
          mapRef.current.style.transform = `translate(${c.x}px, ${c.y}px) scale(${c.scale})`;
        }
      } else if (c.x !== t.x || c.y !== t.y || c.scale !== t.scale) {
        c.x = t.x;
        c.y = t.y;
        c.scale = t.scale;

        if (mapRef.current) {
          mapRef.current.style.transform = `translate(${c.x}px, ${c.y}px) scale(${c.scale})`;
        }
      }

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', updateRect);
    };
  }, []);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!wrapperRef.current) return;

    // Use cached rect — getBoundingClientRect triggers layout reflow
    if (!rectCache.current) {
      rectCache.current = wrapperRef.current.getBoundingClientRect();
    }
    const rect = rectCache.current;
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const scale = 1.45;
    targetTransform.current = {
      x: (centerX - mouseX) * (scale - 1),
      y: (centerY - mouseY) * (scale - 1),
      scale,
    };
  };

  const handleMouseLeave = () => {
    // Reset target state
    targetTransform.current = { x: 0, y: 0, scale: 1 };
  };

  return (
    <div className="map-container">

      {/* Back to Home Button */}
      <div style={{ position: 'absolute', top: '14px', left: '2rem', zIndex: 200, display: 'flex', alignItems: 'center' }}>
        <Link href="/" style={{ textDecoration: 'none' }}>
          <button className="dotted-btn" style={{
            background: 'rgba(255, 255, 255, 0.75)',
            backdropFilter: 'blur(24px)',
            WebkitBackdropFilter: 'blur(24px)',
            color: '#0a1a2e',
            border: '1px solid rgba(255,255,255,0.9)',
            boxShadow: '0 4px 24px rgba(0, 60, 120, 0.15)',
            borderRadius: '50px',
            padding: '0.75rem 2rem',
            fontFamily: "'Outfit', sans-serif",
            fontWeight: 700,
            fontSize: '1.1rem',
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
          }}>
            Back
          </button>
        </Link>
      </div>

      {/* SVG Map Wrapper */}
      <div
        ref={wrapperRef}
        className="svg-map-wrapper"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >

        {/* The World Map with Exact Ocean Components */}
        <svg ref={mapRef} viewBox="0 0 950 620" preserveAspectRatio="xMidYMid meet" className="world-svg" style={{ pointerEvents: 'none' }} xmlns="http://www.w3.org/2000/svg">
          
          <defs>
            <mask id="land-mask">
              <rect x="0" y="0" width="100%" height="100%" fill="white" />
              <use href="#continents-group" className="mask-land" />
            </mask>
          </defs>

          {/* Interactive Oceans Behind the Continents */}
          <g className="oceans" mask="url(#land-mask)" onMouseLeave={() => setHoveredItem(null)}>
            {/* Arctic Ocean */}
            <path className="ocean arctic" d="M0,0 L950,0 L950,110 L0,110 Z" onMouseEnter={() => setHoveredItem('Arctic Ocean')} />

            {/* Southern Ocean */}
            <path className="ocean southern" d="M0,520 L950,520 L950,620 L0,620 Z" onMouseEnter={() => setHoveredItem('Southern Ocean')} />

            {/* Pacific Ocean Left */}
            <path className="ocean pacific" d="M0,110 L260,110 L260,250 L220,310 L305,450 L305,520 L0,520 Z" onMouseEnter={() => setHoveredItem('Pacific Ocean')} />

            {/* Pacific Ocean Right */}
            <path className="ocean pacific" d="M700,110 L950,110 L950,520 L760,520 L760,420 L700,320 Z" onMouseEnter={() => setHoveredItem('Pacific Ocean')} />

            {/* Atlantic Ocean */}
            <path className="ocean atlantic" d="M260,110 L480,110 L480,250 L530,400 L530,520 L305,520 L305,450 L220,310 L260,250 Z" onMouseEnter={() => setHoveredItem('Atlantic Ocean')} />

            {/* Indian Ocean */}
            <path className="ocean indian" d="M480,110 L700,110 L700,320 L760,420 L760,520 L530,520 L530,400 L480,250 Z" onMouseEnter={() => setHoveredItem('Indian Ocean')} />

            {/* Argo Floats Dots */}
            {ARGO_FLOATS.map(f => (
              <circle
                key={f.id}
                cx={f.cx}
                cy={f.cy}
                r={3}
                className="argo-float"
                onMouseEnter={() => setHoveredItem(f.name)}
              />
            ))}
          </g>

          {/* Real World Map Outline */}
          <WorldMapPaths onHover={setHoveredItem} />

        </svg>

        {/* Static Bottom Bar UI */}
        <div className="hover-overlay">
          <div className="hover-overlay-left">FIND A LOCATION</div>
          <div className="hover-overlay-right" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span className="dot" style={{ opacity: hoveredItem ? 1 : 0.2 }}></span>
            <span style={{ fontWeight: 700, fontSize: '1.1rem', letterSpacing: '0.05em' }}>
              {hoveredItem ? hoveredItem.toUpperCase() : 'SELECT A LOCATION'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
