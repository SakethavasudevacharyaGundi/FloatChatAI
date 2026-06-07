const fs = require('fs');

const svgContent = fs.readFileSync('world_map.svg', 'utf8');

// Find the first <path> to start extracting
const firstPathIndex = svgContent.indexOf('<path');
// Find the closing </svg>
const lastSvgIndex = svgContent.lastIndexOf('</svg>');

if (firstPathIndex === -1 || lastSvgIndex === -1) {
  console.error('Could not parse SVG');
  process.exit(1);
}

// Extract just the inner SVG elements (paths)
let paths = svgContent.substring(firstPathIndex, lastSvgIndex);

// React requires some SVG attributes to be camelCase, but since this is standard SVG paths, 
// there might be some issues with 'xml:space' or 'inkscape:*' or 'sodipodi:*'.
// Let's blindly clean up known bad React attributes if they exist on paths.
paths = paths.replace(/inkscape:[a-z-]+="[^"]*"/g, '');
paths = paths.replace(/sodipodi:[a-z-]+="[^"]*"/g, '');
paths = paths.replace(/xmlns:[a-z-]+="[^"]*"/g, '');
paths = paths.replace(/style="[^"]*"/g, ''); // Fix React style prop error
paths = paths.replace(/class="[^"]*"/g, ''); // Fix React class prop error (should be className but we don't need it)

const componentCode = `
import React from 'react';

export default function WorldMapPaths() {
  return (
    <g className="continents" pointerEvents="auto" fill="#050508" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8">
      ${paths}
    </g>
  );
}
`;

fs.writeFileSync('src/app/map/WorldMapPaths.tsx', componentCode);
console.log('Successfully created WorldMapPaths.tsx');
