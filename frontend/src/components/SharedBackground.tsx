'use client';

import { useEffect, useRef } from 'react';
import RippleBackground from './RippleBackground';

export default function SharedBackground() {
  return (
    <div className="background-layer">
      <RippleBackground />
    </div>
  );
}
