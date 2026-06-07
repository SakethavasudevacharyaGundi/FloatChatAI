'use client';

import { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

const MAX_RIPPLES = 30;

const RippleMaterial = {
  uniforms: {
    uTexture1: { value: null },
    uTexture2: { value: null },
    uBlend: { value: 0 },
    uTime: { value: 0 },
    uMousePos: { value: new Array(MAX_RIPPLES).fill(new THREE.Vector2(0, 0)) },
    uMouseTime: { value: new Array(MAX_RIPPLES).fill(0) },
    uResolution: { value: new THREE.Vector2() }
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D uTexture1;
    uniform sampler2D uTexture2;
    uniform float uBlend;
    uniform float uTime;
    uniform vec2 uMousePos[${MAX_RIPPLES}];
    uniform float uMouseTime[${MAX_RIPPLES}];
    uniform vec2 uResolution;
    varying vec2 vUv;

    void main() {
      vec2 st = vUv;
      float aspect = uResolution.x / uResolution.y;
      vec2 offset = vec2(0.0);
      float totalIntensity = 0.0;
      
      for(int i = 0; i < ${MAX_RIPPLES}; i++) {
        float age = uTime - uMouseTime[i];
        if(age > 0.0 && age < 4.0) {
          vec2 center = uMousePos[i];
          vec2 p1 = st; p1.x *= aspect;
          vec2 p2 = center; p2.x *= aspect;
          float dist = distance(p1, p2);
          
          float waveSpeed = 0.5;
          float wavePeak = age * waveSpeed;
          float distToPeak = abs(dist - wavePeak);
          
          float intensity = exp(-distToPeak * 20.0) * exp(-age * 1.5);
          float ripple = sin((dist - age * waveSpeed) * 40.0) * intensity;
          
          offset += normalize(st - center) * ripple * 0.02;
          totalIntensity += intensity;
        }
      }
      
      vec2 distortedUV = st + offset;
      float videoAspect = 16.0 / 9.0;
      float screenAspect = uResolution.x / uResolution.y;
      vec2 finalUV = distortedUV;
      finalUV = (finalUV - 0.5) * 0.95 + 0.5;
      
      if (screenAspect > videoAspect) {
        float yRatio = videoAspect / screenAspect;
        finalUV.y = (finalUV.y - 0.5) * yRatio + 0.5;
      } else {
        float xRatio = screenAspect / videoAspect;
        finalUV.x = (finalUV.x - 0.5) * xRatio + 0.5;
      }
      
      vec4 color1 = texture2D(uTexture1, finalUV);
      vec4 color2 = texture2D(uTexture2, finalUV);
      vec4 texColor = mix(color1, color2, uBlend);
      
      texColor.rgb += vec3(0.5, 0.8, 1.0) * totalIntensity * 0.1;
      gl_FragColor = texColor;
    }
  `
};

const VideoPlane = () => {
  const { viewport, size } = useThree();
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  const [video1, setVideo1] = useState<HTMLVideoElement | null>(null);
  const [video2, setVideo2] = useState<HTMLVideoElement | null>(null);

  useEffect(() => {
    const v1 = document.createElement('video');
    const v2 = document.createElement('video');
    
    [v1, v2].forEach(vid => {
      vid.crossOrigin = 'Anonymous';
      vid.muted = true;
      vid.playsInline = true;
    });

    fetch('/live_ocean.mp4')
      .then(res => res.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        v1.src = url;
        v2.src = url;
        v1.play().catch(console.error);
        setVideo1(v1);
        setVideo2(v2);
      });

    return () => {
      if (v1.src) URL.revokeObjectURL(v1.src);
    };
  }, []);

  const texture1 = useMemo(() => {
    if (!video1) return null;
    const tex = new THREE.VideoTexture(video1);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    return tex;
  }, [video1]);

  const texture2 = useMemo(() => {
    if (!video2) return null;
    const tex = new THREE.VideoTexture(video2);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    return tex;
  }, [video2]);

  const rippleIndex = useRef(0);
  const mouseTimes = useRef(new Array(MAX_RIPPLES).fill(0));
  const mousePositions = useRef(new Array(MAX_RIPPLES).fill(new THREE.Vector2(0, 0)));
  const mouseRef = useRef({ x: 0, y: 0, moved: false });
  const lastMousePos = useRef({ x: 0, y: 0 });
  
  const activeVideo = useRef(1);
  const uBlend = useRef(0);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = e.clientX / window.innerWidth;
      mouseRef.current.y = 1.0 - (e.clientY / window.innerHeight);
      mouseRef.current.moved = true;
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useFrame((state, delta) => {
    if (mouseRef.current.moved) {
      const dx = mouseRef.current.x - lastMousePos.current.x;
      const dy = mouseRef.current.y - lastMousePos.current.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist > 0.01) {
        const idx = rippleIndex.current;
        mousePositions.current[idx] = new THREE.Vector2(mouseRef.current.x, mouseRef.current.y);
        mouseTimes.current[idx] = state.clock.elapsedTime;
        rippleIndex.current = (idx + 1) % MAX_RIPPLES;
        lastMousePos.current = { x: mouseRef.current.x, y: mouseRef.current.y };
      }
      mouseRef.current.moved = false;
    }

    // Crossfade Logic
    const fadeDuration = 2.0; // 2 second crossfade
    if (video1 && video2 && video1.duration) {
      if (activeVideo.current === 1) {
        if (video1.currentTime >= video1.duration - fadeDuration) {
          if (video2.paused) {
            video2.currentTime = 0;
            video2.play().catch(() => {});
          }
          uBlend.current = Math.min(1.0, uBlend.current + (1.0 / fadeDuration) * delta);
          if (video1.currentTime >= video1.duration - 0.1) {
            video1.pause();
            activeVideo.current = 2;
          }
        } else {
          uBlend.current = 0.0;
        }
      } else {
        if (video2.currentTime >= video2.duration - fadeDuration) {
          if (video1.paused) {
            video1.currentTime = 0;
            video1.play().catch(() => {});
          }
          uBlend.current = Math.max(0.0, uBlend.current - (1.0 / fadeDuration) * delta);
          if (video2.currentTime >= video2.duration - 0.1) {
            video2.pause();
            activeVideo.current = 1;
          }
        } else {
          uBlend.current = 1.0;
        }
      }
    }

    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      if (texture1) materialRef.current.uniforms.uTexture1.value = texture1;
      if (texture2) materialRef.current.uniforms.uTexture2.value = texture2;
      materialRef.current.uniforms.uBlend.value = uBlend.current;
      materialRef.current.uniforms.uMousePos.value = mousePositions.current;
      materialRef.current.uniforms.uMouseTime.value = mouseTimes.current;
      materialRef.current.uniforms.uResolution.value.set(size.width, size.height);
    }
  });

  return (
    <mesh>
      <planeGeometry args={[viewport.width, viewport.height]} />
      <shaderMaterial 
        ref={materialRef}
        args={[RippleMaterial]}
        transparent={true}
      />
    </mesh>
  );
};

export default function RippleBackground() {
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: -1, background: '#001a33' }}>
      <Canvas camera={{ position: [0, 0, 1] }}>
        <VideoPlane />
      </Canvas>
    </div>
  );
}
