'use client';
import { useState } from 'react';
import { Html, Float, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';

const Label = ({ text, show }: { text: string, show: boolean }) => (
  <Html
    position={[0.5, 0, 0]}
    center
    style={{
      transition: 'all 0.3s cubic-bezier(0.25, 1, 0.5, 1)',
      opacity: show ? 1 : 0,
      transform: `scale(${show ? 1 : 0.5}) translate3d(20px, 0, 0)`,
      pointerEvents: 'none',
      whiteSpace: 'nowrap',
      zIndex: 1000,
    }}
  >
    <div style={{
      background: 'rgba(10, 10, 15, 0.85)',
      backdropFilter: 'blur(10px)',
      WebkitBackdropFilter: 'blur(10px)',
      border: '1px solid rgba(0, 210, 255, 0.4)',
      padding: '10px 16px',
      borderRadius: '12px',
      color: 'white',
      fontFamily: 'Outfit, sans-serif',
      fontSize: '0.95rem',
      fontWeight: 500,
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      gap: '12px'
    }}>
      <div style={{ width: '8px', height: '8px', background: '#00d2ff', borderRadius: '50%', boxShadow: '0 0 10px #00d2ff' }}></div>
      {text}
    </div>
  </Html>
);

export default function ArgoFloat3D() {
  const [hovered, setHovered] = useState<string | null>(null);

  const handlePointerOver = (e: any, name: string) => {
    e.stopPropagation();
    setHovered(name);
    document.body.style.cursor = 'pointer';
  };
  
  const handlePointerOut = () => {
    setHovered(null);
    document.body.style.cursor = 'none'; // Our global cursor logic handles 'none'
  };

  return (
    <group position={[0, -0.5, 0]} scale={1.5}>
      <Float speed={2.5} rotationIntensity={0.2} floatIntensity={0.8}>
        
        {/* Antenna */}
        <group position={[0, 2.8, 0]}>
          <mesh 
            onPointerOver={(e) => handlePointerOver(e, 'Antenna')} 
            onPointerOut={handlePointerOut}
          >
            <cylinderGeometry args={[0.02, 0.05, 0.8, 16]} />
            <meshStandardMaterial color="#1a1a1a" roughness={0.7} />
          </mesh>
          <mesh position={[0, 0.4, 0]}>
             <coneGeometry args={[0.08, 0.15, 16]} />
             <meshStandardMaterial color="#111" />
          </mesh>
          <Label text="Antenna: Transmits data to satellites" show={hovered === 'Antenna'} />
        </group>

        {/* CTD Sensor */}
        <group position={[0, 2.0, 0]}>
          <mesh
            onPointerOver={(e) => handlePointerOver(e, 'CTD')} 
            onPointerOut={handlePointerOut}
          >
            <cylinderGeometry args={[0.1, 0.1, 0.8, 32]} />
            <meshStandardMaterial color="#2c3e50" metalness={0.6} roughness={0.4} />
          </mesh>
          {/* CTD Ribs */}
          {[...Array(6)].map((_, i) => (
            <mesh key={i} position={[0, -0.3 + i * 0.12, 0]}>
              <torusGeometry args={[0.11, 0.015, 16, 32]} />
              <meshStandardMaterial color="#1a1a1a" />
            </mesh>
          ))}
          <Label text="CTD Sensor: Conductivity, Temperature, Depth" show={hovered === 'CTD'} />
        </group>

        {/* Top Metal Dome */}
        <mesh position={[0, 1.5, 0]}>
          <sphereGeometry args={[0.35, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshPhysicalMaterial color="#e0e0e0" metalness={0.9} roughness={0.2} clearcoat={1} />
        </mesh>

        {/* Main Glass Casing (Cutaway view) */}
        <mesh position={[0, 0.25, 0]}>
          <cylinderGeometry args={[0.35, 0.35, 2.5, 32]} />
          <meshPhysicalMaterial 
            color="#ffffff" 
            transmission={0.95} 
            opacity={1} 
            metalness={0.2} 
            roughness={0.05} 
            ior={1.5} 
            thickness={0.5} 
            transparent
            envMapIntensity={2}
          />
        </mesh>

        {/* Internal Reservoir */}
        <group position={[0, 1.0, 0]}>
          <mesh
            onPointerOver={(e) => handlePointerOver(e, 'Reservoir')} 
            onPointerOut={handlePointerOut}
          >
            <cylinderGeometry args={[0.25, 0.25, 0.4, 32]} />
            <meshStandardMaterial color="#bdc3c7" metalness={0.8} roughness={0.3} />
          </mesh>
          {/* Ribs on reservoir */}
          {[...Array(5)].map((_, i) => (
            <mesh key={i} position={[0, -0.15 + i * 0.075, 0]}>
              <torusGeometry args={[0.255, 0.01, 16, 32]} />
              <meshStandardMaterial color="#7f8c8d" metalness={0.5} />
            </mesh>
          ))}
          <Label text="Internal Reservoir: Oil storage for buoyancy" show={hovered === 'Reservoir'} />
        </group>

        {/* Hydraulic Pump */}
        <group position={[0, 0.3, 0]}>
          <mesh
            onPointerOver={(e) => handlePointerOver(e, 'Pump')} 
            onPointerOut={handlePointerOut}
          >
            <boxGeometry args={[0.18, 0.6, 0.18]} />
            <meshStandardMaterial color="#f39c12" metalness={0.7} roughness={0.3} />
          </mesh>
          <mesh position={[0, 0.35, 0]}>
             <cylinderGeometry args={[0.05, 0.05, 0.2, 16]} />
             <meshStandardMaterial color="#fff" metalness={0.9} roughness={0.1} />
          </mesh>
          <Label text="Hydraulic Pump: Transfers fluid to change depth" show={hovered === 'Pump'} />
        </group>

        {/* Batteries */}
        <group position={[0, -0.4, 0]}>
          {/* Battery 1 */}
          <mesh
            onPointerOver={(e) => handlePointerOver(e, 'Batteries')} 
            onPointerOut={handlePointerOut}
            position={[-0.18, 0, 0]}
          >
            <cylinderGeometry args={[0.09, 0.09, 0.8, 32]} />
            <meshStandardMaterial color="#c0392b" metalness={0.3} roughness={0.6} />
          </mesh>
          {/* Battery 2 */}
          <mesh
            onPointerOver={(e) => handlePointerOver(e, 'Batteries')} 
            onPointerOut={handlePointerOut}
            position={[0.18, 0, 0]}
          >
            <cylinderGeometry args={[0.09, 0.09, 0.8, 32]} />
            <meshStandardMaterial color="#c0392b" metalness={0.3} roughness={0.6} />
          </mesh>
          <Label text="Batteries: Powers operations for up to 5 years" show={hovered === 'Batteries'} />
        </group>

        {/* Bottom Metal Dome */}
        <mesh position={[0, -1.0, 0]} rotation={[Math.PI, 0, 0]}>
          <sphereGeometry args={[0.35, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
          <meshPhysicalMaterial color="#e0e0e0" metalness={0.9} roughness={0.2} clearcoat={1} />
        </mesh>

        {/* External Bladder */}
        <group position={[0, -1.3, 0]}>
          <mesh
            onPointerOver={(e) => handlePointerOver(e, 'Bladder')} 
            onPointerOut={handlePointerOut}
          >
            <sphereGeometry args={[0.25, 64, 64]} />
            <MeshDistortMaterial color="#2c3e50" distort={0.25} speed={2} roughness={0.7} metalness={0.1} />
          </mesh>
          <Label text="External Bladder: Inflates to rise to surface" show={hovered === 'Bladder'} />
        </group>

      </Float>
    </group>
  );
}
