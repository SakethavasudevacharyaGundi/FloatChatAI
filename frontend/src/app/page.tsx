'use client';

import { useEffect, useRef, useState } from 'react';
import CardNav from '../components/CardNav/CardNav';
import SharedBackground from '../components/SharedBackground';
import PixelCard from '../components/PixelCard';
import TextPressure from '../components/TextPressure';
import CustomCursor from '../components/CustomCursor';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

function SectionBackground({ color, className }: { color: string, className: string }) {
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', zIndex: 0, pointerEvents: 'none' }}>
      {[...Array(5)].map((_, i) => (
        <div 
          key={i} 
          className={`bg-bar ${className}`} 
          style={{ flex: 1, backgroundColor: color, transform: 'scaleY(0)', transformOrigin: 'bottom' }} 
        />
      ))}
    </div>
  );
}

export default function Home() {
  const mainRef = useRef<HTMLElement>(null);
  const [isLightBg, setIsLightBg] = useState(false);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Setup scale-reveal for the top hero elements if needed
      gsap.utils.toArray('.scale-reveal-hero').forEach((elem: any) => {
        gsap.fromTo(
          elem,
          { scale: 0.8, opacity: 0 },
          {
            scale: 1,
            opacity: 1,
            ease: 'power2.out',
            scrollTrigger: {
              trigger: elem,
              scroller: mainRef.current,
              start: 'top 85%',
              end: 'top 20%',
              scrub: 1
            }
          }
        );
      });

      const sections = [
        { id: '#section-2', barClass: 'bar-2' },
        { id: '#section-3', barClass: 'bar-3' },
        { id: '#section-4', barClass: 'bar-4' }
      ];

      sections.forEach(({ id, barClass }) => {
        // Initial hidden states
        gsap.set(`.${barClass}`, { scaleY: 0 });
        gsap.set(`${id} .content-wrapper`, { opacity: 0, y: 50 });

        const tl = gsap.timeline({
          scrollTrigger: {
            trigger: id,
            scroller: mainRef.current,
            start: 'top 50%',
            // onEnter: play (grow), onLeave: none (stay), onEnterBack: none (stay), onLeaveBack: reverse (shrink)
            toggleActions: 'play none none reverse'
          }
        });

        // 1. Animate the 5 vertical columns into place sequentially from left to right over 1 second
        tl.to(`.${barClass}`, { 
            scaleY: 1, 
            duration: 1, 
            stagger: 0.1, // sequentially from left to right
            ease: 'power3.inOut' 
          })
          // 2. Animate the section content to appear
          .to(`${id} .content-wrapper`, { 
            opacity: 1, 
            y: 0, 
            duration: 0.8, 
            ease: 'power2.out' 
          }, "-=0.5"); // Start fading in before bars are completely finished
      });

    }, mainRef);

    // Custom JS Wheel Handler for perfect 1-to-1 discrete section snapping
    const main = mainRef.current;
    let isAnimating = false;
    let wheelAccumulator = 0;
    let scrollTimeout: NodeJS.Timeout;

    const handleWheel = (e: WheelEvent) => {
      // 1. Prevent native scroll entirely. This completely bypasses the browser's CSS sticky-snap bug.
      e.preventDefault(); 
      
      // 2. Clear inertia from trackpads when the user stops scrolling for 50ms
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        wheelAccumulator = 0;
      }, 50);

      if (isAnimating || !main) return;

      // 3. Accumulate tiny trackpad scrolls until they reach a threshold
      wheelAccumulator += e.deltaY;

      if (Math.abs(wheelAccumulator) > 40) {
        const direction = wheelAccumulator > 0 ? 1 : -1;
        wheelAccumulator = 0; // Reset immediately to prevent multi-skipping
        
        const vh = window.innerHeight;
        const currentScroll = main.scrollTop;
        const currentIndex = Math.round(currentScroll / vh);
        
        let targetIndex = currentIndex + direction;
        targetIndex = Math.max(0, Math.min(3, targetIndex)); // Clamp between 4 sections (0 to 3)

        const targetScroll = targetIndex * vh;

        // Only animate if there's a new section to go to
        if (Math.abs(targetScroll - currentScroll) > 10) {
          isAnimating = true;
          
          // Use a GSAP proxy to manually drive the scrollTop on every frame.
          // This completely bypasses the browser's native smooth scroll engine, 
          // crushing the CSS sticky-snap bug that was locking the upward scroll!
          const scrollProxy = { y: currentScroll };
          
          gsap.to(scrollProxy, {
            y: targetScroll,
            duration: 1.2, // Smooth, luxurious scroll
            ease: 'power3.inOut',
            onUpdate: () => {
              if (mainRef.current) {
                mainRef.current.scrollTop = scrollProxy.y;
              }
            },
            onComplete: () => {
              isAnimating = false;
            }
          });
        }
      }
    };

    const handleMainScroll = () => {
      if (mainRef.current) {
        // Change color when scrolled past 40% of the first section
        if (mainRef.current.scrollTop > window.innerHeight * 0.4) {
          setIsLightBg(true);
        } else {
          setIsLightBg(false);
        }
      }
    };

    if (main) {
      // passive: false is CRITICAL so we can call e.preventDefault()
      main.addEventListener('wheel', handleWheel, { passive: false });
      main.addEventListener('scroll', handleMainScroll);
    }

    return () => {
      ctx.revert();
      if (main) {
        main.removeEventListener('wheel', handleWheel);
        main.removeEventListener('scroll', handleMainScroll);
      }
    };
  }, []);

  const navTextColor = isLightBg ? '#000' : '#fff';

  const navItems = [
    {
      label: "Explore",
      bgColor: "rgba(0, 0, 0, 0.03)",
      textColor: navTextColor,
      links: [
        { label: "Dashboard", href: "/", ariaLabel: "Go to Dashboard" },
        { label: "Interactive Map", href: "/map", ariaLabel: "Go to Map" }
      ]
    },
    {
      label: "Chat", 
      bgColor: "rgba(0, 0, 0, 0.03)",
      textColor: navTextColor,
      links: [
        { label: "FloatChat AI", href: "/chat", ariaLabel: "Chat with AI" },
        { label: "Past Chats", href: "/chat", ariaLabel: "View Past Chats" }
      ]
    },
    {
      label: "About",
      bgColor: "rgba(0, 0, 0, 0.03)", 
      textColor: navTextColor,
      links: [
        { label: "Argo Project", href: "https://www.noaa.gov/", ariaLabel: "Learn about Argo" },
        { label: "Open Source", href: "https://github.com/NOAA", ariaLabel: "Open Source Data" },
        { label: "Developer", href: "https://github.com/SakethavasudevacharyaGundi", ariaLabel: "About Developer" }
      ]
    }
  ];

  const navLogo = (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: navTextColor, transition: 'color 0.3s' }}>
      <svg viewBox="0 0 24 24" width="24" height="24" xmlns="http://www.w3.org/2000/svg" style={{ stroke: navTextColor, fill: 'none', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round', transition: 'stroke 0.3s' }}>
        <path d="M2 12C2 12 5 9 12 12C19 15 22 12 22 12" />
        <path d="M2 17C2 17 5 14 12 17C19 20 22 17 22 17" />
        <path d="M2 7C2 7 5 4 12 7C19 10 22 7 22 7" />
      </svg>
      FloatChat AI
    </div>
  );

  return (
    <>
      <CustomCursor />
      <SharedBackground />
      <CardNav 
        logo={navLogo}
        items={navItems}
        baseColor="transparent"
        menuColor={navTextColor}
        buttonBgColor="#fff"
        buttonTextColor="#111"
        ctaText="Try Chat Bot"
        ctaHref="/chat"
      />

      <main 
        className="main-container" 
        ref={mainRef}
        style={{
          position: 'relative',
          zIndex: 10,
          height: '100vh',
          width: '100vw',
          overflowY: 'scroll',
          overflowX: 'hidden'
          /* REMOVED scrollSnapType to fix 'goto' jumping during GSAP animation */
        }}
      >
        {/* Section 1: Hero (Remains Live Video + Ripple) */}
        <section 
          className="story-section" 
          id="home" 
          style={{ 
            position: 'sticky',
            top: 0,
            height: '100vh', 
            width: '100%',
            scrollSnapAlign: 'start',
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            justifyContent: 'center',
            zIndex: 1
          }}
        >
          <div style={{ position: 'relative', height: '180px', width: '100%', maxWidth: '800px' }}>
            <TextPressure 
              text="FloatChat AI"
              flex={true}
              alpha={false}
              stroke={false}
              width={true}
              weight={true}
              italic={true}
              textColor="#ffffff"
              minFontSize={36}
            />
          </div>
          <p className="hero-subtitle scale-reveal-hero">
            The world's first intelligent assistant powered by global Argo float data. 
            Dive into the ocean's metrics and converse with the data.
          </p>
        </section>

        {/* Section 2: Row 1 */}
        <section 
          id="section-2"
          style={{ 
            position: 'sticky',
            top: 0,
            height: '100vh', 
            width: '100%',
            scrollSnapAlign: 'start',
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            overflow: 'hidden',
            zIndex: 2
          }}
        >
          <SectionBackground color="#ffffff" className="bar-2" />
          <div className="content-wrapper" style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'center', gap: '5rem', maxWidth: '1400px', width: '100%', padding: '0 2rem' }}>
            <div style={{ flex: '1', display: 'flex', justifyContent: 'center' }}>
              <PixelCard variant="blue" style={{ width: '100%', maxWidth: '550px', height: '650px' }}>
                <img src="/Argo_1.jpg" alt="Argo Float" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.9 }} />
              </PixelCard>
            </div>
            
            <div style={{ flex: '1' }}>
              <div style={{ position: 'relative', height: '100px', marginBottom: '2rem' }}>
                <TextPressure 
                  text="The Crown Jewel"
                  textColor="#000000"
                  minFontSize={24}
                />
              </div>
              <p style={{ fontSize: '1.4rem', lineHeight: '1.8', color: '#000000' }}>
                Argo is the crown jewel of the ocean observing system. A global array of thousands of free-drifting profiling floats that measures the temperature and salinity of the upper ocean.
              </p>
            </div>
          </div>
        </section>

        {/* Section 3: Row 2 */}
        <section 
          id="section-3"
          style={{ 
            position: 'sticky',
            top: 0,
            height: '100vh', 
            width: '100%',
            scrollSnapAlign: 'start',
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            overflow: 'hidden',
            zIndex: 3
          }}
        >
          <SectionBackground color="#fcfaf5" className="bar-3" />
          <div className="content-wrapper" style={{ position: 'relative', zIndex: 1, display: 'flex', alignItems: 'center', gap: '5rem', maxWidth: '1400px', width: '100%', padding: '0 2rem', flexDirection: 'row-reverse' }}>
            <div style={{ flex: '1', display: 'flex', justifyContent: 'center' }}>
              <PixelCard variant="blue" style={{ width: '100%', maxWidth: '550px', height: '650px' }}>
                <img src="/Argo_2.avif" alt="Argo Float 2" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.9 }} />
              </PixelCard>
            </div>
            
            <div style={{ flex: '1' }}>
              <div style={{ position: 'relative', height: '100px', marginBottom: '2rem' }}>
                <TextPressure 
                  text="Open Source"
                  textColor="#000000"
                  minFontSize={24}
                />
              </div>
              <p style={{ fontSize: '1.4rem', lineHeight: '1.8', color: '#000000' }}>
                The data from Argo floats is managed globally and kept completely open source. This project helps analyze that vast repository of data, transforming complex metrics into simple insights.
              </p>
            </div>
          </div>
        </section>

        {/* Section 4: About Developer */}
        <section 
          id="section-4"
          style={{ 
            position: 'sticky',
            top: 0,
            height: '100vh', 
            width: '100%',
            scrollSnapAlign: 'start',
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            overflow: 'hidden',
            zIndex: 4
          }}
        >
          <SectionBackground color="#f0f9ff" className="bar-4" />
          <div className="content-wrapper" style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', maxWidth: '800px', padding: '0 2rem' }}>
            <div style={{ position: 'relative', height: '120px', marginBottom: '1rem', width: '100%' }}>
              <TextPressure 
                text="Sakethavasudev"
                textColor="#0ea5e9"
                minFontSize={24}
              />
            </div>
            <p style={{ fontSize: '1.5rem', lineHeight: '1.8', color: '#003366', fontWeight: 500, marginBottom: '2rem' }}>
              Full-Stack Developer & Ocean Data Enthusiast
            </p>
            <p style={{ fontSize: '1.2rem', lineHeight: '1.8', color: '#1e293b' }}>
              I built FloatChat AI to bridge the gap between complex oceanographic data and simple human interaction. Using real-time data from the global Argo float array, this platform transforms millions of metrics into conversational insights.
            </p>
          </div>
        </section>

      </main>
    </>
  );
}
