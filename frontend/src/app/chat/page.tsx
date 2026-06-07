'use client';

import { useState, useEffect, useRef } from 'react';
import GradientBlinds from '../../components/GradientBlinds';
import CustomCursor from '../../components/CustomCursor';
import Link from 'next/link';
import SendButton from '../../components/SendButton/SendButton';
import ShapeGrid from '../../components/ShapeGrid/ShapeGrid';
import { 
  SidebarProvider, 
  Sidebar, 
  SidebarHeader, 
  SidebarContent, 
  SidebarGroup, 
  SidebarGroupLabel, 
  SidebarMenu, 
  SidebarMenuItem, 
  SidebarMenuButton, 
  SidebarFooter,
  SidebarTrigger 
} from '../../components/ui/sidebar';
import { MessageSquare, Map, Settings, Waves, Anchor, Droplets } from 'lucide-react';

const HEADINGS = [
  "What would you like to ask?",
  "How can I help you explore?",
  "What's on your mind today?",
  "Ready to dive into ocean data?",
  "Ask me anything about Argo!"
];

function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Waves size={24} color="#000" />
          <span>FloatChat AI</span>
        </div>
      </SidebarHeader>
      <SidebarContent>

        
        <div style={{ marginTop: 'auto' }}>
          <SidebarGroup>
            <SidebarGroupLabel>Tools</SidebarGroupLabel>
            <SidebarMenu>
              <SidebarMenuItem>
                <Link href="/map" style={{ textDecoration: 'none', color: 'inherit', display: 'block', width: '100%' }}>
                  <SidebarMenuButton>
                    <Map size={18} />
                    <span>Map Direction</span>
                  </SidebarMenuButton>
                </Link>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton>
                  <Settings size={18} />
                  <span>Settings</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroup>
        </div>
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenuButton>
          <div style={{ width: 24, height: 24, borderRadius: '50%', backgroundColor: '#000', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12 }}>
            S
          </div>
          <span>Sakethavasudev</span>
        </SidebarMenuButton>
      </SidebarFooter>
    </Sidebar>
  );
}

export default function ChatPage() {
  const [heading, setHeading] = useState("");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{text: string, sender: 'user' | 'bot', streaming?: boolean}[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const randomHeading = HEADINGS[Math.floor(Math.random() * HEADINGS.length)];
    setHeading(randomHeading);
  }, []);

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
    }
  };

  // Typewriter streaming effect
  const streamBotReply = (fullText: string) => {
    setMessages(prev => [...prev, { text: '', sender: 'bot', streaming: true }]);
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.sender === 'bot') {
          updated[updated.length - 1] = { ...last, text: fullText.slice(0, i) };
        }
        return updated;
      });
      // Keep scrolling as text streams in
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      if (i >= fullText.length) {
        clearInterval(interval);
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.sender === 'bot') {
            updated[updated.length - 1] = { ...last, streaming: false };
          }
          return updated;
        });
      }
    }, 18); // ~18ms per character for a fast but visible typewriter effect
  };

  const handleSend = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isSending) return;

    setIsSending(true);
    setTimeout(() => setIsSending(false), 1500); // matches the animation duration

    const userInput = input.trim();
    setMessages(prev => [...prev, { text: userInput, sender: 'user' }]);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    const fetchResponse = async () => {
      setIsChatLoading(true);
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/demo-query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query: userInput })
        });
        
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        const botReply = data.response || data.answer || data.reply || data.message || (typeof data === 'string' ? data : JSON.stringify(data));
        setIsChatLoading(false);
        streamBotReply(botReply);
      } catch (error) {
        console.error('Error fetching query:', error);
        setIsChatLoading(false);
        streamBotReply("Sorry, I couldn't reach the server right now.");
      }
    };

    fetchResponse();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <SidebarProvider defaultOpen={false}>
      <CustomCursor />
      {/* ShapeGrid background */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: -1, backgroundColor: '#fff' }}>
        <ShapeGrid 
          speed={0.1} 
          squareSize={40}
          direction='diagonal'
          borderColor='rgba(0,0,0,0.1)'
          hoverFillColor='#000'
          shape='square'
          hoverTrailAmount={20}
        />
      </div>
      
      <AppSidebar />
      <SidebarTrigger />
      
      <main className="chat-layout" style={{ flex: 1, paddingLeft: 0, overflow: 'hidden' }}>
        <div style={{ position: 'fixed', top: '14px', right: '2rem', zIndex: 10 }}>
          <Link href="/map" style={{ textDecoration: 'none' }}>
            <button className="dotted-btn">
              Try Map
            </button>
          </Link>
        </div>

        {/* Main Chat Area - Fixed to the exact center of the screen */}
        <div style={{ 
          position: 'fixed',
          top: '80px',
          bottom: '0',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '100%', 
          maxWidth: '760px',
          display: 'flex', 
          flexDirection: 'column',
          zIndex: 5,
          pointerEvents: 'none'
        }}>
          
          {/* Messages scroll area */}
          <div className="messages-scroll-area" style={{
            flex: 1,
            overflowY: 'auto',
            padding: '1rem 1rem 0', 
            display: 'flex',
            flexDirection: 'column',
            scrollbarWidth: 'none',
            pointerEvents: 'auto',
          }}>
            {messages.length === 0 ? (
              <h1 className="chat-header" style={{ 
                margin: 'auto', 
                textAlign: 'center', 
                width: '100%',
                background: 'linear-gradient(135deg, #000000 0%, #666666 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              }}>{heading}</h1>
            ) : (
            messages.map((msg, i) => (
              <div key={i} style={{
                display: 'flex',
                justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                marginBottom: '1rem'
              }}>
                {msg.sender === 'user' ? (
                  // User bubble — right aligned
                  <div style={{
                    maxWidth: '85%',
                    background: 'rgba(255,255,255,0.75)',
                    backdropFilter: 'blur(8px)',
                    borderRadius: '18px 18px 4px 18px',
                    padding: '0.75rem 1.1rem',
                    color: '#111',
                    fontSize: '0.97rem',
                    lineHeight: '1.6',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                    wordBreak: 'break-word',
                  }}>
                    {msg.text}
                  </div>
                ) : (
                  // AI plain text — left aligned
                  <div style={{
                    maxWidth: '90%',
                    color: '#111',
                    fontSize: '0.97rem',
                    lineHeight: '1.8',
                    wordBreak: 'break-word',
                  }}>
                    {msg.text}
                    {msg.streaming && (
                      <span style={{
                        display: 'inline-block',
                        width: '2px',
                        height: '1em',
                        background: '#555',
                        marginLeft: '2px',
                        verticalAlign: 'text-bottom',
                        animation: 'blink 0.7s step-end infinite',
                      }} />
                    )}
                  </div>
                )}
              </div>
            ))
          )}
            {isChatLoading && (
              <div style={{
                display: 'flex',
                justifyContent: 'flex-start',
                marginBottom: '1rem'
              }}>
                <div style={{
                  maxWidth: '90%',
                  padding: '0.5rem 0',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}>
                  <span className="loading-dot"></span>
                  <span className="loading-dot"></span>
                  <span className="loading-dot"></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div style={{
            padding: '0.5rem 1rem 1.5rem 1rem',
            width: '100%',
            flexShrink: 0,
            pointerEvents: 'auto',
          }}>
            <form
              onSubmit={handleSend}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'flex-end',
                gap: '0.75rem',
                background: 'rgba(255, 255, 255, 0.5)',
                backdropFilter: 'blur(28px) saturate(200%)',
                WebkitBackdropFilter: 'blur(28px) saturate(200%)',
                borderRadius: '16px',
                padding: '0.5rem 1.2rem',
                boxShadow: '0 8px 32px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.6)',
                border: '1px solid rgba(0,0,0,0.1)',
              }}
            >
              <textarea
                ref={textareaRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Ask FloatChat AI..."
                rows={1}
                style={{
                  flex: 1,
                  background: 'transparent',
                  border: 'none',
                  outline: 'none',
                  resize: 'none',
                  fontFamily: 'inherit',
                  fontSize: '1rem',
                  color: '#2a2a2a',
                  lineHeight: '1.6',
                  maxHeight: '180px',
                  overflowY: 'auto',
                  padding: '8px 0',
                  scrollbarWidth: 'none',
                }}
              />
              <SendButton isSending={isSending} />
            </form>
          </div>

          <style>{`
            @keyframes blink {
              0%, 100% { opacity: 1; }
              50% { opacity: 0; }
            }
            .loading-dot {
              width: 6px;
              height: 6px;
              background-color: #999;
              border-radius: 50%;
              animation: bounce 1.4s infinite ease-in-out both;
            }
            .loading-dot:nth-child(1) { animation-delay: -0.32s; }
            .loading-dot:nth-child(2) { animation-delay: -0.16s; }
            @keyframes bounce {
              0%, 80%, 100% { transform: scale(0); }
              40% { transform: scale(1); }
            }
            textarea::placeholder {
              color: rgba(0, 0, 0, 0.5);
            }
            textarea::-webkit-scrollbar {
              display: none;
            }
            .messages-scroll-area::-webkit-scrollbar {
              display: none;
            }
          `}</style>
        </div>
      </main>
    </SidebarProvider>
  );
}
