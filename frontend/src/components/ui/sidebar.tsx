import React, { createContext, useContext, useState, ReactNode } from 'react';
import './sidebar.css';
import { Menu } from 'lucide-react';

interface SidebarContextType {
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  toggle: () => void;
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined);

export function useSidebar() {
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error('useSidebar must be used within a SidebarProvider');
  }
  return context;
}

export function SidebarProvider({ children, defaultOpen = true }: { children: ReactNode, defaultOpen?: boolean }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const toggle = () => setIsOpen(!isOpen);

  return (
    <SidebarContext.Provider value={{ isOpen, setIsOpen, toggle }}>
      <div className={`sidebar-provider ${isOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
        {children}
      </div>
    </SidebarContext.Provider>
  );
}

export function Sidebar({ children, side = 'left' }: { children: ReactNode, side?: 'left' | 'right' }) {
  const { isOpen } = useSidebar();
  return (
    <aside className={`shad-sidebar shad-sidebar-${side} ${isOpen ? 'expanded' : 'collapsed'}`}>
      <div className="shad-sidebar-inner">
        {children}
      </div>
    </aside>
  );
}

export function SidebarHeader({ children }: { children: ReactNode }) {
  return <div className="shad-sidebar-header">{children}</div>;
}

export function SidebarContent({ children }: { children: ReactNode }) {
  return <div className="shad-sidebar-content">{children}</div>;
}

export function SidebarFooter({ children }: { children: ReactNode }) {
  return <div className="shad-sidebar-footer">{children}</div>;
}

export function SidebarGroup({ children }: { children: ReactNode }) {
  return <div className="shad-sidebar-group">{children}</div>;
}

export function SidebarGroupLabel({ children }: { children: ReactNode }) {
  return <div className="shad-sidebar-group-label">{children}</div>;
}

export function SidebarMenu({ children }: { children: ReactNode }) {
  return <ul className="shad-sidebar-menu">{children}</ul>;
}

export function SidebarMenuItem({ children }: { children: ReactNode }) {
  return <li className="shad-sidebar-menu-item">{children}</li>;
}

export function SidebarMenuButton({ children, isActive = false, onClick }: { children: ReactNode, isActive?: boolean, onClick?: () => void }) {
  return (
    <button 
      className={`shad-sidebar-menu-button ${isActive ? 'active' : ''}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

export function SidebarTrigger() {
  const { toggle } = useSidebar();
  return (
    <button className="shad-sidebar-trigger" onClick={toggle} aria-label="Toggle Sidebar">
      <Menu size={20} />
    </button>
  );
}
