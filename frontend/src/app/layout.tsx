import './globals.css';
import { ReactNode } from 'react';

export const metadata = {
  title: 'FloatChat AI - Modern Intelligent Assistant',
  description: 'Experience the next generation of conversational AI with FloatChat.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
