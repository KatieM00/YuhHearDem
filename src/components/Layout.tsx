import React, { ReactNode } from 'react';
import Navbar from './Navbar';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div 
      className="min-h-screen p-4 md:p-6 overflow-x-hidden"
      style={{
        background: `
          radial-gradient(circle at 20% 20%, rgba(251, 191, 36, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 80% 80%, rgba(6, 182, 212, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 40% 70%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
          linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)
        `,
        backgroundAttachment: 'fixed'
      }}
    >
      <Navbar />
      <main>
        {children}
      </main>
    </div>
  );
};

export default Layout;