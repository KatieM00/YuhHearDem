import React from 'react';
import Layout from './components/Layout';
import ChatPanel from './components/chat/ChatPanel';
import VideoPanel from './components/video/VideoPanel';
import MobileParliamentModal from './components/mobile/MobileParliamentModal';
import MobileVideoModal from './components/mobile/MobileVideoModal';
import AboutModal from './components/AboutModal';
import { AppProvider } from './context/AppContext';

function App() {
  return (
    <AppProvider>
      <Layout>
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 h-[calc(100vh-140px)]">
          {/* Chat Panel - Full width on mobile, left on desktop */}
          <div className="lg:col-span-1">
            <ChatPanel />
          </div>
          
          {/* Video Panel - Hidden on mobile, right on desktop */}
          <div className="hidden lg:block lg:col-span-1">
            <VideoPanel />
          </div>
        </div>
        
        {/* Mobile-only modals */}
        <MobileParliamentModal />
        <MobileVideoModal />
        
        {/* About Modal - Available on all screen sizes */}
        <AboutModal />
      </Layout>
    </AppProvider>
  );
}

export default App;