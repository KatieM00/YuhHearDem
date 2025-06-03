import React from 'react';
import Layout from './components/Layout';
import ChatPanel from './components/chat/ChatPanel';
import VideoPanel from './components/video/VideoPanel';
import { AppProvider } from './context/AppContext';

function App() {
  return (
    <AppProvider>
      <Layout>
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 h-[calc(100vh-140px)]">
          {/* Chat Panel - Left */}
          <div className="lg:col-span-1">
            <ChatPanel />
          </div>
          
          {/* Video Panel - Right */}
          <div className="lg:col-span-1">
            <VideoPanel />
          </div>
        </div>
      </Layout>
    </AppProvider>
  );
}

export default App;