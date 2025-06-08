import React, { useEffect } from 'react';
import { useAppContext } from '../../context/AppContext';
import VideoPlayer from '../video/VideoPlayer';
import { X } from 'lucide-react';

const MobileVideoModal: React.FC = () => {
  const { currentVideo, setCurrentVideo } = useAppContext();

  // Check if we're on mobile
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 1024;

  // Close modal when screen size changes to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        // Don't auto-close video on desktop resize, let desktop handle it
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Don't render on desktop or when no video
  if (!currentVideo || !isMobile) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 bg-white lg:hidden">
      {/* Header with close button */}
      <div className="bg-white/80 backdrop-blur-xl border-b border-gray-100 px-6 py-4 flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-800">Parliamentary Video</h2>
        <button
          onClick={() => setCurrentVideo(null)}
          className="w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center transition-all duration-200"
          aria-label="Close video"
        >
          <X size={18} className="text-gray-600" />
        </button>
      </div>

      {/* Content */}
      <div className="p-6 h-[calc(100vh-80px)] overflow-y-auto">
        <VideoPlayer />
      </div>
    </div>
  );
};

export default MobileVideoModal;