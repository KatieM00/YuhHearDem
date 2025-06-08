import React, { useEffect } from 'react';
import { useAppContext } from '../../context/AppContext';
import ParliamentPlaceholder from '../video/ParliamentPlaceholder';
import { X } from 'lucide-react';

const MobileParliamentModal: React.FC = () => {
  const { showMobileParliament, setShowMobileParliament } = useAppContext();

  // Check if we're on mobile
  const isMobile = typeof window !== 'undefined' && window.innerWidth < 1024;

  // Close modal when screen size changes to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setShowMobileParliament(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [setShowMobileParliament]);

  // Don't render on desktop or when not shown
  if (!showMobileParliament || !isMobile) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 bg-white lg:hidden">
      {/* Header with close button */}
      <div className="bg-white/80 backdrop-blur-xl border-b border-gray-100 px-6 py-4 flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-800">Parliamentary Coverage</h2>
        <button
          onClick={() => setShowMobileParliament(false)}
          className="w-8 h-8 bg-gray-100 hover:bg-gray-200 rounded-full flex items-center justify-center transition-all duration-200"
          aria-label="Close"
        >
          <X size={18} className="text-gray-600" />
        </button>
      </div>

      {/* Content */}
      <div className="p-6 h-[calc(100vh-80px)] overflow-y-auto">
        <ParliamentPlaceholder />
      </div>
    </div>
  );
};

export default MobileParliamentModal;