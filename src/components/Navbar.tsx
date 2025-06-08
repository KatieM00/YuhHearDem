import React from 'react';
import { Info, Building2 } from 'lucide-react';
import { useAppContext } from '../context/AppContext';

const Navbar: React.FC = () => {
  const { setShowMobileParliament, setShowAboutModal } = useAppContext();

  return (
    <nav className="max-w-7xl mx-auto mb-8">
      <div className="bg-white/80 backdrop-blur-xl rounded-3xl shadow-xl border border-white/20 px-6 md:px-8 py-4">
        <div className="flex flex-col space-y-3 md:space-y-0 md:flex-row md:items-center md:justify-between">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <svg width="24" height="16" viewBox="0 0 24 16" className="flex-shrink-0">
              <rect width="8" height="16" fill="#00267F" />
              <rect x="8" width="8" height="16" fill="#FFC72C" />
              <rect x="16" width="8" height="16" fill="#00267F" />
              <path d="M12 3 L10 8 L12 13 L14 8 Z M12 3 L9 4 L12 5 L15 4 Z" fill="#000000" />
            </svg>
            <button 
              onClick={() => window.location.reload()} 
              className="text-xl md:text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent hover:from-blue-700 hover:to-purple-700 transition-all duration-200"
            >
              Yuh Hear Dem
            </button>
          </div>
          
          {/* Mobile Layout - Only buttons */}
          <div className="flex items-center justify-between md:hidden">
            <button 
              onClick={() => setShowAboutModal(true)}
              className="flex items-center space-x-2 bg-gradient-to-r from-blue-100 to-purple-100 hover:from-blue-200 hover:to-purple-200 text-blue-700 hover:text-blue-800 px-4 py-2 rounded-xl font-medium transition-all duration-200 transform hover:scale-105"
            >
              <Info size={16} />
              <span>About</span>
            </button>
            
            <button 
              onClick={() => setShowMobileParliament(true)}
              className="flex items-center space-x-2 bg-gradient-to-r from-yellow-100 to-orange-100 hover:from-yellow-200 hover:to-orange-200 text-orange-700 hover:text-orange-800 px-4 py-2 rounded-xl font-medium transition-all duration-200 transform hover:scale-105"
            >
              <Building2 size={16} />
              <span>Recent</span>
            </button>
          </div>
          
          {/* Desktop Layout - Only About button */}
          <div className="hidden md:flex">
            <button 
              onClick={() => setShowAboutModal(true)}
              className="flex items-center space-x-2 bg-gradient-to-r from-blue-100 to-purple-100 hover:from-blue-200 hover:to-purple-200 text-blue-700 hover:text-blue-800 px-4 py-2 rounded-xl font-medium transition-all duration-200 transform hover:scale-105"
            >
              <Info size={16} />
              <span>About</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;