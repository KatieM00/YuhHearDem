import React from 'react';
import { MessageSquare } from 'lucide-react';

const Navbar: React.FC = () => {
  return (
    <nav className="max-w-7xl mx-auto mb-8">
      <div className="bg-white/80 backdrop-blur-xl rounded-3xl shadow-xl border border-white/20 px-8 py-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-2xl" role="img" aria-label="Barbados Flag">🇧🇧</span>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Yuh Hear Dem
            </h1>
          </div>
          <div className="text-sm text-gray-600 flex items-center mt-2 md:mt-0">
            <MessageSquare size={16} className="mr-1" />
            <span>Making Barbadian Politics Accessible</span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;