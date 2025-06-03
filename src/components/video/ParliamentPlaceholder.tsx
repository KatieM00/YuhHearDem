import React from 'react';
import { useAppContext } from '../../context/AppContext';
import { Building2, Clock, Users, Briefcase, Calendar, Film } from 'lucide-react';

const ParliamentPlaceholder: React.FC = () => {
  const { setInput, sendMessage } = useAppContext();

  const handleWatchSession = () => {
    setInput("Show me the latest parliamentary session");
    setTimeout(() => {
      sendMessage();
    }, 100);
  };

  return (
    <div className="h-full bg-gradient-to-br from-blue-50 via-teal-50 to-green-50 rounded-2xl p-6 border-2 border-dashed border-blue-200 flex flex-col">
      <div className="text-center mb-6 animate-fadeIn">
        <div className="text-6xl mb-4 flex justify-center">🏛️</div>
        <h3 className="text-2xl font-bold text-gray-800 mb-2">Latest Parliamentary Session</h3>
        <p className="text-gray-600">Most recent debate information</p>
      </div>
      
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-lg flex-grow">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-gray-800 flex items-center">
            <Calendar size={18} className="mr-2" />
            January 15, 2025
          </h4>
          <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium flex items-center mt-2 sm:mt-0 w-fit">
            <Clock size={14} className="mr-1" />
            2:30 PM - 5:45 PM
          </span>
        </div>
        
        <div className="mb-4">
          <h5 className="font-semibold text-gray-700 mb-2 flex items-center">
            <Users size={16} className="mr-2" />
            Present Members (12)
          </h5>
          <div className="flex flex-wrap gap-2">
            {['Prime Minister', 'Opposition Leader', 'Minister of Finance', 'Minister of Health'].map((member, idx) => (
              <span key={idx} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                {member}
              </span>
            ))}
            <span className="bg-gray-100 text-gray-600 px-3 py-1 rounded-full text-sm">+8 more</span>
          </div>
        </div>
        
        <div className="mb-6">
          <h5 className="font-semibold text-gray-700 mb-2 flex items-center">
            <Briefcase size={16} className="mr-2" />
            Key Topics
          </h5>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
              <span className="text-sm text-gray-700">Economic Recovery Plan (45 min)</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-green-400 rounded-full"></span>
              <span className="text-sm text-gray-700">Healthcare Infrastructure (30 min)</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
              <span className="text-sm text-gray-700">Tourism Development (25 min)</span>
            </div>
          </div>
        </div>
        
        <button 
          onClick={handleWatchSession}
          className="w-full bg-gradient-to-r from-yellow-400 to-orange-400 hover:from-yellow-500 hover:to-orange-500 text-white py-3 px-4 rounded-xl font-semibold transform hover:scale-105 transition-all duration-200 flex items-center justify-center"
        >
          <Film size={18} className="mr-2" />
          Watch Full Session
        </button>
      </div>
    </div>
  );
};

export default ParliamentPlaceholder;