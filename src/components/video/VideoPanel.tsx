import React from 'react';
import { useAppContext } from '../../context/AppContext';
import VideoPlayer from './VideoPlayer';
import ParliamentPlaceholder from './ParliamentPlaceholder';
import { Film } from 'lucide-react';

const VideoPanel: React.FC = () => {
  const { currentVideo } = useAppContext();

  return (
    <div className="bg-white/70 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 flex flex-col h-full transition-all duration-300 hover:shadow-2xl">
      <div className="p-6 border-b border-gray-100">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center">
          <Film size={20} className="mr-2" />
          Parliamentary Coverage
        </h2>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        {currentVideo ? <VideoPlayer /> : <ParliamentPlaceholder />}
      </div>
    </div>
  );
};

export default VideoPanel;