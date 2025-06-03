import React from 'react';
import { useAppContext } from '../../context/AppContext';
import { Calendar, Clock } from 'lucide-react';

const VideoPlayer: React.FC = () => {
  const { currentVideo } = useAppContext();

  if (!currentVideo) return null;

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  return (
    <div className="space-y-4 animate-fadeIn">
      <div className="w-full h-64 md:h-80 bg-black rounded-2xl overflow-hidden shadow-lg">
        <iframe 
          width="100%" 
          height="100%" 
          src={`https://www.youtube.com/embed/${currentVideo.id}?start=${currentVideo.startTime || 0}&autoplay=1`}
          title={currentVideo.title}
          className="w-full h-full"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        ></iframe>
      </div>
      
      <div className="bg-gradient-to-r from-blue-50 to-teal-50 rounded-2xl p-4 border border-blue-200">
        <h3 className="font-semibold text-gray-800 text-lg">{currentVideo.title}</h3>
        <div className="flex flex-col sm:flex-row sm:justify-between mt-2">
          <p className="text-sm text-gray-600 flex items-center">
            <Calendar size={14} className="mr-1" />
            {currentVideo.date}
          </p>
          <p className="text-sm text-gray-600 flex items-center mt-1 sm:mt-0">
            <Clock size={14} className="mr-1" />
            Playing from {formatTime(currentVideo.startTime || 0)}
          </p>
        </div>
      </div>

      <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-4 shadow-md border border-gray-100">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">About this session:</h4>
        <p className="text-xs text-gray-600">
          This video shows a parliamentary session where representatives discuss important matters
          related to Barbadian governance. The timestamp corresponds to the specific moment where
          the topic mentioned in the chat was discussed.
        </p>
      </div>
    </div>
  );
};

export default VideoPlayer;