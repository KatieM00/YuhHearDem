import React from 'react';
import { useAppContext } from '../../context/AppContext';
import { Calendar, Clock, X } from 'lucide-react';

const VideoPlayer: React.FC = () => {
  const { currentVideo, setCurrentVideo } = useAppContext();

  if (!currentVideo) return null;

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  // Construct the YouTube embed URL properly with better error handling
  const constructEmbedUrl = (videoId: string, startTime: number = 0): string => {
    // Ensure videoId is valid (11 characters, alphanumeric + _ -)
    const cleanVideoId = videoId.match(/^[a-zA-Z0-9_-]{11}$/) ? videoId : 'dQw4w9WgXcQ';
    const cleanStartTime = Math.max(0, Math.floor(startTime));
    
    return `https://www.youtube.com/embed/${cleanVideoId}?start=${cleanStartTime}&autoplay=1&rel=0`;
  };

  const embedUrl = constructEmbedUrl(currentVideo.id, currentVideo.startTime);

  return (
    <div className="space-y-4 animate-fadeIn">
      <div className="relative w-full h-64 md:h-80 bg-black rounded-2xl overflow-hidden shadow-lg">
        <iframe 
          key={`${currentVideo.id}-${currentVideo.startTime}`}
          width="100%" 
          height="100%" 
          src={embedUrl}
          title={currentVideo.title}
          className="w-full h-full"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
        ></iframe>
        
        {/* Close Button */}
        <button
          onClick={() => setCurrentVideo(null)}
          className="absolute top-2 right-2 w-8 h-8 bg-black/50 hover:bg-black/70 rounded-full flex items-center justify-center transition-all duration-200"
          aria-label="Close video"
        >
          <X size={16} className="text-white" />
        </button>
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