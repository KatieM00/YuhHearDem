import React, { useEffect, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';
import { Message as MessageType } from '../../types';

const MessageList: React.FC = () => {
  const { messages, loadVideo } = useAppContext();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
  };

  return (
    <div className="space-y-6">
      {messages.map((message, index) => (
        <div key={index} className={message.type === 'user' ? 'flex justify-end' : 'flex justify-start'}>
          {message.type === 'user' ? (
            <UserMessage message={message} />
          ) : (
            <BotMessage message={message} loadVideo={loadVideo} formatTime={formatTime} />
          )}
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

const UserMessage: React.FC<{ message: MessageType }> = ({ message }) => (
  <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-4 rounded-3xl rounded-br-lg shadow-lg max-w-md transform transition-all duration-300 hover:scale-[1.02] mb-4">
    <p>{message.text}</p>
    <span className="text-xs text-blue-100 mt-2 block">{message.timestamp}</span>
  </div>
);

const BotMessage: React.FC<{ 
  message: MessageType, 
  loadVideo: (videoId: string, startTime: number) => void,
  formatTime: (seconds: number) => string
}> = ({ message, loadVideo, formatTime }) => (
  <div className="bg-white/90 backdrop-blur-sm px-6 py-4 rounded-3xl rounded-bl-lg shadow-lg border border-gray-100 max-w-md transform transition-all duration-300 hover:shadow-xl mb-4">
    <p className="text-gray-800 mb-4">{message.text}</p>
    
    {message.sources && message.sources.length > 0 && (
      <div className="space-y-2">
        <p className="text-sm font-semibold text-gray-600 mb-2">📋 Sources:</p>
        {message.sources.map((source, idx) => (
          <button 
            key={idx}
            onClick={() => loadVideo(source.videoId, source.timestamp)}
            className="block w-full text-left bg-gradient-to-r from-yellow-100 to-orange-100 hover:from-yellow-200 hover:to-orange-200 px-4 py-2 rounded-xl text-sm text-gray-700 transition-all duration-200 transform hover:scale-105"
          >
            📅 {source.date} - {source.speaker}
            {source.timestamp && (
              <span className="ml-2 text-xs text-gray-500">
                ⏱️ {formatTime(source.timestamp)}
              </span>
            )}
          </button>
        ))}
      </div>
    )}
    
    <span className="text-xs text-gray-500 mt-2 block">{message.timestamp}</span>
  </div>
);

export default MessageList;