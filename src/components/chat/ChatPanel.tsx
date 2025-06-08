import React from 'react';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import StarterPrompts from './StarterPrompts';
import { useAppContext } from '../../context/AppContext';
import { MessageSquare } from 'lucide-react';

const ChatPanel: React.FC = () => {
  const { messages, isTyping } = useAppContext();

  return (
    <div className="bg-white/70 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/20 flex flex-col h-full transition-all duration-300 hover:shadow-2xl">
      {/* Header */}
      <div className="p-6 border-b border-gray-100">
        <h2 className="text-xl font-semibold text-gray-800 flex items-center">
          <MessageSquare size={20} className="mr-2" />
          Ask About Barbadian Politics
        </h2>
      </div>

      {/* Messages Area - Fixed height with internal scrolling */}
      <div className="h-[450px] overflow-y-auto p-6 scroll-smooth" id="messages-container">
        {messages.length === 0 ? <StarterPrompts /> : <MessageList />}
        
        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex items-center space-x-2 p-4 animate-pulse">
            <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce\" style={{ animationDelay: '0ms' }}></div>
            <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '600ms' }}></div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-6 border-t border-gray-100">
        <MessageInput />
      </div>
    </div>
  );
};

export default ChatPanel;