import React, { KeyboardEvent } from 'react';
import { useAppContext } from '../../context/AppContext';
import { Send } from 'lucide-react';

const MessageInput: React.FC = () => {
  const { input, setInput, sendMessage } = useAppContext();

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div className="flex space-x-4">
      <input 
        type="text"
        placeholder="Ask about Barbadian politics..."
        className="flex-1 bg-gray-50/80 backdrop-blur-sm border-2 border-gray-200 focus:border-blue-400 focus:bg-white/90 rounded-2xl px-6 py-4 text-gray-800 placeholder-gray-500 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-blue-200/50"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={handleKeyPress}
      />
      <button 
        onClick={sendMessage}
        className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white px-6 py-4 rounded-2xl font-semibold shadow-lg transform hover:scale-105 transition-all duration-200 flex items-center"
        aria-label="Send message"
      >
        <span className="mr-2 hidden md:inline">Send</span>
        <Send size={18} />
      </button>
    </div>
  );
};

export default MessageInput;