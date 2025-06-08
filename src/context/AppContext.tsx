import React, { createContext, useState, useContext, ReactNode } from 'react';
import { Message, Video } from '../types';
import { sendMessageToApi } from '../services/api';

interface AppContextType {
  messages: Message[];
  currentVideo: Video | null;
  input: string;
  isTyping: boolean;
  addMessage: (message: Message) => void;
  setCurrentVideo: (video: Video | null) => void;
  setInput: (input: string) => void;
  setIsTyping: (isTyping: boolean) => void;
  sendMessage: () => void;
  loadVideo: (videoId: string, startTime: number) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentVideo, setCurrentVideo] = useState<Video | null>(null);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  const addMessage = (message: Message) => {
    setMessages((prev) => [...prev, message]);
  };

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const userInput = input.trim();
    
    // Add user message immediately
    const userMessage: Message = {
      type: 'user',
      text: userInput,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    addMessage(userMessage);
    setInput('');
    setIsTyping(true);

    try {
      // Call real API
      const apiResponse = await sendMessageToApi(userInput);
      
      // Create bot message from API response
      const botMessage: Message = {
        type: 'bot',
        text: apiResponse.text,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        sources: apiResponse.sources
      };
      
      addMessage(botMessage);
      
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      const errorMessage: Message = {
        type: 'bot',
        text: "I'm having trouble connecting to the parliamentary database right now. Please try again in a moment.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      
      addMessage(errorMessage);
    } finally {
      setIsTyping(false);
    }
  };

  const loadVideo = (videoId: string, startTime: number) => {
    // Find source details from messages to get better title and date
    let videoTitle = "Parliamentary Session";
    let videoDate = "Recent Session";
    
    for (const message of messages) {
      if (message.type === 'bot' && message.sources) {
        for (const source of message.sources) {
          if (source.videoId === videoId && source.timestamp === startTime) {
            videoTitle = source.speaker;
            videoDate = source.date;
            break;
          }
        }
      }
    }
    
    setCurrentVideo({
      id: videoId,
      title: videoTitle,
      date: videoDate,
      startTime: startTime
    });
  };

  return (
    <AppContext.Provider
      value={{
        messages,
        currentVideo,
        input,
        isTyping,
        addMessage,
        setCurrentVideo,
        setInput,
        setIsTyping,
        sendMessage,
        loadVideo
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};