import React, { createContext, useState, useContext, ReactNode } from 'react';
import axios from 'axios';
import { Message, Video, ChatResponse } from '../types';
import { sampleMessages } from '../data/sampleData';

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

    // Add user message
    const userMessage: Message = {
      type: 'user',
      text: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    addMessage(userMessage);
    setInput('');
    setIsTyping(true);

    try {
      // Call backend API
      const response = await axios.post<ChatResponse>('/api/chat', {
        question: input
      });

      const botMessage: Message = {
        type: 'bot',
        text: response.data.response,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        sources: response.data.sources
      };

      addMessage(botMessage);
    } catch (error) {
      // Fallback to sample data if API fails
      const mockResponse = sampleMessages.find(m => m.type === 'bot') || {
        type: 'bot',
        text: '🔍 I found some information about that topic in recent parliamentary sessions.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        sources: [
          {
            date: 'Jan 15, 2025',
            speaker: 'Parliamentary Session',
            videoId: 'dQw4w9WgXcQ',
            timestamp: 930
          }
        ]
      };
      
      addMessage(mockResponse as Message);
    } finally {
      setIsTyping(false);
    }
  };

  const loadVideo = (videoId: string, startTime: number) => {
    let videoTitle = "Parliamentary Session";
    let videoDate = "January 15, 2025";
    
    for (const message of messages) {
      if (message.type === 'bot' && message.sources) {
        for (const source of message.sources) {
          if (source.videoId === videoId) {
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