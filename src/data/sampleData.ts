import { Message } from '../types';

export const sampleMessages: Message[] = [
  {
    type: 'user',
    text: 'What has the PM said about economic reform?',
    timestamp: '2:30 PM'
  },
  {
    type: 'bot',
    text: '🔍 The Prime Minister has outlined several economic reform initiatives in recent sessions. Key points include digital transformation of government services, support for small businesses through tax incentives, and a focus on sustainable tourism. These were highlighted in the January budget address.',
    timestamp: '2:31 PM',
    sources: [
      {
        date: 'Jan 15, 2025',
        speaker: 'PM Economic Speech',
        videoId: 'dQw4w9WgXcQ',
        timestamp: 930
      },
      {
        date: 'Jan 12, 2025',
        speaker: 'Budget Discussion',
        videoId: 'dQw4w9WgXcQ', 
        timestamp: 1205
      }
    ]
  },
  {
    type: 'user',
    text: 'What about healthcare improvements?',
    timestamp: '2:35 PM'
  },
  {
    type: 'bot',
    text: '🏥 Healthcare has been a major focus in recent parliamentary debates. The Minister of Health announced a $50M investment in hospital infrastructure, plans for a new medical center in Bridgetown, and expanded telehealth services for rural communities. Opposition has questioned the timeline for implementation.',
    timestamp: '2:36 PM',
    sources: [
      {
        date: 'Jan 18, 2025',
        speaker: 'Healthcare Debate',
        videoId: 'dQw4w9WgXcQ',
        timestamp: 425
      }
    ]
  }
];