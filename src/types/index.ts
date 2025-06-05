export interface Source {
  date: string;
  speaker: string;
  videoId: string;
  timestamp: number;
}

export interface Message {
  type: 'user' | 'bot';
  text: string;
  timestamp: string;
  sources?: Source[];
}

export interface Video {
  id: string;
  title: string;
  date: string;
  startTime: number;
}

export interface ChatResponse {
  response: string;
  used_knowledge_graph: boolean;
  sources?: Source[];
}