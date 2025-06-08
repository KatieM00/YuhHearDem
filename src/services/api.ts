interface ApiRequest {
  message: string;
  session_id: string;
}

interface ProvenanceItem {
  claim: string;
  youtube_url: string;
  description: string;
  session_date: string;
  confidence: number;
}

interface ApiResponse {
  success: boolean;
  message: string;
  provenance: ProvenanceItem[];
  session_id: string;
  timestamp?: string;
  metadata?: {
    response_time_ms: number;
    model_version: string;
    total_sources: number;
  };
}

const API_URL = 'https://us-central1-yuhheardem.cloudfunctions.net/barbados-parliament-chatbot/chat';

// Generate or retrieve session ID
export const getSessionId = (): string => {
  let sessionId = localStorage.getItem('yuh_hear_dem_session_id');
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('yuh_hear_dem_session_id', sessionId);
  }
  return sessionId;
};

// Extract video ID and timestamp from YouTube URL
export const parseYouTubeUrl = (url: string): { videoId: string; timestamp: number } => {
  const videoId = extractVideoId(url);
  const timestamp = extractTimestamp(url);
  return { videoId, timestamp };
};

const extractVideoId = (url: string): string => {
  // Handle various YouTube URL formats
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
    /youtube\.com\/v\/([a-zA-Z0-9_-]{11})/,
    /youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})/
  ];
  
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match && match[1]) {
      return match[1];
    }
  }
  
  // If no match found, try to extract 11-character alphanumeric string
  const fallbackMatch = url.match(/([a-zA-Z0-9_-]{11})/);
  if (fallbackMatch) {
    return fallbackMatch[1];
  }
  
  // Fallback to default video if parsing fails
  return 'dQw4w9WgXcQ';
};

const extractTimestamp = (url: string): number => {
  // Extract timestamp from URL (t=123s, &t=123, or #t=123)
  const patterns = [
    /[?&]t=(\d+)s?/,
    /#t=(\d+)s?/,
    /&t=(\d+)s?/
  ];
  
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match && match[1]) {
      return parseInt(match[1], 10);
    }
  }
  
  return 0;
};

// Format date for display
const formatDate = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric' 
    });
  } catch {
    return dateString; // Return original if parsing fails
  }
};

// Send message to API
export const sendMessageToApi = async (message: string): Promise<{
  text: string;
  sources?: Array<{
    date: string;
    speaker: string;
    videoId: string;
    timestamp: number;
  }>;
}> => {
  const sessionId = getSessionId();
  
  const requestBody: ApiRequest = {
    message: message.trim(),
    session_id: sessionId
  };

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data: ApiResponse = await response.json();

    if (!data.success) {
      throw new Error(data.message || 'API request failed');
    }

    // Convert API response to expected format
    const sources = data.provenance?.map(item => {
      const { videoId, timestamp } = parseYouTubeUrl(item.youtube_url);
      return {
        date: formatDate(item.session_date),
        speaker: item.description || 'Parliamentary Session',
        videoId,
        timestamp
      };
    }) || [];

    return {
      text: data.message,
      sources: sources.length > 0 ? sources : undefined
    };

  } catch (error) {
    console.error('API Error:', error);
    
    // Return user-friendly error message
    return {
      text: "I'm having trouble connecting to the parliamentary database right now. Please try again in a moment, or ask a different question.",
      sources: undefined
    };
  }
};

// Health check function (optional)
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(API_URL.replace('/chat', '/health'), {
      method: 'GET',
    });
    return response.ok;
  } catch {
    return false;
  }
};