import json
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static test data for Barbados parliamentary discussions
SAMPLE_RESPONSES = [
    {
        "response": "The 2024 Budget allocated $2.8 billion for healthcare improvements, with specific emphasis on upgrading equipment at the Queen Elizabeth Hospital. The Minister of Health outlined plans to reduce waiting times and improve emergency services.",
        "provenance": [
            {
                "claim": "The 2024 Budget allocated $2.8 billion for healthcare improvements",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s",
                "description": "Budget Debate 2024 - Health Ministry Allocation",
                "session_date": "2024-03-15",
                "confidence": 0.95
            },
            {
                "claim": "plans to reduce waiting times and improve emergency services",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=20s", 
                "description": "Minister of Health - QEH Improvement Plans",
                "session_date": "2024-03-15",
                "confidence": 0.88
            }
        ]
    },
    {
        "response": "The Tourism Recovery Act was passed with bipartisan support, establishing a $50 million fund to support local tourism businesses affected by recent challenges. The Act includes provisions for sustainable tourism development and support for small hotel operators.",
        "provenance": [
            {
                "claim": "Tourism Recovery Act was passed with bipartisan support",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
                "description": "Parliamentary Session - Tourism Recovery Bill Final Reading",
                "session_date": "2024-02-28",
                "confidence": 0.92
            },
            {
                "claim": "establishing a $50 million fund to support local tourism businesses",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=80s",
                "description": "Tourism Minister - Recovery Fund Details",
                "session_date": "2024-02-28",
                "confidence": 0.91
            }
        ]
    },
    {
        "response": "The Education Reform Bill introduced mandatory digital literacy programs in all primary schools. The Prime Minister emphasized that this initiative will prepare Barbadian students for the digital economy and ensure equal access to technology education.",
        "provenance": [
            {
                "claim": "Education Reform Bill introduced mandatory digital literacy programs",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=20s",
                "description": "Education Reform Debate - Digital Literacy Mandate",
                "session_date": "2024-01-20",
                "confidence": 0.94
            },
            {
                "claim": "ensure equal access to technology education",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=70s",
                "description": "Prime Minister's Address - Education Equality",
                "session_date": "2024-01-20",
                "confidence": 0.87
            }
        ]
    },
    {
        "response": "The Climate Resilience Infrastructure Project received approval for $150 million in funding. The project includes upgrading drainage systems, strengthening coastal defenses, and implementing renewable energy solutions across government buildings.",
        "provenance": [
            {
                "claim": "Climate Resilience Infrastructure Project received approval for $150 million",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=44s",
                "description": "Infrastructure Committee - Climate Project Approval",
                "session_date": "2024-04-10",
                "confidence": 0.96
            },
            {
                "claim": "upgrading drainage systems, strengthening coastal defenses",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=56s",
                "description": "Public Works Minister - Infrastructure Details",
                "session_date": "2024-04-10",
                "confidence": 0.89
            }
        ]
    },
    {
        "response": "The Small Business Support Amendment increased the maximum loan amount under the government's entrepreneurship program from $25,000 to $75,000. The Minister of Commerce stated this change will help more Barbadians start and expand their businesses.",
        "provenance": [
            {
                "claim": "Small Business Support Amendment increased maximum loan from $25,000 to $75,000",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=33s",
                "description": "Commerce Committee - Small Business Loan Amendment",
                "session_date": "2024-03-22",
                "confidence": 0.93
            },
            {
                "claim": "help more Barbadians start and expand their businesses",
                "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=41s",
                "description": "Minister of Commerce - Entrepreneurship Support",
                "session_date": "2024-03-22",
                "confidence": 0.85
            }
        ]
    }
]

def get_relevant_response(user_message):
    """
    Select a relevant response based on keywords in the user message.
    In a real implementation, this would be replaced by LLM processing.
    """
    message_lower = user_message.lower()
    
    # Simple keyword matching for demo purposes
    if any(word in message_lower for word in ['health', 'hospital', 'medical', 'healthcare']):
        return SAMPLE_RESPONSES[0]
    elif any(word in message_lower for word in ['tourism', 'hotel', 'travel', 'visitor']):
        return SAMPLE_RESPONSES[1]
    elif any(word in message_lower for word in ['education', 'school', 'student', 'digital']):
        return SAMPLE_RESPONSES[2]
    elif any(word in message_lower for word in ['climate', 'environment', 'infrastructure', 'drainage']):
        return SAMPLE_RESPONSES[3]
    elif any(word in message_lower for word in ['business', 'loan', 'entrepreneur', 'commerce']):
        return SAMPLE_RESPONSES[4]
    else:
        # Return a random response if no keywords match
        return random.choice(SAMPLE_RESPONSES)

def process_chat_request(request_json):
    """
    Process chat request and return response.
    """
    try:
        if not request_json:
            return {'error': 'No JSON data provided'}, 400
        
        user_message = request_json.get('message', '').strip()
        session_id = request_json.get('session_id', 'default')
        conversation_history = request_json.get('history', [])
        
        if not user_message:
            return {'error': 'Message cannot be empty'}, 400
        
        logger.info(f"Processing message from session {session_id}: {user_message[:100]}...")
        
        # Get relevant response (in real implementation, this would call the LLM)
        response_data = get_relevant_response(user_message)
        
        # Prepare response
        response = {
            'success': True,
            'message': response_data['response'],
            'provenance': response_data['provenance'],
            'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': {
                'response_time_ms': random.randint(200, 800),  # Simulated response time
                'model_version': 'static-test-v1.0',
                'total_sources': len(response_data['provenance'])
            }
        }
        
        logger.info(f"Returning response with {len(response_data['provenance'])} provenance items")
        
        return response, 200
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return {
            'success': False,
            'error': 'Internal server error',
            'message': 'An error occurred while processing your request'
        }, 500

def health_check_response():
    """Health check response."""
    return {
        'status': 'healthy',
        'service': 'barbados-parliament-chatbot',
        'version': '1.0.0',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }, 200

def get_session_history_response(session_id):
    """
    Get conversation history for a session.
    Currently returns mock data - in real implementation would fetch from database.
    """
    # Mock conversation history
    mock_history = [
        {
            'user_message': 'Tell me about healthcare spending',
            'bot_response': 'The 2024 Budget allocated $2.8 billion for healthcare improvements...',
            'timestamp': '2024-06-08T11:45:00Z'
        }
    ]
    
    return {
        'session_id': session_id,
        'history': mock_history,
        'total_messages': len(mock_history)
    }, 200

# Google Cloud Function entry point
def main(request):
    """
    Google Cloud Function entry point.
    """
    try:
        # Set CORS headers
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return ('', 204, headers)
        
        # Route requests based on path and method
        path = request.path if hasattr(request, 'path') else '/'
        method = request.method
        
        if path == '/health' and method == 'GET':
            response_data, status_code = health_check_response()
        elif path == '/chat' and method == 'POST':
            response_data, status_code = process_chat_request(request.get_json())
        elif path.startswith('/sessions/') and method == 'GET' and '/history' in path:
            # Extract session_id from path like /sessions/abc123/history
            path_parts = path.strip('/').split('/')
            if len(path_parts) >= 3:
                session_id = path_parts[1]
                response_data, status_code = get_session_history_response(session_id)
            else:
                response_data, status_code = {'error': 'Invalid session path'}, 400
        elif method == 'POST':  # Default to chat for root POST requests
            response_data, status_code = process_chat_request(request.get_json())
        else:
            response_data, status_code = {'error': 'Endpoint not found'}, 404
        
        # Return response with headers
        return (json.dumps(response_data), status_code, {**headers, 'Content-Type': 'application/json'})
        
    except Exception as e:
        logger.error(f"Error in Cloud Function: {str(e)}")
        error_response = {'error': 'Internal server error', 'message': str(e)}
        return (json.dumps(error_response), 500, {'Content-Type': 'application/json'})

# Initialize Flask app for local development
app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    """Flask route for local development"""
    response_data, status_code = process_chat_request(request.get_json())
    return jsonify(response_data), status_code

@app.route('/health', methods=['GET'])
def health_check():
    """Flask route for local development"""
    response_data, status_code = health_check_response()
    return jsonify(response_data), status_code

@app.route('/sessions/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """Flask route for local development"""
    response_data, status_code = get_session_history_response(session_id)
    return jsonify(response_data), status_code

# For local development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)