#!/usr/bin/env python3
"""
YuhHearDem Web Service - Separated Files Version
===============================================

FastAPI web service with separate HTML, CSS, and JS files.
"""

import os
import sys
import json
import asyncio
import logging
import uuid
import re # Import regex for cleaner cleaning
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field
import uvicorn

# Add markdown and BeautifulSoup imports for processing
import markdown
from bs4 import BeautifulSoup

# Add the current directory to Python path so we can import yuhheardem_core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Google GenAI warnings
logging.getLogger('google_genai.types').setLevel(logging.ERROR)

# Try to import our core system and rdflib components for parsing state changes
try:
    # Import namespaces and necessary rdflib components
    from yuhheardem_core import YuhHearDemADK, BBP, SCHEMA, PROV
    from google.adk.runners import Runner
    from google.genai.types import Content, Part
    from rdflib import Graph, URIRef # Import Graph and URIRef for parsing state
    logger.info("✅ Successfully imported YuhHearDem core components and utilities")
except ImportError as e:
    logger.error(f"❌ Failed to import YuhHearDem core or dependencies: {e}")
    logger.error("Make sure yuhheardem_core.py is in the same directory and rdflib is installed (`pip install rdflib fastmcp google-adk`)")
    sys.exit(1)


# Global variables
yuhheardem_system = None

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question about Parliament")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Session ID (will be generated if not provided)")

class QueryResponse(BaseModel):
    session_id: str
    user_id: str
    message_id: str
    status: str
    message: Optional[str] = None

# Model for session clearing request
class ClearSessionRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Session ID to clear") # Session ID is required to know which session to clear


class AgentEvent(BaseModel):
    type: str
    agent: str
    message: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None

def convert_markdown_to_html_and_filter_links(markdown_text: str) -> str:
    """
    Convert markdown to HTML and remove all links that don't contain youtube.com
    
    Args:
        markdown_text: The markdown text to process
        
    Returns:
        HTML string with only YouTube links preserved
    """
    try:
        # Convert markdown to HTML
        html = markdown.markdown(markdown_text, extensions=['extra', 'codehilite'])
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all anchor tags
        links = soup.find_all('a')
        
        # Process each link
        for link in links:
            href = link.get('href', '')
            
            # Check if the link contains youtube.com
            if 'youtube.com' not in href.lower():
                # Remove the link but keep the text content
                link.replace_with(link.get_text())
                logger.debug(f"Removed non-YouTube link: {href}")
            else:
                logger.debug(f"Preserved YouTube link: {href}")
        
        # Return the processed HTML
        processed_html = str(soup)
        logger.info(f"Converted markdown to HTML and filtered links. Original length: {len(markdown_text)}, Processed length: {len(processed_html)}")
        
        return processed_html
        
    except Exception as e:
        logger.error(f"Error processing markdown to HTML: {e}")
        # Fall back to original markdown text if processing fails
        return markdown_text

def format_sse_event(event_type: str, agent: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    """Format Server-Sent Event data according to SSE protocol."""
    
    # Create the event data structure
    event_data = {
        "type": event_type,
        "agent": agent,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add optional data if provided
    if data:
        event_data["data"] = data
    
    # Format as SSE event
    # SSE format requires 'data:' prefix and double newline terminator
    return f"data: {json.dumps(event_data)}\n\n"

async def process_query_with_realtime_events(query: str, user_id: str, session_id: str) -> AsyncGenerator[str, None]:
    """Process query with REAL backend events streaming in real-time."""

    try:
        # Initial event
        yield format_sse_event("query_start", "System", f"Processing query: {query[:50]}...")

        logger.info(f"🔄 Starting real-time core system processing for session {session_id}...")

        # Create or get session
        if not yuhheardem_system.current_session or str(yuhheardem_system.current_session.id) != session_id:
             if yuhheardem_system.current_session:
                  logger.info(f"Switching session from {yuhheardem_system.current_session.id[:8]}... to {session_id[:8]}.... Clearing previous conversation history.")
                  yuhheardem_system.conversation_history = []

             try:
                 yuhheardem_system.current_session = await yuhheardem_system.session_service.create_session(
                    session_id=session_id,
                    app_name="YuhHearDem",
                    user_id=user_id
                 )
                 logger.info(f"✅ Created new session: {session_id[:8]}...")
             except Exception as e:
                 logger.warning(f"Failed to create session with ID {session_id[:8]}...: {e}")
                 yuhheardem_system.current_session = await yuhheardem_system.session_service.create_session(
                    app_name="YuhHearDem",
                    user_id=user_id
                 )
                 logger.info(f"✅ Created fallback session: {yuhheardem_system.current_session.id[:8]}...")

             yield format_sse_event("session_info", "System", f"Using session: {yuhheardem_system.current_session.id[:8]}...")

        # Create runner for conversational agent
        runner = Runner(
            agent=yuhheardem_system.conversational_agent,
            session_service=yuhheardem_system.session_service,
            app_name="YuhHearDem"
        )

        yield format_sse_event("runner_created", "System", "Agent runner initialized")

        # Build conversation context
        context = ""
        if yuhheardem_system.conversation_history:
            context = "\n\nPREVIOUS CONVERSATION:\n"
            history_slice = yuhheardem_system.conversation_history[-4:]
            for exchange in history_slice:
                context += f"User: {exchange['user']}\n"
                assistant_text = exchange.get('assistant', '')
                if len(assistant_text) > 300:
                    assistant_text = assistant_text[:300] + "..."
                context += f"Assistant: {assistant_text}\n\n"

        # Create the prompt with context
        full_prompt = f"{context}CURRENT QUERY: {query}"
        user_message = Content(role="user", parts=[Part.from_text(text=full_prompt)])

        yield format_sse_event("message_prepared", "System", "User message prepared, starting agent execution...")

        # Run the conversational agent
        events = runner.run(
            user_id=user_id,
            session_id=yuhheardem_system.current_session.id,
            new_message=user_message
        )

        # Track state for immediate conversational responses
        conversational_response_sent = False
        final_response_text = ""
        event_count = 0
        agents_seen = set()
        
        # NEW: Track conversational text separately from function calls
        conversational_text_buffer = ""
        pending_transfer = False

        # Process events synchronously
        for event in events:
            event_count += 1
            sse_agent = event.author or "System"

            # Handle ConversationalAgent events
            if sse_agent == "ConversationalAgent" and not conversational_response_sent:
                
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            
                            # Collect text content from ConversationalAgent
                            if hasattr(part, 'text') and part.text:
                                text_content = part.text.strip()
                                if text_content:
                                    conversational_text_buffer = text_content
                                    logger.info(f"📝 ConversationalAgent text captured: {text_content[:50]}...")
                            
                            # Detect function call (transfer to research)
                            elif hasattr(part, 'function_call') and part.function_call:
                                func_name = part.function_call.name
                                if func_name == "transfer_to_agent":
                                    pending_transfer = True
                                    logger.info(f"🔄 ConversationalAgent transfer detected")

                # If we have collected conversational text and detected transfer, send the immediate response
                if conversational_text_buffer and (pending_transfer or event_count > 1) and not conversational_response_sent:
                    logger.info(f"🎯 Sending immediate conversational response: {conversational_text_buffer[:50]}...")
                    
                    # Convert markdown to HTML and filter links for conversational response
                    processed_response = convert_markdown_to_html_and_filter_links(conversational_text_buffer)
                    
                    yield format_sse_event("immediate_response", "ConversationalAgent", "Response ready", {
                        "response": processed_response,
                        "message_id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "type": "conversational"
                    })
                    
                    conversational_response_sent = True
                    
                    # Update conversation history with the original text (not HTML)
                    yuhheardem_system.conversation_history.append({
                        "user": query,
                        "assistant": conversational_text_buffer
                    })
                    
                    # Clear the buffer
                    conversational_text_buffer = ""

            # Announce agent start the first time we see output from them
            if sse_agent in ["ConversationalAgent", "ResearcherAgent", "ProvenanceAgent", "WriterAgent"] and sse_agent not in agents_seen:
                 agents_seen.add(sse_agent)
                 if sse_agent == "ResearcherAgent":
                      yield format_sse_event("researcher_agent", sse_agent, "Starting parliamentary research searches...")
                 elif sse_agent == "ProvenanceAgent":
                      yield format_sse_event("provenance_agent", sse_agent, "Enriching data with video sources...")
                 elif sse_agent == "WriterAgent":
                      yield format_sse_event("writer_agent", sse_agent, "Synthesizing findings into response...")

            # Process event content (text and tool calls/responses)
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_content = part.text
                            # Only accumulate text content if it's from the WriterAgent for the final response body
                            if sse_agent == "WriterAgent":
                                final_response_text += text_content

                        # Handle function calls - signal to frontend that a tool is being used
                        elif hasattr(part, 'function_call') and part.function_call:
                            func_name = part.function_call.name
                            func_args = part.function_call.args
                            if func_name == "transfer_to_agent":
                                target_agent = func_args.get('agent_name', 'Unknown')
                                yield format_sse_event("agent_transfer", sse_agent, f"Transferring to {target_agent}...")
                            elif "search" in func_name.lower():
                                query_text = func_args.get('query', 'Unknown query')
                                search_msg = f"Searching parliamentary sessions for '{query_text[:40]}'"
                                if "hybrid_search" in func_name: 
                                    search_msg = f"Searching parliamentary discussions about '{query_text[:40]}'"
                                elif "authority_search" in func_name: 
                                    search_msg = f"Looking for what parliamentary leaders said about '{query_text[:40]}'"
                                elif "topic_search" in func_name: 
                                    search_msg = f"Exploring detailed discussions on '{query_text[:40]}'"
                                yield format_sse_event("search_started", sse_agent or "ResearcherAgent", search_msg + "...")

            # Handle state changes
            if event.actions and event.actions.state_delta:
                 if 'cumulative_turtle_results' in event.actions.state_delta or 'turtle_results' in event.actions.state_delta:
                     cumulative_turtle_count = len(yuhheardem_system.current_session.state.get("cumulative_turtle_results", []))
                     yield format_sse_event("data_collected", sse_agent or "ResearcherAgent", f"Collected parliamentary datasets. Total knowledge: {cumulative_turtle_count} datasets.")

                 elif 'cumulative_enriched_turtle' in event.actions.state_delta or 'enriched_turtle' in event.actions.state_delta:
                     enriched_data_str = event.actions.state_delta.get('cumulative_enriched_turtle', event.actions.state_delta.get('enriched_turtle', ''))

                     video_count = 0
                     if enriched_data_str and not str(enriched_data_str).startswith("Error") and "no parliamentary entities found" not in str(enriched_data_str).lower():
                         try:
                             g = Graph()
                             g.bind("bbp", BBP)
                             g.bind("schema", SCHEMA)
                             g.bind("prov", PROV)
                             g.parse(data=str(enriched_data_str), format='turtle')
                             video_count = len(list(g.subjects(predicate=URIRef(SCHEMA["url"]))))
                         except Exception as e:
                             logger.warning(f"Failed to parse or count videos from enriched data for event: {e}")
                             video_count = "some"

                     yield format_sse_event("data_enriched", sse_agent or "ProvenanceAgent", f"Parliamentary data enriched with {video_count} video sources.")

        # Send final WriterAgent response if we have one
        if final_response_text.strip():
            logger.info(f"🎯 Sending detailed research response from WriterAgent...")
            
            message_id = str(uuid.uuid4())

            # Convert markdown to HTML and filter links for the final response
            processed_final_response = convert_markdown_to_html_and_filter_links(final_response_text)

            # Update conversation history with the original text (not HTML)
            if yuhheardem_system.conversation_history and yuhheardem_system.conversation_history[-1]["user"] == query:
                yuhheardem_system.conversation_history[-1]["assistant"] = final_response_text
            else:
                yuhheardem_system.conversation_history.append({
                    "user": query,
                    "assistant": final_response_text
                })

            # Trim history if too long
            if len(yuhheardem_system.conversation_history) > 10:
                yuhheardem_system.conversation_history = yuhheardem_system.conversation_history[-10:]
                logger.info("Conversation history trimmed.")

            # Send final detailed response payload with processed HTML
            yield format_sse_event("final_response", "WriterAgent", "Detailed research response ready", {
                "response": processed_final_response,
                "message_id": message_id,
                "session_id": session_id,
                "type": "research"
            })

        yield format_sse_event("processing_complete", "System", f"Processed {event_count} agent events.")
        yield format_sse_event("stream_complete", "System", "Query processing complete.")

    except Exception as e:
        logger.error(f"Real-time query processing error: {e}")
        import traceback
        traceback.print_exc()
        yield format_sse_event("error", "System", f"Error processing query: {str(e)}")


# Initialize web service
def create_web_service() -> YuhHearDemADK:
    """Create the YuhHearDem web service instance."""

    # Get environment variables
    mcp_endpoint = os.getenv('MCP_ENDPOINT', 'http://localhost:8003/sse') # Default to localhost
    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')

    if not google_api_key:
        logger.error("GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY environment variable required")
        # In a real app, you might want to raise an exception here or have a stricter check
        # For now, allow None but expect the ADK to fail initialization later if needed.
        # raise ValueError("GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY environment variable required")

    logger.info(f"🔧 Creating web service with MCP endpoint: {mcp_endpoint}")

    return YuhHearDemADK(mcp_endpoint, google_api_key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""

    # Startup
    logger.info("🚀 Starting YuhHearDem Web Service...")

    # Initialize web service
    global yuhheardem_system
    try:
        yuhheardem_system = create_web_service()
        logger.info("✅ YuhHearDem core system created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create YuhHearDem system: {e}")
        # Depending on how critical ADK initialization is, you might want to exit
        # For now, log and proceed, but mark system as None
        yuhheardem_system = None # Ensure it's None if creation fails


    # Test MCP connection if system initialized
    if yuhheardem_system:
        logger.info("🔧 Testing MCP connection...")
        try:
            if await yuhheardem_system.test_connection():
                logger.info("✅ MCP connection successful")
            else:
                logger.warning("⚠️ MCP connection failed - some features may not work")
        except Exception as e:
            logger.warning(f"⚠️ MCP connection test error: {e}")
    else:
        logger.warning("⚠️ YuhHearDem system failed to initialize. MCP connection test skipped.")


    logger.info("✅ YuhHearDem Web Service ready!")

    yield

    # Shutdown
    logger.info("👋 Shutting down YuhHearDem Web Service...")
    # Perform any necessary cleanup


# Create FastAPI app
app = FastAPI(
    title="YuhHearDem Parliamentary Research API",
    description="AI-powered Parliamentary research system for Barbados Parliament with real-time streaming",
    version="2.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web interface."""
    # Generate a new session ID for the UI on page load if not provided? Or handle in JS?
    # Let JS handle session ID generation and persistence.
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("🏥 Health check requested")

    if not yuhheardem_system:
        logger.warning("⚠️ System not initialized")
        # Indicate if ADK core failed to initialize
        return {"status": "unhealthy", "error": "System not initialized", "mcp_connection": "unknown"}

    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        mcp_status = await yuhheardem_system.test_connection()
        health_data["mcp_connection"] = mcp_status
        logger.info(f"🔗 MCP connection status: {mcp_status}")
        if not mcp_status:
            health_data["status"] = "degraded" # Mark as degraded if MCP fails

    except Exception as e:
        mcp_status = False
        health_data["mcp_connection"] = False
        health_data["status"] = "degraded" # Mark as degraded on MCP test error
        health_data["mcp_error"] = str(e)
        logger.error(f"Health check MCP test failed: {e}")


    logger.info(f"📊 Returning health data: {health_data['status']}")
    return health_data

# Fix 1: Update the session clear endpoint in webmain_separated.py
@app.post("/session/clear")
async def clear_session_endpoint(request: ClearSessionRequest):
    """Clear session state for a specific session."""

    if not yuhheardem_system:
        raise HTTPException(status_code=500, detail="System not initialized")

    logger.info(f"🧹 Received request to clear session: {request.session_id} for user: {request.user_id}")

    try:
        # Invalidate the session in the service (InMemorySessionService doesn't support deletion)
        # For InMemory, the best we can do is clear the *global* current_session
        # if it matches the requested one, and clear the global conversation history.
        # This is a limitation of the InMemorySessionService for multi-user scenarios.
        cleared_global = False
        if (yuhheardem_system.current_session and 
            hasattr(yuhheardem_system.current_session, 'id') and 
            str(yuhheardem_system.current_session.id) == request.session_id):
             yuhheardem_system.current_session = None
             yuhheardem_system.conversation_history = [] # Clear associated history
             cleared_global = True
             logger.info(f"✅ Global current_session cleared as it matched request ID: {request.session_id[:8]}...")
        else:
             # If the requested session is not the current global one, we cannot clear it
             # using InMemorySessionService. We can still clear the history associated
             # with this request's user/session ID if the session service supported it.
             # But with InMemory, history is global anyway.
             current_id = "None" if not yuhheardem_system.current_session else str(yuhheardem_system.current_session.id)[:8] + "..."
             logger.warning(f"Requested session {request.session_id[:8]}... does not match current global session ({current_id}). Cannot clear specific non-current session with InMemorySessionService.")

        return {"status": "success", "message": f"Session {request.session_id[:8]}... clear request processed. Global session state and history cleared: {cleared_global}."}

    except Exception as e:
        logger.error(f"Session clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Process query with REAL-TIME Server-Sent Events streaming."""

    if not yuhheardem_system:
        # Check for API key existence if system is None due to failed init
        api_key_set = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')
        if not api_key_set:
             raise HTTPException(status_code=500, detail="System not initialized: Missing GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY")
        else:
            raise HTTPException(status_code=500, detail="System not initialized: Check backend logs for ADK/MCP errors")


    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"🔍 Real-time stream query: {request.query[:50]}... | User: {request.user_id} | Session: {session_id}")

    async def event_generator():
        # The core processing function is an async generator now
        async for event_data in process_query_with_realtime_events(
            request.query, request.user_id, session_id
        ):
            yield event_data

    # Note: text/plain is used for SSE by some clients/frameworks, text/event-stream is standard.
    # ADK example uses text/plain. Let's stick to text/plain for now.
    return StreamingResponse(
        event_generator(),
        media_type="text/plain", # Or "text/event-stream"
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id # Include session ID in header
        }
    )

@app.post("/query", response_model=QueryResponse)
async def query_non_stream(request: QueryRequest):
    """Process query without streaming (returns final result only)."""

    if not yuhheardem_system:
        api_key_set = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')
        if not api_key_set:
             raise HTTPException(status_code=500, detail="System not initialized: Missing GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY")
        else:
            raise HTTPException(status_code=500, detail="System not initialized: Check backend logs for ADK/MCP errors")

    session_id = request.session_id or str(uuid.uuid4()) # Generate session ID even for non-stream if none provided
    # message_id generated by the core process_query now

    logger.info(f"🔍 Non-stream query: {request.query[:50]}... | User: {request.user_id} | Session: {session_id}")

    try:
        # Process query directly via core system
        # yuhheardem_system.process_query will create/use a session internally, but it's
        # less robust for multi-user than the streaming approach which respects session_id.
        # For non-stream, we default user_id to "user".
        response_content, status_info = await yuhheardem_system.process_query(request.query)

        if status_info.get("success", False):
             # Convert markdown to HTML and filter links for non-stream response
             processed_response = convert_markdown_to_html_and_filter_links(response_content)
             
             return QueryResponse(
                 session_id=session_id, # Use the session_id from the request or generated
                 user_id=request.user_id,
                 message_id=str(uuid.uuid4()), # Generate message ID here
                 status="success",
                 message=processed_response
             )
        else:
             raise HTTPException(status_code=500, detail=response_content)


    except Exception as e:
        logger.error(f"Non-stream query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "webmain:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True # Auto-reload during development
    )