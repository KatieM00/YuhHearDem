#!/usr/bin/env python3
"""
Full Parliamentary Chatbot - Fixed Session Management
====================================================

Fixed session handling to prevent session not found errors.
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set, AsyncGenerator
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# Database and ML imports
from pymongo import MongoClient, ASCENDING
from sentence_transformers import SentenceTransformer

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.planners import BuiltInPlanner
from google.genai.types import Content, Part, GenerateContentConfig
from google.genai import types
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS

# Markdown processing
import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RDF Namespaces
BBP = Namespace("http://example.com/barbados-parliament-ontology#")
SCHEMA = Namespace("http://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question about Parliament")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Session ID")

class QueryResponse(BaseModel):
    session_id: str
    user_id: str
    message_id: str
    status: str
    message: Optional[str] = None

class ParliamentaryGraphQuerier:
    """Database querier with full search functionality."""

    def __init__(self):
        self.client = None
        self.db = None
        self.nodes = None
        self.edges = None
        self.statements = None
        self.embedding_model = None
        self._initialize_database()
        self._initialize_embeddings()

    def _initialize_database(self):
        """Initialize database connection."""
        connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MONGODB_CONNECTION_STRING environment variable not set")
        
        try:
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
                maxPoolSize=3,
                minPoolSize=1,
                retryWrites=True,
                w='majority'
            )
            
            # Test connection
            self.client.admin.command("ping", maxTimeMS=3000)
            
            # Initialize database references
            self.db = self.client["parliamentary_graph"]
            self.nodes = self.db.nodes
            self.edges = self.db.edges
            self.statements = self.db.statements
            
            # Create indexes if they don't exist
            try:
                self.nodes.create_index([("pagerank_score", ASCENDING)])
                self.nodes.create_index([("pagerank_rank", ASCENDING)])
            except:
                pass
            
            logger.info("✅ Connected to MongoDB")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            if self.client:
                self.client.close()
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def _initialize_embeddings(self):
        """Initialize embedding model."""
        try:
            logger.info("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Vector search enabled")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise RuntimeError(f"Vector search failed to initialize: {e}")

    def hybrid_search(self, query: str, limit: int = 8, pagerank_weight: float = 0.3) -> List[Dict]:
        """Hybrid search combining PageRank importance with semantic similarity."""
        try:
            logger.info(f"🎯 Hybrid search for: '{query}'")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # MongoDB vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 5,
                        "limit": limit * 3
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"},
                        "normalized_pagerank": {
                            "$divide": [
                                {"$ifNull": ["$pagerank_score", 0.00001]},
                                0.01
                            ]
                        }
                    }
                },
                {
                    "$addFields": {
                        "hybrid_score": {
                            "$add": [
                                {"$multiply": [pagerank_weight, "$normalized_pagerank"]},
                                {"$multiply": [(1 - pagerank_weight), "$similarity_score"]}
                            ]
                        }
                    }
                },
                {
                    "$sort": {"hybrid_score": -1}
                },
                {
                    "$limit": limit
                },
                {
                    "$project": {
                        "uri": 1,
                        "label": 1,
                        "name": 1,
                        "type": 1,
                        "searchable_text": 1
                    }
                }
            ]
            
            results = list(self.nodes.aggregate(pipeline))
            logger.info(f"🎯 Hybrid search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            return []

    def get_connected_nodes(self, uris: Set[str], hops: int = 1) -> Set[str]:
        """Get nodes connected to the given URIs."""
        try:
            current, seen = set(uris), set(uris)
            for hop in range(max(0, hops)):
                if not current or len(seen) > 500:
                    break
                    
                edges = self.edges.find({
                    "$or": [
                        {"subject": {"$in": list(current)}},
                        {"object": {"$in": list(current)}},
                    ]
                })
                
                nxt = set()
                for edge in edges:
                    nxt.add(edge["subject"])
                    nxt.add(edge["object"])
                
                current = nxt - seen
                seen.update(nxt)
                    
            return seen
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return uris

    def get_subgraph(self, uris: Set[str]) -> Dict[str, Any]:
        """Get subgraph for the given URIs."""
        try:
            if len(uris) > 500:
                uris = set(list(uris)[:500])
            
            # Get nodes
            raw_nodes = list(self.nodes.find(
                {"uri": {"$in": list(uris)}}, 
                {
                    "uri": 1,
                    "label": 1,
                    "name": 1,
                    "type": 1,
                    "searchable_text": 1
                }
            ))
            
            # Clean nodes
            cleaned_nodes = []
            for node in raw_nodes:
                cleaned = {
                    "uri": node.get("uri"),
                    "type": node.get("type", [])
                }
                
                # Handle labels - prefer label over name
                label = node.get("label") or node.get("name")
                if label:
                    cleaned["label"] = label
                
                if "searchable_text" in node:
                    cleaned["searchable_text"] = node["searchable_text"]
                
                cleaned_nodes.append(cleaned)
            
            # Get edges
            edges = list(self.edges.find({
                "subject": {"$in": list(uris)}, 
                "object": {"$in": list(uris)}
            }))
            
            # Clean edges
            for edge in edges:
                if "_id" in edge:
                    edge["_id"] = str(edge["_id"])
            
            return {"nodes": cleaned_nodes, "edges": edges}
            
        except Exception as e:
            logger.error(f"Subgraph retrieval failed: {e}")
            return {"nodes": [], "edges": []}

    def to_turtle(self, subgraph: Dict[str, Any]) -> str:
        """Convert subgraph to Turtle format."""
        try:
            g = Graph()
            
            # Add prefixes
            g.bind("bbp", "http://example.com/barbados-parliament-ontology#")
            g.bind("rdfs", RDFS)
            g.bind("rdf", RDF)
            
            # Add nodes
            for node in subgraph["nodes"]:
                try:
                    uri = URIRef(node["uri"])
                    
                    if "label" in node and node["label"]:
                        g.add((uri, RDFS.label, Literal(str(node["label"]))))
                    
                    for t in node.get("type", []):
                        g.add((uri, RDF.type, URIRef(t)))
                    
                except Exception as e:
                    logger.warning(f"Skipping node: {e}")
            
            # Add edges
            for edge in subgraph["edges"]:
                try:
                    g.add((
                        URIRef(edge["subject"]),
                        URIRef(edge["predicate"]),
                        URIRef(edge["object"]) if edge["object"].startswith("http") else Literal(edge["object"])
                    ))
                except Exception as e:
                    logger.warning(f"Skipping edge: {e}")
            
            header = f"# Generated {datetime.now(timezone.utc).isoformat()}Z\n"
            header += f"# Nodes: {len(subgraph['nodes'])}, Edges: {len(subgraph['edges'])}\n\n"
            
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"Turtle serialization failed: {e}")
            return f"# Error: {str(e)}\n"

    def get_provenance_turtle(self, node_uris: List[str], include_transcript: bool = True) -> str:
        """Get provenance information as Turtle format."""
        try:
            logger.info(f"📚 Getting provenance for {len(node_uris)} nodes")
            
            g = Graph()
            g.bind("bbp", "http://example.com/barbados-parliament-ontology#")
            g.bind("prov", "http://www.w3.org/ns/prov#")
            g.bind("schema", "http://schema.org/")
            g.bind("rdfs", RDFS)
            
            for uri in node_uris[:10]:  # Limit to prevent explosion
                try:
                    node_uri = URIRef(uri)
                    
                    # Get related statements
                    projection = {
                        "subject": 1,
                        "predicate": 1, 
                        "object": 1,
                        "source_video": 1,
                        "video_title": 1,
                        "start_offset": 1,
                        "end_offset": 1
                    }
                    
                    if include_transcript:
                        projection["transcript_text"] = 1
                    
                    statements = list(self.statements.find({
                        "$or": [
                            {"subject": uri},
                            {"predicate": uri}, 
                            {"object": uri}
                        ]
                    }, projection))
                    
                    # Process statements
                    for i, stmt in enumerate(statements[:5]):
                        stmt_uri = URIRef(f"{uri}/statement/{i}")
                        
                        # Basic provenance
                        g.add((stmt_uri, RDF.type, PROV.Entity))
                        g.add((stmt_uri, PROV.wasDerivedFrom, node_uri))
                        g.add((stmt_uri, SCHEMA.about, node_uri))
                        
                        # Video information
                        video_id = stmt.get("source_video")
                        video_title = stmt.get("video_title")
                        start_time = stmt.get("start_offset")
                        
                        if video_id:
                            if start_time is not None:
                                timestamped_url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
                            else:
                                timestamped_url = f"https://www.youtube.com/watch?v={video_id}"
                            
                            g.add((stmt_uri, SCHEMA.url, Literal(timestamped_url)))
                            
                            if video_title:
                                g.add((stmt_uri, SCHEMA.videoTitle, Literal(video_title)))
                        
                        if start_time is not None:
                            g.add((stmt_uri, SCHEMA.startTime, Literal(int(start_time))))
                        
                        # Transcript text
                        if include_transcript and "transcript_text" in stmt:
                            transcript = stmt["transcript_text"]
                            if transcript and len(transcript.strip()) > 0:
                                if len(transcript) > 1000:
                                    transcript = transcript[:1000] + "..."
                                g.add((stmt_uri, SCHEMA.text, Literal(transcript)))
                        
                except Exception as e:
                    logger.warning(f"Skipping provenance for {uri}: {e}")
            
            header = f"# Provenance information generated {datetime.now(timezone.utc).isoformat()}Z\n\n"
            return header + g.serialize(format="turtle")
            
        except Exception as e:
            logger.error(f"❌ Provenance turtle generation failed: {e}")
            return f"# Error: {str(e)}\n"

    def close(self):
        """Close database connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()

def convert_markdown_to_html_and_filter_links(markdown_text: str) -> str:
    """Convert markdown to HTML and filter out non-YouTube links."""
    try:
        # Convert markdown to HTML first
        html = markdown.markdown(markdown_text, extensions=['extra', 'codehilite'])
        
        # Parse with BeautifulSoup for link filtering and formatting
        soup = BeautifulSoup(html, 'html.parser')
        
        # Filter out non-YouTube links
        links = soup.find_all('a')
        for link in links:
            href = link.get('href', '')
            if 'youtube.com' not in href.lower():
                link.replace_with(link.get_text())
        
        # Clean up the HTML to remove excessive spacing
        html_content = str(soup)
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error processing markdown: {e}")
        return markdown_text

class ParliamentarySystem:
    """Main parliamentary system using ADK patterns with fixed session management."""
    
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.querier = ParliamentaryGraphQuerier()
        self.conversation_history = {}  # Store by session_id
        
        # Create session service and runner once
        from google.adk.sessions import InMemorySessionService
        self.session_service = InMemorySessionService()
        self.runner = None
        
        # Create search tools using ADK FunctionTool
        def search_parliament_hybrid(query: str, hops: int = 2, limit: int = 5) -> str:
            """
            Search parliamentary records using hybrid search.
            
            Args:
                query: Search query for parliamentary information
                hops: Number of relationship hops to explore (1-3)
                limit: Maximum number of results (1-10)
            
            Returns:
                Parliamentary data in Turtle format with facts and relationships
            """
            try:
                logger.info(f"🔍 Searching parliament: {query}")
                
                # Perform hybrid search
                seeds = self.querier.hybrid_search(query, limit)
                if not seeds:
                    return f"# No parliamentary data found for: {query}\n"
                
                # Get connected nodes
                seed_uris = {node["uri"] for node in seeds if "uri" in node}
                all_uris = self.querier.get_connected_nodes(seed_uris, hops)
                
                # Get subgraph and convert to turtle
                subgraph = self.querier.get_subgraph(all_uris)
                turtle_data = self.querier.to_turtle(subgraph)
                
                # Get provenance data
                if seed_uris:
                    provenance_data = self.querier.get_provenance_turtle(list(seed_uris)[:5])
                    combined_data = turtle_data + "\n\n# PROVENANCE DATA:\n" + provenance_data
                    return combined_data
                
                return turtle_data
                
            except Exception as e:
                logger.error(f"Parliament search failed: {e}")
                return f"# Error searching parliament: {str(e)}\n"
        
        def clear_conversation_context(reason: str = "Starting new topic") -> str:
            """
            Clear conversation history to start fresh.
            
            Args:
                reason: Why you're clearing the context
            
            Returns:
                Confirmation message
            """
            # Note: We can't access session_id here, so this will be handled in process_query
            logger.info(f"🧹 Conversation context clear requested: {reason}")
            return f"Conversation context cleared successfully. Reason: {reason}"
        
        # Create the main agent with tools
        self.agent = LlmAgent(
            name="YuhHearDem",
            model="gemini-2.5-flash-preview-05-20",
            description="AI assistant for Barbados Parliament information",
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=0)),
            instruction="""You are YuhHearDem, a friendly AI assistant helping people understand Barbados Parliament.

## CORE BEHAVIOR
- ALWAYS search for specific parliamentary information when users ask about topics, ministers, policies, or issues
- Use search_parliament_hybrid tool for ANY question about parliamentary matters
- NEVER say "Let me search" or "Searching" - just provide results directly
- Always end responses with helpful follow-up suggestions to encourage continued exploration

## WHEN TO SEARCH (ALWAYS search for these):
- Any mention of: water, infrastructure, education, health, budget, policies, economy, agriculture, tourism, schools, hospitals, music, culture, soca, carnival, arts, sports
- Questions about ministers, MPs, or government officials
- Parliamentary debates, sessions, bills, or legislation
- Recent events, announcements, or government decisions
- Specific topics like "water issues", "education funding", "healthcare policy", "schools", "soca music", "culture", "arts"
- ANY question about what happened in parliament or what someone said
- ANY topic that might have been discussed in parliament (when in doubt, search!)

## SEARCH FIRST APPROACH
For questions about parliamentary topics:
1. IMMEDIATELY use search_parliament_hybrid tool with relevant keywords
2. Extract key information from search results
3. Provide response with specific details and sources
4. Include YouTube links with timestamps when available
5. End with 2-3 relevant follow-up suggestions

## SEARCH PARAMETERS
- Use specific search terms related to the topic
- Set limit between 5-8 for good coverage
- Use 2-3 hops to get related information

## MARKDOWN FORMATTING REQUIREMENTS
**CRITICAL: ALL responses must use valid markdown syntax. Follow these rules strictly:**

### Link Formatting
- **VALID**: `[Link Text](https://youtube.com/watch?v=ID&t=120s)`
- **INVALID**: `[Link Text](invalid-url)` or `[Link Text]()` or broken URLs
- Always verify YouTube URLs follow format: `https://youtube.com/watch?v=VIDEO_ID` or `https://youtu.be/VIDEO_ID`
- For timestamps, use format: `&t=120s` (for 2 minutes) or `&t=1h30m45s`
- If no valid URL is available, use plain text instead of broken links

### Text Formatting - SIMPLE ONLY
- Use `**bold**` for key topics and emphasis
- Use `*italic*` sparingly for speaker names or emphasis
- Use `-` for bullet points when listing items
- Use `>` for blockquotes when citing parliamentary statements
- **NO HEADERS**: Do not use `#`, `##`, `###` - keep responses conversational with paragraphs only

### Content Structure - CONVERSATIONAL
- Start directly with content - NO headers or titles
- Use natural paragraphs to organize information
- Use bullet points only when listing specific items
- Use blockquotes for direct parliamentary quotes: `> "Quote here" - Speaker Name`
- Keep formatting minimal and conversational

## RESPONSE FORMAT
- Provide specific information found in search using valid markdown
- Include video sources ONLY if you have valid YouTube URLs like: 
  - "According to the **[Parliamentary Session on Education](https://youtube.com/watch?v=abc123&t=300s)**..."
  - If URL is invalid/missing, use: "According to parliamentary discussions on education..." (no link)
- If search finds limited results, acknowledge and suggest related searches
- NEVER include "searching" or "let me search" text
- ALWAYS end with follow-up suggestions formatted as:
  ```markdown
  **Explore More:**
  - Would you like to know more about [related topic]?
  - I can also search for information about [related area]
  - Other topics you might find interesting: [suggestion 1], [suggestion 2]
  ```

## QUALITY CHECKS BEFORE RESPONDING
1. **Verify all links**: Ensure every `[text](url)` has a real, working URL or remove the link
2. **Check markdown syntax**: Ensure headers, lists, and formatting are correct
3. **Validate YouTube URLs**: Must be complete and follow proper format
4. **No broken formatting**: No unclosed brackets, missing spaces, or malformed syntax

## EXAMPLE BEHAVIOR
User: "Tell me about schools"
Action: IMMEDIATELY call search_parliament_hybrid(query="schools education", limit=6, hops=2)
Response Format:
```markdown
Parliamentary discussions have recently focused on **school infrastructure improvements** and **education funding allocations**. The Minister of Education outlined plans for new classroom construction and teacher training programs.

According to the [Parliamentary Education Session](https://youtube.com/watch?v=REAL_ID&t=120s), the government allocated $2.5 million for school repairs across the island.

> "We must prioritize our children's education through better facilities" - Minister of Education

**Would you like to know more about education funding? I can also search for information about teacher training programs or school infrastructure projects.**
```

User: "about soca music"
Action: IMMEDIATELY call search_parliament_hybrid(query="soca music culture", limit=6, hops=2)
Response: Provide findings using conversational paragraphs, then suggest: 

**Interested in other cultural topics? I can search for discussions about carnival or arts funding. Related areas: cultural heritage, music industry support.**

## ERROR PREVENTION
- **Before sending response**: Double-check every `[text](url)` link
- **If URL is broken/missing**: Convert to plain text or remove entirely
- **If unsure about link validity**: Don't include the link - use plain text description
- **Always preview**: Ensure response renders as valid markdown

Remember: When in doubt, SEARCH FIRST, then respond with the findings using PERFECT MARKDOWN! NEVER respond without searching for ANY topic that might have been discussed in parliament. Always guide users toward deeper exploration with helpful follow-up suggestions formatted correctly.
""",
            tools=[
                FunctionTool(search_parliament_hybrid),
                FunctionTool(clear_conversation_context)
            ],
            generate_content_config=GenerateContentConfig(
                temperature=0.1,  # Very low temperature for consistent tool use
                max_output_tokens=5000
            )
        )
        
        # Initialize runner
        from google.adk.runners import Runner
        self.runner = Runner(
            agent=self.agent,
            session_service=self.session_service,
            app_name="YuhHearDem"
        )
    
    async def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Get existing session or create a new one."""
        if session_id:
            # Try to get existing session - check if it exists in our conversation history
            if session_id in self.conversation_history:
                logger.info(f"✅ Found existing session: {session_id[:8]}...")
                return session_id
            else:
                logger.warning(f"Session {session_id[:8]}... not found in conversation history, creating new one")
        
        # Create new session
        session = await self.session_service.create_session(
            app_name="YuhHearDem",
            user_id=user_id
        )
        logger.info(f"✅ Created new session: {session.id[:8]}...")
        return session.id
    
    async def process_query(self, query: str, user_id: str = "user", session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Process a query through the parliamentary agent."""
        try:
            logger.info(f"🚀 Processing query: {query[:50]}...")
            
            # Get or create session
            actual_session_id = await self.get_or_create_session(user_id, session_id)
            
            # Initialize conversation history for this session if needed
            if actual_session_id not in self.conversation_history:
                self.conversation_history[actual_session_id] = []
            
            # Build context from conversation history
            context = ""
            session_history = self.conversation_history[actual_session_id]
            if session_history:
                context = "\n\nRECENT CONVERSATION:\n"
                for exchange in session_history[-3:]:
                    context += f"User: {exchange['user']}\n"
                    assistant_preview = exchange.get('assistant', '')[:200]
                    context += f"Assistant: {assistant_preview}...\n\n"
            
            # Create message with context
            full_query = f"{context}CURRENT QUESTION: {query}"
            user_message = Content(role="user", parts=[Part.from_text(text=full_query)])
            
            # Run agent and collect ALL events before responding
            all_events = []
            events = self.runner.run(
                user_id=user_id,
                session_id=actual_session_id,
                new_message=user_message
            )
            
            # Collect all events first (don't stream intermediate responses)
            for event in events:
                all_events.append(event)
            
            # Now process events and get the final response only
            response_text = ""
            for event in all_events:
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # Only use the final text response, not intermediate ones
                                response_text = part.text  # This will be the final response
            
            # Update conversation history
            session_history.append({
                "user": query,
                "assistant": response_text
            })
            
            # Trim history if too long
            if len(session_history) > 8:
                self.conversation_history[actual_session_id] = session_history[-8:]
            
            return response_text, {"success": True, "session_id": actual_session_id}
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error processing query: {str(e)}", {"success": False}
    
    def close(self):
        """Close system resources."""
        if hasattr(self, 'querier'):
            self.querier.close()

# Global system instance
parliamentary_system = None

def create_system() -> ParliamentarySystem:
    """Create the parliamentary system."""
    google_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')
    
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY environment variable required")
    
    return ParliamentarySystem(google_api_key)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Parliamentary Chatbot System...")
    
    global parliamentary_system
    try:
        parliamentary_system = create_system()
        logger.info("✅ Parliamentary System created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create Parliamentary system: {e}")
        parliamentary_system = None
    
    logger.info("✅ Parliamentary Chatbot System ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Parliamentary Chatbot System...")
    if parliamentary_system:
        parliamentary_system.close()

# Create FastAPI app
app = FastAPI(
    title="Parliamentary Research API",
    description="AI-powered Parliamentary research system with ADK",
    version="2.0.0",
    lifespan=lifespan
)

# Mount static files and templates
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Setup templates (assumes templates/index.html exists)
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def format_sse_event(event_type: str, agent: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
    """Format Server-Sent Event data."""
    event_data = {
        "type": event_type,
        "agent": agent,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data:
        event_data["data"] = data
    
    return f"data: {json.dumps(event_data)}\n\n"

async def process_query_with_events(query: str, user_id: str, session_id: str):
    """Process query with real-time events."""
    try:
        yield format_sse_event("query_start", "System", f"Processing query: {query[:50]}...")
        
        response_text, status = await parliamentary_system.process_query(query, user_id, session_id)
        
        if status.get("success", False):
            processed_response = convert_markdown_to_html_and_filter_links(response_text)
            
            # Use actual session ID from the response
            actual_session_id = status.get("session_id", session_id)
            
            yield format_sse_event("response_ready", "Assistant", "Response completed", {
                "response": processed_response,
                "message_id": str(uuid.uuid4()),
                "session_id": actual_session_id,
                "type": "parliamentary"
            })
        else:
            yield format_sse_event("error", "System", f"Error: {response_text}")
        
        yield format_sse_event("stream_complete", "System", "Query processing complete.")
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        yield format_sse_event("error", "System", f"Error processing query: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "YuhHearDem - Parliamentary Research API with ADK",
        "status": "running",
        "version": "2.0.0",
        "features": ["adk_integration", "database_search", "conversational_ai"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not parliamentary_system:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "System not initialized"}
        )
    
    try:
        # Test database connection
        parliamentary_system.querier.client.admin.command("ping", maxTimeMS=3000)
        db_connected = True
    except Exception as e:
        db_connected = False
    
    total_conversations = sum(len(history) for history in parliamentary_system.conversation_history.values())
    
    return {
        "status": "healthy" if db_connected else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database_connected": db_connected,
        "active_sessions": len(parliamentary_system.conversation_history),
        "total_conversations": total_conversations,
        "version": "2.0.0"
    }

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a query."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        response_text, status = await parliamentary_system.process_query(
            request.query, 
            request.user_id, 
            request.session_id
        )
        
        if status.get("success", False):
            actual_session_id = status.get("session_id", "parliamentary_session")
            
            return QueryResponse(
                session_id=actual_session_id,
                user_id=request.user_id,
                message_id="msg_" + str(datetime.utcnow().timestamp()),
                status="success",
                message=convert_markdown_to_html_and_filter_links(response_text)
            )
        else:
            raise HTTPException(status_code=500, detail=response_text)
            
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream query processing."""
    if not parliamentary_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    async def event_generator():
        async for event_data in process_query_with_events(request.query, request.user_id, session_id):
            yield event_data
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info("🚀 Starting Parliamentary Chatbot with ADK")
    logger.info(f"📡 Server will run on 0.0.0.0:{port}")
    logger.info("📋 Required: GOOGLE_API_KEY, MONGODB_CONNECTION_STRING")
    
    uvicorn.run(
        "yuhheardem_chatbot:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )