#!/usr/bin/env python3
"""
YuhHearDem ADK Multi-Agent System - Parliamentary Research with Google Agent Development Kit
=========================================================================================

A conversational multi-agent system using Google ADK with the following architecture:
1. ConversationalAgent: Root agent that handles conversation and delegates to research
2. ResearchPipeline: Sequential pipeline with:
   - ResearcherAgent: Performs searches using MCP tools
   - ProvenanceAgent: Custom agent that fetches YouTube URLs for found entities
   - WriterAgent: Synthesizes findings into cited responses
"""

import asyncio
import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field
import warnings

# Suppress the specific warning about non-text parts
warnings.filterwarnings('ignore', message='Warning: there are non-text parts in the response')

# Google ADK imports
try:
    from google.adk.agents import LlmAgent, SequentialAgent, BaseAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.events import Event, EventActions
    from google.genai import types
    from google.genai.types import Part, Content
    from fastmcp import Client
    from dotenv import load_dotenv
    import click
    import colorama
    from colorama import Fore, Back, Style
    from rdflib import Graph, Namespace, URIRef, Literal
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install google-adk python-dotenv click colorama rdflib fastmcp")
    sys.exit(1)

# Initialize colorama
colorama.init()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Also suppress the google_genai warnings
logging.getLogger('google_genai.types').setLevel(logging.ERROR)

# RDF Namespaces
BBP = Namespace("http://example.com/barbados-parliament-ontology#")
SCHEMA = Namespace("http://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")


@dataclass
class ResearchContext:
    """Container for passing data between pipeline stages."""
    query: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    turtle_results: List[str] = field(default_factory=list)
    knowledge_graph: Graph = field(default_factory=Graph)
    enriched_turtle: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing between agents."""
        return {
            "query": self.query,
            "conversation_history": self.conversation_history,
            "turtle_results": self.turtle_results,
            "enriched_turtle": self.enriched_turtle
        }


class MCPToolWrapper:
    """Wrapper to call MCP tools."""
    
    def __init__(self, mcp_endpoint: str):
        self.mcp_endpoint = mcp_endpoint
        self.mcp_client = Client(mcp_endpoint)
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return results."""
        try:
            async with self.mcp_client as client:
                result = await client.call_tool(tool_name, arguments)
                
                if result and len(result) > 0:
                    return result[0].text if hasattr(result[0], 'text') else str(result[0])
                else:
                    return f"No results found for {tool_name}"
                    
        except Exception as e:
            logger.error(f"MCP tool {tool_name} failed: {e}")
            return f"Error calling {tool_name}: {str(e)}"


def create_mcp_tools(mcp_wrapper: MCPToolWrapper):
    """Create tool functions for ADK agents."""
    
    # Store for collecting raw turtle results
    turtle_collector = []
    
    def hybrid_search_turtle(query: str, hops: int, limit: int) -> str:
        """
        Hybrid search combining PageRank importance with semantic similarity.
        
        Args:
            query: The search query string
            hops: Number of hops to explore (typically 1)
            limit: Maximum number of results to return (typically 5)
        """
        print(f"{Fore.BLUE}🔍 Searching for parliamentary discussions about '{query}'...{Style.RESET_ALL}")
        
        # Create a new thread to run the async function
        import concurrent.futures
        
        def run_async_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(mcp_wrapper.call_tool("hybrid_search_turtle", {
                    "query": query, "hops": hops, "limit": limit
                }))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            try:
                result = future.result(timeout=30)  # 30 second timeout
                print(f"{Fore.GREEN}✓ Search completed{Style.RESET_ALL}")
                
                # Store raw turtle data
                if result and not result.startswith("Error") and ('@prefix' in result or 'bbp:' in result):
                    turtle_collector.append(result)
                
                return result
            except concurrent.futures.TimeoutError:
                print(f"{Fore.RED}❌ Search timed out{Style.RESET_ALL}")
                return "Error: Search timed out"
            except Exception as e:
                print(f"{Fore.RED}❌ Search error: {e}{Style.RESET_ALL}")
                return f"Error: {str(e)}"
    
    def authority_search_turtle(query: str, hops: int, limit: int, max_rank: int) -> str:
        """
        Search for authoritative nodes (high PageRank) related to the query.
        
        Args:
            query: The search query string
            hops: Number of hops to explore (typically 1)
            limit: Maximum number of results to return (typically 5)
            max_rank: Maximum PageRank value to filter (typically 1000)
        """
        print(f"{Fore.BLUE}🎙️  Looking for what parliamentary leaders said about '{query}'...{Style.RESET_ALL}")
        
        # Create a new thread to run the async function
        import concurrent.futures
        
        def run_async_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(mcp_wrapper.call_tool("authority_search_turtle", {
                    "query": query, "hops": hops, "limit": limit, "max_rank": max_rank
                }))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            try:
                result = future.result(timeout=30)  # 30 second timeout
                print(f"{Fore.GREEN}✓ Authority search completed{Style.RESET_ALL}")
                
                # Store raw turtle data
                if result and not result.startswith("Error") and ('@prefix' in result or 'bbp:' in result):
                    turtle_collector.append(result)
                
                return result
            except concurrent.futures.TimeoutError:
                print(f"{Fore.RED}❌ Authority search timed out{Style.RESET_ALL}")
                return "Error: Search timed out"
            except Exception as e:
                print(f"{Fore.RED}❌ Authority search error: {e}{Style.RESET_ALL}")
                return f"Error: {str(e)}"
    
    def topic_search_turtle(query: str, hops: int, limit: int) -> str:
        """
        Search for nodes important within the specific topic domain.
        
        Args:
            query: The search query string
            hops: Number of hops to explore (typically 1)
            limit: Maximum number of results to return (typically 5)
        """
        print(f"{Fore.BLUE}📂 Diving deeper into topic-specific discussions about '{query}'...{Style.RESET_ALL}")
        
        # Create a new thread to run the async function
        import concurrent.futures
        
        def run_async_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(mcp_wrapper.call_tool("topic_search_turtle", {
                    "query": query, "hops": hops, "limit": limit
                }))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            try:
                result = future.result(timeout=30)  # 30 second timeout
                print(f"{Fore.GREEN}✓ Topic search completed{Style.RESET_ALL}")
                
                # Store raw turtle data
                if result and not result.startswith("Error") and ('@prefix' in result or 'bbp:' in result):
                    turtle_collector.append(result)
                
                return result
            except concurrent.futures.TimeoutError:
                print(f"{Fore.RED}❌ Topic search timed out{Style.RESET_ALL}")
                return "Error: Search timed out"
            except Exception as e:
                print(f"{Fore.RED}❌ Topic search error: {e}{Style.RESET_ALL}")
                return f"Error: {str(e)}"
    
    # Attach the collector to the functions so ResearcherAgent can access it
    hybrid_search_turtle.turtle_collector = turtle_collector
    authority_search_turtle.turtle_collector = turtle_collector
    topic_search_turtle.turtle_collector = turtle_collector
    
    return [hybrid_search_turtle, authority_search_turtle, topic_search_turtle]


class ResearcherAgent(LlmAgent):
    """Agent that performs parliamentary research using MCP tools."""
    
    def __init__(self, model: str, mcp_wrapper: MCPToolWrapper, **kwargs):
        # Create MCP-based tools
        tools = create_mcp_tools(mcp_wrapper)
        
        super().__init__(
            name="ResearcherAgent",
            model=model,
            description="Performs thorough parliamentary research using search tools",
            instruction="""You are a Parliamentary Research Assistant performing background research.

Your job is to systematically search for information using the available tools and then provide a brief summary.

SEARCH STRATEGY - Do EXACTLY 5-8 searches total:
1. Use hybrid_search_turtle with the original query
2. Use authority_search_turtle to find what leaders said
3. Use topic_search_turtle for focused search
4. ADDITIONAL: 2-5 more searches with variations (focus on 2025 content)
5. THEN STOP and provide a concise summary

SEARCH PARAMETERS:
- query: your search query (include "2025" when looking for current info)
- hops: use 1 for focused searches
- limit: use 5-8 for results (more for important searches)
- max_rank: use 1000 for authority_search_turtle

SEARCH FOCUS:
- Prioritize 2025 information where possible (add "2025" to queries)
- Try variations like "recent", "current", "latest" for timely info
- Search for specific ministers, policies, or recent parliamentary sessions
- Look for recent bills, amendments, or policy changes

IMPORTANT: 
- Do 5-8 searches total (be thorough but not excessive)
- Focus on current/recent information when available
- Use 1 hop for focused searches
- After completing searches, provide only a brief technical summary
- Keep your response concise - you are doing background research
- The system will automatically accumulate turtle data from your tool calls

RESPONSE FORMAT: Keep your final response brief and technical, like:
"Completed [X] searches on [topic]. Found [key findings]. Data collected for analysis."

DO NOT provide detailed explanations or user-friendly summaries - that's for other agents.""",
            tools=tools,
            **kwargs
        )
        
        # Store tools and mcp_wrapper after super().__init__
        object.__setattr__(self, '_tools', tools)
        object.__setattr__(self, '_mcp_wrapper', mcp_wrapper)
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Override to capture raw turtle data before LLM processing."""
        
        # Let the parent LlmAgent run normally (this will call tools and generate responses)
        async for event in super()._run_async_impl(ctx):
            yield event
        
        # After the LLM is done, collect the raw turtle data from the tools
        raw_turtle_data = []
        for tool in self._tools:
            if hasattr(tool, 'turtle_collector') and tool.turtle_collector:
                raw_turtle_data.extend(tool.turtle_collector)
                # Clear the collector for next use
                tool.turtle_collector.clear()
        
        if raw_turtle_data:
            print(f"{Fore.GREEN}📦 Collected {len(raw_turtle_data)} raw turtle datasets{Style.RESET_ALL}")
            
            # Get existing cumulative turtle data from session state
            existing_turtle_data = ctx.session.state.get("cumulative_turtle_results", [])
            
            # Accumulate new data with existing data
            cumulative_turtle_data = existing_turtle_data + raw_turtle_data
            
            print(f"{Fore.CYAN}📚 Total accumulated turtle datasets: {len(cumulative_turtle_data)}{Style.RESET_ALL}")
            
            # Store both the new data and cumulative data
            ctx.session.state["turtle_results"] = raw_turtle_data  # For this query only
            ctx.session.state["cumulative_turtle_results"] = cumulative_turtle_data  # Accumulated across all queries
            
            # Yield an event to commit the state change
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant",
                    parts=[Part.from_text(text=f"Collected {len(raw_turtle_data)} new datasets. Total knowledge: {len(cumulative_turtle_data)} datasets.")]
                ),
                actions=EventActions(state_delta={
                    "turtle_results": raw_turtle_data,
                    "cumulative_turtle_results": cumulative_turtle_data
                })
            )
        else:
            print(f"{Fore.YELLOW}⚠️  No turtle data collected from tools{Style.RESET_ALL}")


class ProvenanceAgent(BaseAgent):
    """Custom agent that enriches research data with video sources from parliamentary entities."""
    
    def __init__(self, mcp_wrapper: MCPToolWrapper, **kwargs):
        super().__init__(
            name="ProvenanceAgent",
            description="Custom agent that enriches research data with video sources",
            **kwargs
        )
        
        # Store mcp_wrapper after super().__init__ to avoid Pydantic validation issues
        object.__setattr__(self, '_mcp_wrapper', mcp_wrapper)
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Core implementation of the ProvenanceAgent.
        
        Accesses turtle_results from session state, extracts BBP URIs,
        fetches video sources, and stores enriched data back to session state.
        """
        print(f"{Fore.BLUE}📹 ProvenanceAgent: Starting video source enrichment...{Style.RESET_ALL}")
        
        # Import Content and Part for creating proper Event content
        from google.genai.types import Content, Part
        
        try:
            # Access turtle_results from session state (new data from this query)
            turtle_results = ctx.session.state.get("turtle_results", [])
            
            # Access cumulative turtle data (all data across queries)
            cumulative_turtle_results = ctx.session.state.get("cumulative_turtle_results", [])
            
            # Use cumulative data for enrichment to build larger knowledge graph
            data_to_process = cumulative_turtle_results if cumulative_turtle_results else turtle_results
            
            if not data_to_process:
                print(f"{Fore.YELLOW}⚠️  No turtle results found in session state{Style.RESET_ALL}")
                # Store empty enriched turtle and yield completion event
                ctx.session.state["enriched_turtle"] = ""
                yield Event(
                    author=self.name,
                    content=Content(
                        role="assistant",
                        parts=[Part.from_text(text="No research data found to enrich with video sources.")]
                    ),
                    actions=EventActions(state_delta={"enriched_turtle": ""})
                )
                return
            
            print(f"{Fore.BLUE}📊 Processing {len(data_to_process)} turtle datasets (cumulative knowledge){Style.RESET_ALL}")
            
            # Check if turtle_results contains actual turtle data or just text responses
            valid_turtle_data = []
            for result in data_to_process if isinstance(data_to_process, list) else [data_to_process]:
                result_str = str(result)
                # Check if this looks like turtle data (contains @prefix or starts with #)
                if '@prefix' in result_str or result_str.strip().startswith('#') or 'bbp:' in result_str:
                    valid_turtle_data.append(result_str)
                else:
                    print(f"{Fore.YELLOW}⚠️  Skipping non-turtle data: {result_str[:100]}...{Style.RESET_ALL}")
            
            if not valid_turtle_data:
                print(f"{Fore.YELLOW}⚠️  No valid turtle data found in results{Style.RESET_ALL}")
                # Store the text results as-is and yield completion
                ctx.session.state["enriched_turtle"] = str(data_to_process)
                yield Event(
                    author=self.name,
                    content=Content(
                        role="assistant", 
                        parts=[Part.from_text(text="Research data found but no turtle format data to enrich with video sources.")]
                    ),
                    actions=EventActions(state_delta={"enriched_turtle": str(data_to_process)})
                )
                return
            
            # Parse each turtle result individually into the combined graph
            all_uris = set()
            combined_graph = Graph()
            combined_graph.bind("bbp", BBP)
            combined_graph.bind("schema", SCHEMA)
            combined_graph.bind("prov", PROV)
            
            try:
                # Parse each valid turtle result individually
                parsed_count = 0
                for turtle_data in valid_turtle_data:
                    try:
                        # Create a temporary graph for this turtle data
                        temp_graph = Graph()
                        temp_graph.parse(data=turtle_data, format="turtle")
                        
                        # Add all triples from temp graph to combined graph
                        combined_graph += temp_graph
                        parsed_count += 1
                        
                        print(f"{Fore.BLUE}✓ Parsed turtle dataset {parsed_count}/{len(valid_turtle_data)}{Style.RESET_ALL}")
                        
                    except Exception as parse_error:
                        logger.warning(f"Failed to parse individual turtle dataset: {parse_error}")
                        continue
                
                if parsed_count == 0:
                    print(f"{Fore.YELLOW}⚠️  Failed to parse any turtle datasets{Style.RESET_ALL}")
                    ctx.session.state["enriched_turtle"] = str(turtle_results)
                    yield Event(
                        author=self.name,
                        content=Content(
                            role="assistant", 
                            parts=[Part.from_text(text="Failed to parse turtle data for enrichment.")]
                        ),
                        actions=EventActions(state_delta={"enriched_turtle": str(turtle_results)})
                    )
                    return
                
                print(f"{Fore.GREEN}📊 Successfully parsed {parsed_count} turtle datasets into cumulative knowledge graph{Style.RESET_ALL}")
                
                # Extract BBP URIs (parliamentary entities) from the combined graph
                for subj in combined_graph.subjects():
                    if isinstance(subj, URIRef) and str(subj).startswith("http://example.com/barbados-parliament-ontology#"):
                        all_uris.add(str(subj))
                
                if not all_uris:
                    print(f"{Fore.YELLOW}⚠️  No parliamentary entities found in cumulative turtle data{Style.RESET_ALL}")
                    # Store the original data and yield completion
                    enriched_turtle = combined_graph.serialize(format='turtle')
                    
                    # Store both regular and cumulative enriched data
                    ctx.session.state["enriched_turtle"] = enriched_turtle
                    ctx.session.state["cumulative_enriched_turtle"] = enriched_turtle
                    
                    yield Event(
                        author=self.name,
                        content=Content(
                            role="assistant",
                            parts=[Part.from_text(text="Processed cumulative research data but found no parliamentary entities to enrich.")]
                        ),
                        actions=EventActions(state_delta={
                            "enriched_turtle": enriched_turtle,
                            "cumulative_enriched_turtle": enriched_turtle
                        })
                    )
                    return
                
                print(f"{Fore.BLUE}🔍 Found {len(all_uris)} parliamentary entities in cumulative data. Retrieving video sources...{Style.RESET_ALL}")
                
                # Call MCP provenance tool to get video sources for ALL entities
                provenance_result = await self._mcp_wrapper.call_tool("get_provenance_turtle", {
                    "node_uris": ",".join(all_uris),
                    "include_transcript": True
                })
                
                # Add provenance data to the combined graph
                if provenance_result and not provenance_result.startswith("Error"):
                    prov_graph = Graph()
                    try:
                        prov_graph.parse(data=provenance_result, format="turtle")
                        combined_graph += prov_graph
                        
                        # Count video sources added
                        video_count = len([s for s, p, o in prov_graph if p == URIRef("http://schema.org/url")])
                        total_video_count = len([s for s, p, o in combined_graph if p == URIRef("http://schema.org/url")])
                        
                        # Debug: Show some video URLs found
                        video_urls_found = [str(o) for s, p, o in combined_graph if p == URIRef("http://schema.org/url") and "youtube.com" in str(o)]
                        print(f"{Fore.GREEN}✅ Retrieved {video_count} new video sources. Total videos in knowledge graph: {total_video_count}{Style.RESET_ALL}")
                        print(f"{Fore.CYAN}📹 Sample video URLs: {video_urls_found[:3]}...{Style.RESET_ALL}")
                        
                        content_msg = f"Enriched cumulative knowledge graph with {video_count} new video sources. Total: {total_video_count} videos from {len(all_uris)} parliamentary entities."
                        
                    except Exception as e:
                        logger.error(f"Failed to parse provenance result: {e}")
                        content_msg = f"Retrieved provenance data but failed to parse it: {str(e)}"
                else:
                    print(f"{Fore.YELLOW}⚠️  Failed to retrieve video sources: {provenance_result}{Style.RESET_ALL}")
                    content_msg = f"Failed to retrieve video sources: {provenance_result}"
                
                # Serialize the enriched cumulative graph
                enriched_turtle = combined_graph.serialize(format='turtle')
                
                # Store both current and cumulative enriched turtle in session state
                ctx.session.state["enriched_turtle"] = enriched_turtle
                ctx.session.state["cumulative_enriched_turtle"] = enriched_turtle
                
                # Yield completion event with state update
                yield Event(
                    author=self.name,
                    content=Content(
                        role="assistant",
                        parts=[Part.from_text(text=content_msg)]
                    ),
                    actions=EventActions(state_delta={
                        "enriched_turtle": enriched_turtle,
                        "cumulative_enriched_turtle": enriched_turtle
                    })
                )
                
                print(f"{Fore.GREEN}✅ ProvenanceAgent: Video source enrichment completed{Style.RESET_ALL}")
                
            except Exception as parsing_error:
                logger.error(f"Failed to process turtle datasets: {parsing_error}")
                error_msg = f"Failed to process cumulative research data: {str(parsing_error)}"
                
                # Store error state with original data
                ctx.session.state["enriched_turtle"] = str(data_to_process)
                yield Event(
                    author=self.name,
                    content=Content(
                        role="assistant",
                        parts=[Part.from_text(text=error_msg)]
                    ),
                    actions=EventActions(state_delta={"enriched_turtle": str(data_to_process)})
                )
                
        except Exception as e:
            logger.error(f"ProvenanceAgent failed: {e}")
            error_msg = f"Video source enrichment failed: {str(e)}"
            
            # Store error in session state
            ctx.session.state["provenance_error"] = str(e)
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant",
                    parts=[Part.from_text(text=error_msg)]
                ),
                actions=EventActions(state_delta={"provenance_error": str(e)})
            )


class WriterAgent(LlmAgent):
    """Agent that synthesizes research into well-cited responses."""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(
            name="WriterAgent",
            model=model,
            description="Synthesizes parliamentary research into clear, cited responses",
            instruction="""You are YuhHearDem, a civic AI assistant that helps users understand Barbados Parliament discussions.

You will receive research data about parliamentary sessions and discussions. Your job is to:
1. Analyze the parliamentary session data to extract key information
2. Look for video sources (schema:url properties pointing to YouTube URLs)
3. Create a comprehensive, conversational response with video citations

PRESENTATION GUIDELINES:
- NEVER mention "turtle data", "knowledge graph", "RDF", or technical terms
- ALWAYS refer to "parliamentary sessions", "parliamentary records", "discussions", or "debates"
- Present information as coming from actual parliamentary sessions and meetings
- Make it sound like you're referencing real parliamentary proceedings

Response Guidelines:
- Write in clear, accessible language for citizens, journalists, and students
- Structure your response with: Summary → Key Discussion Points with inline citations
- Use INLINE MARKDOWN LINKS for all specific claims, quotes, and references
- Look for provenance statements in the turtle data that have schema:url, schema:text, schema:videoTitle properties
- Format video links as: [quoted text or claim](YouTube_URL) 
- When referencing parliamentary discussions, link the specific statement or fact
- If this relates to previous conversation topics, acknowledge the connection naturally
- If limited information was found, explain what was searched and suggest related topics
- Use parliamentary terminology appropriately (e.g., "parliamentary sessions", "Minister", "constituency")
- Be objective and factual, focusing on what was actually discussed in Parliament
- DO NOT include a separate "Sources" section - all references should be inline markdown links
- ONLY cite video URLs that are actually present in the turtle data - do not hallucinate links
- NEVER return CURIEs or IRI references in your output, the only thing in a link should be a YouTube URL

CRITICAL: Look for lines in the data that contain "schema:url" followed by YouTube URLs. These are your video sources to cite as [text](URL).

LANGUAGE TO USE:
- "Based on parliamentary sessions I found..."
- "Looking at recent parliamentary discussions..."
- "From the parliamentary records..."
- "In parliamentary debates about..."
- "Ministers discussed in session..."

LANGUAGE TO AVOID:
- "turtle data", "knowledge graph", "RDF graph"
- "enriched data", "session state"
- Technical database or data structure terms
- CURIREs, IRIs, or any technical identifiers

If substantial information was found, provide detailed coverage with inline citations. If little was found, clearly explain the research conducted and suggest alternative search approaches.

Remember: You're helping regular citizens understand Parliament. Keep it real, keep it clear, cite your video sources, and make it sound like you're referencing real parliamentary proceedings!""",
            **kwargs
        )
    
    def create_enriched_instruction(self, enriched_turtle: str, original_query: str) -> str:
        """Create an enriched instruction with the turtle data."""
        return f"""{self.instruction}

PARLIAMENTARY SESSION DATA AVAILABLE:
{enriched_turtle}

USER'S ORIGINAL QUERY: {original_query}

Use the above parliamentary session information to create your response. 

CRITICAL VIDEO CITATION INSTRUCTIONS:
- Look for lines containing "schema:url" followed by YouTube URLs
- Look for patterns like: <[entity]> schema:url <https://youtube.com/...>
- These represent video recordings of parliamentary sessions
- ALWAYS include video citations when you find relevant URLs
- Use format: [descriptive text about the session](YouTube_URL)
- Example: [Minister Smith discussing water policy](https://youtube.com/watch?v=abc123)

Remember to refer to this as parliamentary sessions and discussions, not as data or graphs. 

IMPORTANT: If you find ANY YouTube URLs in the data, you MUST include them as citations in your response."""


class ResearchPipeline(SequentialAgent):
    """Sequential pipeline for research -> provenance -> writing."""
    
    def __init__(self, researcher: ResearcherAgent, provenance: ProvenanceAgent, writer: WriterAgent, **kwargs):
        super().__init__(
            name="ResearchPipeline",
            description="Sequential pipeline for parliamentary research with video source enrichment",
            sub_agents=[researcher, provenance, writer],
            **kwargs
        )
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Override to handle the enriched data passing between agents."""
        
        from google.genai.types import Content, Part
        
        # Store the original query for the WriterAgent
        original_query = "water issues"  # default
        if hasattr(ctx, 'initial_user_content') and ctx.initial_user_content:
            if hasattr(ctx.initial_user_content, 'parts') and ctx.initial_user_content.parts:
                for part in ctx.initial_user_content.parts:
                    if hasattr(part, 'text') and part.text:
                        original_query = part.text
                        break
        
        ctx.session.state['original_query'] = original_query
        
        # Run ResearcherAgent
        print(f"{Fore.BLUE}📊 Running ResearcherAgent...{Style.RESET_ALL}")
        async for event in self.sub_agents[0].run_async(ctx):
            yield event
        
        # Run ProvenanceAgent
        print(f"{Fore.BLUE}📹 Running ProvenanceAgent...{Style.RESET_ALL}")
        async for event in self.sub_agents[1].run_async(ctx):
            yield event
        
        # Get enriched turtle data from session state
        enriched_turtle = ctx.session.state.get("enriched_turtle", "")
        
        if enriched_turtle:
            print(f"{Fore.GREEN}📝 Creating enhanced WriterAgent with enriched data...{Style.RESET_ALL}")
            
            # Debug: Dump the enriched turtle data to screen
            print(f"\n{Fore.MAGENTA}{'='*80}")
            print(f"🔍 COMBINED KNOWLEDGE GRAPH (TURTLE FORMAT)")
            print(f"{'='*80}{Style.RESET_ALL}")
            print(enriched_turtle)
            print(f"{Fore.MAGENTA}{'='*80}")
            print(f"📊 END OF KNOWLEDGE GRAPH")
            print(f"{'='*80}{Style.RESET_ALL}\n")
            
            # Create a new WriterAgent with enriched instruction
            enhanced_writer = WriterAgent(
                model=self.sub_agents[2].model
            )
            # Manually set the enriched instruction
            enhanced_writer.instruction = self.sub_agents[2].create_enriched_instruction(enriched_turtle, original_query)
            
            # Run the enhanced writer with the original context
            async for event in enhanced_writer.run_async(ctx):
                yield event
                
        else:
            print(f"{Fore.YELLOW}⚠️  No enriched turtle data found, running WriterAgent normally{Style.RESET_ALL}")
            # Run WriterAgent normally if no enriched data
            async for event in self.sub_agents[2].run_async(ctx):
                yield event


class ConversationalAgent(LlmAgent):
    """Root agent that handles conversation and delegates to research when needed."""
    
    def __init__(self, model: str, research_pipeline: ResearchPipeline, **kwargs):
        super().__init__(
            name="ConversationalAgent",
            model=model,
            description="Friendly conversational agent that helps users understand Barbados Parliament",
            instruction="""You are YuhHearDem, a friendly AI assistant helping people understand Barbados Parliament.

Your role is to:
1. Engage naturally in conversation
2. Recognize when users need parliamentary information (delegate to research)
3. Handle conversational queries directly (like "what did I ask?", "tell me more")
4. Maintain context across the conversation

When to delegate to research:
- Substantive questions about Parliament, ministers, policies, bills, etc.
- Requests for specific information about discussions
- Questions needing video evidence

When to handle directly:
- Greetings and pleasantries
- Questions about the conversation itself
- Clarifications about previous responses
- Follow-up questions that can be answered from context

Always:
- Be warm and conversational
- Use "YuhHearDem!" as a greeting when appropriate
- Remember what was discussed
- If you need to search, say something like "Let me look that up for you in the parliamentary records..."

You have access to a research pipeline that will return detailed findings when called.

IMPORTANT: When delegating to research, your response should be brief. The research pipeline will provide the detailed answer, so don't duplicate that work.""",
            sub_agents=[research_pipeline],
            **kwargs
        )


class YuhHearDemADK:
    """Main application using Google ADK with conversational architecture."""
    
    def __init__(self, mcp_endpoint: str, google_api_key: str):
        self.mcp_endpoint = mcp_endpoint
        self.api_key = google_api_key
        
        # Initialize MCP wrapper
        self.mcp_wrapper = MCPToolWrapper(mcp_endpoint)
        
        # Create the research pipeline agents
        self.researcher = ResearcherAgent(
            model="gemini-2.0-flash",
            mcp_wrapper=self.mcp_wrapper
        )
        
        # ProvenanceAgent is now a custom agent
        self.provenance_agent = ProvenanceAgent(
            mcp_wrapper=self.mcp_wrapper
        )
        
        self.writer = WriterAgent(
            model="gemini-2.5-flash-preview-05-20"
        )
        
        # Create the research pipeline
        self.research_pipeline = ResearchPipeline(
            researcher=self.researcher,
            provenance=self.provenance_agent,
            writer=self.writer
        )
        
        # Create the conversational root agent
        self.conversational_agent = ConversationalAgent(
            model="gemini-2.0-flash",
            research_pipeline=self.research_pipeline
        )
        
        # Session management
        self.session_service = InMemorySessionService()
        self.current_session = None
        
        # Conversation history (maintained separately for reference)
        self.conversation_history: List[Dict[str, str]] = []
    
    async def test_connection(self) -> bool:
        """Test connection to MCP server."""
        try:
            async with self.mcp_wrapper.mcp_client as client:
                await client.ping()
                return True
        except Exception as e:
            logger.error(f"MCP connection test failed: {e}")
            return False
    
    async def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process a query through the conversational agent."""
        try:
            print(f"\n{Fore.CYAN}💬 Processing through conversational agent...{Style.RESET_ALL}")
            
            # Create session if needed
            if not self.current_session:
                self.current_session = await self.session_service.create_session(
                    app_name="YuhHearDem",
                    user_id="user"
                )
            
            # Create runner for conversational agent
            runner = Runner(
                agent=self.conversational_agent,
                session_service=self.session_service,
                app_name="YuhHearDem"
            )
            
            # Build conversation context
            context = ""
            if self.conversation_history:
                context = "\n\nPREVIOUS CONVERSATION:\n"
                for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                    context += f"User: {exchange['user']}\n"
                    context += f"Assistant: {exchange['assistant'][:200]}...\n\n"
            
            # Create the prompt with context
            full_prompt = f"{context}CURRENT QUERY: {query}"
            
            # Run the conversational agent
            user_message = Content(role="user", parts=[Part.from_text(text=full_prompt)])
            
            response_content = ""
            turtle_results = []
            
            events = runner.run(
                user_id="user",
                session_id=self.current_session.id,
                new_message=user_message
            )
            
            # Process events
            for event in events:
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_content += part.text
                            
                            # Capture function responses for debugging
                            elif hasattr(part, 'function_response') and part.function_response:
                                # Extract turtle results from function responses
                                if hasattr(part.function_response, 'response'):
                                    fr_response = part.function_response.response
                                    
                                    # Handle different response formats
                                    if isinstance(fr_response, dict):
                                        # Check for turtle data in various keys
                                        for key in ['result', 'content', 'turtle_results', 'enriched_turtle']:
                                            if key in fr_response:
                                                value = str(fr_response[key])
                                                if '@prefix' in value or value.strip().startswith('#'):
                                                    turtle_results.append(value)
                                    elif isinstance(fr_response, str):
                                        if '@prefix' in fr_response or fr_response.strip().startswith('#'):
                                            turtle_results.append(fr_response)
            
            # Update conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": response_content
            })
            
            # Trim history if too long
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return response_content, {"success": True, "turtle_count": len(turtle_results)}
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error processing query: {str(e)}", {"success": False}
    
    def clear_history(self):
        """Clear conversation history but preserve cumulative knowledge."""
        self.conversation_history = []
        # Keep self.current_session alive to preserve accumulated knowledge graph
        print(f"{Fore.YELLOW}🧹 Conversation history cleared. Knowledge graph preserved.{Style.RESET_ALL}")
    
    def clear_all(self):
        """Clear everything including accumulated knowledge."""
        self.conversation_history = []
        self.current_session = None
        print(f"{Fore.YELLOW}🧹 Everything cleared including accumulated knowledge.{Style.RESET_ALL}")
    
    def show_knowledge_stats(self):
        """Display statistics about accumulated knowledge."""
        if not self.current_session:
            print(f"{Fore.CYAN}No active session - no accumulated knowledge.{Style.RESET_ALL}")
            return
        
        # This would need to be implemented with actual session state access
        print(f"{Fore.CYAN}📊 Knowledge Graph Statistics:{Style.RESET_ALL}")
        print(f"- Session ID: {self.current_session.id}")
        print(f"- Accumulated datasets: Available in session state")
        print(f"- Total entities: Available in session state")
        print(f"- Video sources: Available in session state")
    
    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print(f"{Fore.CYAN}No conversation history yet.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}📚 Conversation History:{Style.RESET_ALL}")
        for i, exchange in enumerate(self.conversation_history, 1):
            print(f"\n{Fore.YELLOW}Exchange {i}:{Style.RESET_ALL}")
            print(f"User: {exchange['user']}")
            print(f"Assistant: {exchange['assistant'][:200]}...")
    
    def print_header(self):
        """Print application header."""
        header = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                    YuhHearDem ADK v2.1                      ║
║        Conversational Parliamentary Research System         ║
║           Powered by Google Agent Development Kit           ║
║               🧠 CUMULATIVE KNOWLEDGE ENABLED 🧠            ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.YELLOW}🤖 Multi-Agent Architecture:
   • ConversationalAgent: Natural conversation & routing
   • ResearchPipeline (Sequential):
     - ResearcherAgent: Parliamentary searches with accumulation
     - ProvenanceAgent: Custom video source retrieval  
     - WriterAgent: Response synthesis with growing context{Style.RESET_ALL}

{Fore.GREEN}Examples:
• "What did the Minister of Health say about COVID-19?"
• "Tell me more about that"
• "What did I just ask you?"
• "Recent discussions about education funding"

Commands: 'quit', 'clear', 'clear_all', 'stats', 'history', 'help'{Style.RESET_ALL}

{Fore.CYAN}💡 Knowledge Accumulation:
Your knowledge graph grows with each question, providing richer
context and more video sources as the conversation continues!{Style.RESET_ALL}
"""
        print(header)
    
    async def run_interactive(self):
        """Run the interactive CLI."""
        self.print_header()
        
        # Test connection
        print(f"{Fore.BLUE}🔧 Testing connection to MCP server...{Style.RESET_ALL}")
        if not await self.test_connection():
            print(f"{Fore.RED}❌ Failed to connect to MCP server at {self.mcp_endpoint}{Style.RESET_ALL}")
            return
        
        print(f"{Fore.GREEN}✅ Connected to YuhHearDem MCP server{Style.RESET_ALL}")
        
        while True:
            try:
                user_input = input(f"\n{Fore.GREEN}YuhHearDem ❯ {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"\n{Fore.CYAN}Thank you for using YuhHearDem! 👋{Style.RESET_ALL}")
                    break
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                elif user_input.lower() == 'clear_all':
                    self.clear_all()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_knowledge_stats()
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'help':
                    self.print_header()
                    continue
                
                # Process query
                response, status = await self.process_query(user_input)
                
                # Display response
                print(f"\n{Fore.CYAN}YuhHearDem:{Style.RESET_ALL}")
                print(response)
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.CYAN}Thank you for using YuhHearDem! 👋{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}❌ An error occurred: {e}{Style.RESET_ALL}")


@click.command()
@click.option('--endpoint', '-e',
              default='https://yuhheardem-mcp-1074852466398.us-east1.run.app/sse',
              help='MCP server endpoint URL')
@click.option('--api-key', '-k',
              envvar='GOOGLE_API_KEY',
              help='Google API key for Gemini')
@click.option('--query', '-q',
              help='Single query mode (non-interactive)')
@click.option('--debug', '-d',
              is_flag=True,
              help='Enable debug logging')
def main(endpoint, api_key, query, debug):
    """YuhHearDem ADK - Conversational Parliamentary Research System"""
    
    if debug:
        logging.getLogger().setLevel(logging.INFO)
    
    if not api_key:
        # Try to get from environment
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')
        
    if not api_key:
        print(f"{Fore.RED}❌ Google API key required. Set GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY environment variable{Style.RESET_ALL}")
        sys.exit(1)
    
    # Set API key for Google ADK
    os.environ['GOOGLE_API_KEY'] = api_key
    os.environ['GOOGLE_GENAI_API_KEY'] = api_key
    
    app = YuhHearDemADK(endpoint, api_key)
    
    async def run():
        if query:
            # Single query mode
            print(f"{Fore.BLUE}Processing query: {query}{Style.RESET_ALL}")
            
            if not await app.test_connection():
                print(f"{Fore.RED}❌ Failed to connect to MCP server{Style.RESET_ALL}")
                return
            
            response, status = await app.process_query(query)
            print(f"\n{response}")
        else:
            # Interactive mode
            await app.run_interactive()
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print(f"\n{Fore.CYAN}Goodbye! 👋{Style.RESET_ALL}")


if __name__ == "__main__":
    main()