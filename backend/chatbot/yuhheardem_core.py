#!/usr/bin/env python3
"""
YuhHearDem Core ADK System - Parliamentary Research Components (KNOWLEDGE GRAPH REFACTORED)
============================================================

Rearchitected to build cumulative knowledge graph where:
1. Each search tool parses its turtle output and adds to cumulative graph
2. Tools return serialized cumulative graph (not raw results)
3. ResearcherAgent gets current knowledge state with each query
4. Enables intelligent, context-aware searching
"""

import asyncio
import json
import os
import sys
import logging
import re
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
    from google.adk.planners import BuiltInPlanner
    from google.genai import types
    from google.genai.types import Part, Content
    from fastmcp import Client
    from rdflib import Graph, Namespace, URIRef, Literal
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install google-adk rdflib fastmcp")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)

# Also suppress the google_genai warnings
logging.getLogger('google_genai.types').setLevel(logging.ERROR)

# RDF Namespaces - Define these here so they can be imported by webmain if needed
BBP = Namespace("http://example.com/barbados-parliament-ontology#")
SCHEMA = Namespace("http://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")


@dataclass
class ResearchContext:
    """Container for passing data between pipeline stages."""
    query: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    cumulative_knowledge_graph: Graph = field(default_factory=Graph) # The main cumulative graph
    enriched_turtle: str = "" # Enriched data for current research step
    cumulative_enriched_turtle: str = "" # Enriched data across session


class CumulativeKnowledgeGraph:
    """Manages the cumulative knowledge graph across all searches."""
    
    def __init__(self):
        self.graph = Graph()
        self.graph.bind("bbp", BBP)
        self.graph.bind("schema", SCHEMA)
        self.graph.bind("prov", PROV)
        self._search_count = 0
    
    def add_turtle_data(self, turtle_data: str) -> bool:
        """Add turtle data to the cumulative graph. Returns True if successful."""
        if not turtle_data or not turtle_data.strip():
            return False
            
        try:
            temp_graph = Graph()
            temp_graph.parse(data=turtle_data, format="turtle")
            
            # Add to cumulative graph
            self.graph += temp_graph
            self._search_count += 1
            
            logger.info(f"CumulativeKnowledgeGraph: Added search result #{self._search_count}. "
                       f"Graph now has {len(self.graph)} triples.")
            return True
            
        except Exception as e:
            logger.error(f"CumulativeKnowledgeGraph: Failed to parse turtle data: {e}")
            return False
    
    def get_serialized_turtle(self) -> str:
        """Get the current cumulative graph as turtle."""
        try:
            return self.graph.serialize(format='turtle')
        except Exception as e:
            logger.error(f"CumulativeKnowledgeGraph: Failed to serialize: {e}")
            return ""
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the current knowledge graph."""
        return {
            "total_triples": len(self.graph),
            "search_count": self._search_count,
            "entities": len(set(self.graph.subjects())),
            "properties": len(set(self.graph.predicates())),
        }
    
    def get_knowledge_summary(self) -> str:
        """Get a human-readable summary of current knowledge for the LLM."""
        stats = self.get_summary_stats()
        
        if stats["total_triples"] == 0:
            return "CURRENT KNOWLEDGE: Empty - no parliamentary data gathered yet."
        
        # Extract some key entities for context
        entities = []
        for subj in self.graph.subjects():
            if isinstance(subj, URIRef) and str(subj).startswith(str(BBP)):
                entity_name = str(subj).split('#')[-1].replace('_', ' ')
                entities.append(entity_name)
                if len(entities) >= 5:  # Limit for brevity
                    break
        
        entity_list = ", ".join(entities[:3])
        if len(entities) > 3:
            entity_list += f" and {len(entities) - 3} others"
        
        return (f"CURRENT KNOWLEDGE: {stats['total_triples']} facts from {stats['search_count']} searches. "
               f"Key entities: {entity_list}.")


class MCPToolWrapper:
    """Wrapper to call MCP tools."""

    def __init__(self, mcp_endpoint: str):
        self.mcp_endpoint = mcp_endpoint

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return results."""
        try:
            client = Client(self.mcp_endpoint)
            async with client as async_client:
                result = await async_client.call_tool(tool_name, arguments)

                if result and len(result) > 0:
                    return "\n".join(r.text if hasattr(r, 'text') else str(r) for r in result)
                else:
                    return f"No results found for {tool_name} with args {arguments}"

        except Exception as e:
            logger.error(f"MCP tool {tool_name} failed with args {arguments}: {e}")
            return f"Error calling {tool_name}: {str(e)}"


def sync_mcp_tool_caller_with_kg(mcp_wrapper: MCPToolWrapper, tool_name: str, knowledge_graph: CumulativeKnowledgeGraph):
    """Creates a synchronous function that calls an async MCP tool and updates the knowledge graph."""
    
    def sync_tool_func_with_kg(query: str, **kwargs) -> str:
        """Synchronous wrapper that updates knowledge graph and returns summary."""
        logger.info(f"🔧 Starting {tool_name} with query: '{query}' and kwargs: {kwargs}")
        
        try:
            try:
                loop = asyncio.get_running_loop()
                logger.info(f"🔄 Event loop detected, using thread-based execution for {tool_name}")
                
                import concurrent.futures
                import threading
                
                def run_async_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        args = {"query": query}
                        args.update(kwargs)
                        logger.info(f"🔍 Calling MCP tool {tool_name} with args: {args}")
                        result = new_loop.run_until_complete(mcp_wrapper.call_tool(tool_name, args))
                        logger.info(f"📥 MCP tool {tool_name} returned: {len(str(result)) if result else 0} characters")
                        return result
                    except Exception as e:
                        logger.error(f"❌ Error in thread for {tool_name}: {e}")
                        return f"Error in thread: {str(e)}"
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_thread)
                    raw_result = future.result(timeout=30)
                    
            except RuntimeError:
                logger.info(f"🔄 No event loop running, creating new one for {tool_name}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    args = {"query": query}
                    args.update(kwargs)
                    logger.info(f"🔍 Calling MCP tool {tool_name} with args: {args}")
                    raw_result = loop.run_until_complete(mcp_wrapper.call_tool(tool_name, args))
                    logger.info(f"📥 MCP tool {tool_name} returned: {len(str(raw_result)) if raw_result else 0} characters")
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
            
            # Process the raw result
            if not raw_result or raw_result.startswith("Error"):
                logger.warning(f"⚠️ {tool_name} returned error or empty result: {raw_result}")
                return f"Search '{query}' completed but found no new parliamentary data."
            
            # Check if result looks like turtle data and add to knowledge graph
            result_str = str(raw_result)
            is_turtle = ('@prefix' in result_str or 'bbp:' in result_str or result_str.strip().startswith('#'))
            
            if is_turtle:
                success = knowledge_graph.add_turtle_data(result_str)
                if success:
                    # Return the current cumulative knowledge graph as turtle
                    cumulative_turtle = knowledge_graph.get_serialized_turtle()
                    stats = knowledge_graph.get_summary_stats()
                    
                    logger.info(f"✅ {tool_name} added to knowledge graph. Total: {stats['total_triples']} triples")
                    
                    # Return a message + the cumulative turtle data
                    response = f"Search '{query}' found parliamentary data. Knowledge graph updated.\n\nCURRENT_CUMULATIVE_KNOWLEDGE_GRAPH:\n{cumulative_turtle}"
                    return response
                else:
                    logger.error(f"❌ Failed to add {tool_name} result to knowledge graph")
                    return f"Search '{query}' returned data but failed to parse into knowledge graph."
            else:
                logger.warning(f"⚠️ {tool_name} result doesn't look like turtle data: {result_str[:100]}...")
                return f"Search '{query}' completed but returned non-turtle data."
            
        except concurrent.futures.TimeoutError:
            error_msg = f"Timeout calling {tool_name} after 30 seconds"
            logger.error(f"⏰ {error_msg}")
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Exception in {tool_name}: {str(e)}"
            logger.error(f"💥 {error_msg}")
            import traceback
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            return f"Error: {error_msg}"

    sync_tool_func_with_kg.__name__ = tool_name
    return sync_tool_func_with_kg


def create_mcp_tools_with_kg(mcp_wrapper: MCPToolWrapper, knowledge_graph: CumulativeKnowledgeGraph):
    """Create tool functions for ADK agents that update the cumulative knowledge graph."""
    
    logger.info("🛠️ Creating MCP tools with cumulative knowledge graph integration")

    # Create synchronous wrappers for each tool
    hybrid_search_tool = sync_mcp_tool_caller_with_kg(mcp_wrapper, "hybrid_search_turtle", knowledge_graph)
    authority_search_tool = sync_mcp_tool_caller_with_kg(mcp_wrapper, "authority_search_turtle", knowledge_graph)
    topic_search_tool = sync_mcp_tool_caller_with_kg(mcp_wrapper, "topic_search_turtle", knowledge_graph)

    # Define tool metadata (args structure) - LLM needs this
    hybrid_search_tool.__name__ = "hybrid_search_turtle"
    hybrid_search_tool.__doc__ = """
        Hybrid search combining PageRank importance with semantic similarity.
        Updates the cumulative knowledge graph and returns current state.

        Args:
            query: The search query string
            hops: Number of hops to explore (typically 2 for broader results)
            limit: Maximum number of results to return (typically 5-8)
        """

    authority_search_tool.__name__ = "authority_search_turtle"
    authority_search_tool.__doc__ = """
        Search for authoritative nodes (high PageRank) related to the query.
        Updates the cumulative knowledge graph and returns current state.

        Args:
            query: The search query string
            hops: Number of hops to explore (typically 2 for broader results)
            limit: Maximum number of results to return (typically 5-8)
            max_rank: Maximum PageRank value to filter (typically 1000)
        """

    topic_search_tool.__name__ = "topic_search_turtle"
    topic_search_tool.__doc__ = """
        Search for nodes important within the specific topic domain.
        Updates the cumulative knowledge graph and returns current state.

        Args:
            query: The search query string
            hops: Number of hops to explore (typically 2 for broader results)
            limit: Maximum number of results to return (typically 5-8)
        """

    logger.info("✅ MCP tools created with knowledge graph integration")
    return [hybrid_search_tool, authority_search_tool, topic_search_tool]


class ResearcherAgent(LlmAgent):
    """Agent that performs parliamentary research using MCP tools with cumulative knowledge graph."""

    def __init__(self, model: str, mcp_wrapper: MCPToolWrapper, knowledge_graph: CumulativeKnowledgeGraph, **kwargs):
        # Create MCP-based tools wrapped for synchronous use by LlmAgent
        tools = create_mcp_tools_with_kg(mcp_wrapper, knowledge_graph)

        super().__init__(
            name="ResearcherAgent",
            model=model,
            description="Performs thorough parliamentary research using search tools with cumulative knowledge awareness",
            instruction="""You are a Parliamentary Research Assistant with ACCESS TO CUMULATIVE KNOWLEDGE.

You can see all parliamentary data gathered so far in this conversation. Use this to make SMART search decisions:

KNOWLEDGE-AWARE RESEARCH STRATEGY:
1. ANALYZE CURRENT KNOWLEDGE: Look at what entities, topics, and facts you already have
2. IDENTIFY GAPS: What's missing for a complete answer to the user's question?
3. AVOID REDUNDANCY: Don't search for information you already have
4. BUILD CONNECTIONS: Look for related entities and expand the knowledge network
5. STRATEGIC SEARCHING: Use different search types based on what you need:
   - hybrid_search_turtle: For balanced importance + relevance
   - authority_search_turtle: For what leaders/ministers said
   - topic_search_turtle: For domain-specific deep dives

SEARCH PROCESS:
1. First, review your current knowledge graph (if any)
2. Plan 3-5 strategic searches that complement existing knowledge
3. Start broad, then get specific based on what you find
4. Look for connections between new and existing entities
5. Stop when you have comprehensive coverage of the topic

TEMPORAL FOCUS: Prioritize 2025 information when available, but connect to historical context.

IMPORTANT: Each search tool will return the UPDATED cumulative knowledge graph. Pay attention to what's been added and adjust your next searches accordingly.

After completing strategic searches, respond with exactly: "Research complete."

This signals completion and prevents infinite loops.
""",
            tools=tools,
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=1500,
                temperature=0.3
            ),
            **kwargs
        )

        # Store references for internal use
        object.__setattr__(self, '_tools', tools)
        object.__setattr__(self, '_mcp_wrapper', mcp_wrapper)
        object.__setattr__(self, '_knowledge_graph', knowledge_graph)

    def _build_context_with_knowledge(self, original_query: str, conversation_history: List[Dict[str, str]]) -> str:
        """Build enhanced context that includes current knowledge state."""
        
        # Get knowledge summary
        knowledge_summary = self._knowledge_graph.get_knowledge_summary()
        
        # Build conversation context
        context = f"USER QUERY: {original_query}\n\n"
        context += f"{knowledge_summary}\n\n"
        
        if conversation_history:
            context += "CONVERSATION HISTORY:\n"
            # Include last few exchanges for context
            for exchange in conversation_history[-3:]:
                context += f"User: {exchange['user']}\n"
                assistant_preview = exchange.get('assistant', '')[:200]
                context += f"Assistant: {assistant_preview}...\n\n"
        
        # Add current cumulative knowledge if it exists
        current_turtle = self._knowledge_graph.get_serialized_turtle()
        if current_turtle.strip():
            context += "CURRENT CUMULATIVE KNOWLEDGE GRAPH:\n"
            context += current_turtle + "\n\n"
        
        context += "Based on the above knowledge and context, perform strategic searches to complete your understanding."
        
        return context

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Override to provide knowledge-aware context to the LLM."""
        
        # Extract original query and conversation history from context
        original_query = ctx.session.state.get('original_query', 'parliamentary matters')
        conversation_history = ctx.session.state.get('conversation_history', [])
        
        # Build enhanced context with knowledge awareness
        enhanced_context = self._build_context_with_knowledge(original_query, conversation_history)
        
        # Replace the user message with our enhanced context
        if hasattr(ctx, 'initial_user_content') and ctx.initial_user_content:
            # Create new content with enhanced context
            enhanced_content = Content(
                role="user",
                parts=[Part.from_text(text=enhanced_context)]
            )
            ctx.initial_user_content = enhanced_content
        
        # Run the parent LlmAgent with enhanced context
        async for event in super()._run_async_impl(ctx):
            yield event

        # After completion, store the final knowledge graph state
        final_turtle = self._knowledge_graph.get_serialized_turtle()
        
        # PRESERVE cumulative knowledge: Don't overwrite, just ensure it's available
        if final_turtle:
            ctx.session.state["cumulative_turtle_results"] = [final_turtle]
            ctx.session.state["persistent_knowledge_turtle"] = final_turtle
        
        stats = self._knowledge_graph.get_summary_stats()
        yield Event(
            author=self.name,
            content=Content(
                role="assistant",
                parts=[Part.from_text(text=f"Research completed. Knowledge graph: {stats['total_triples']} triples from {stats['search_count']} searches.")]
            ),
            actions=EventActions(state_delta={
                "cumulative_turtle_results": [final_turtle] if final_turtle else [],
                "persistent_knowledge_turtle": final_turtle if final_turtle else "",
                "knowledge_graph_stats": stats
            })
        )


class ProvenanceAgent(BaseAgent):
    """Custom agent that enriches research data with video sources from parliamentary entities."""

    def __init__(self, mcp_wrapper: MCPToolWrapper, **kwargs):
        super().__init__(
            name="ProvenanceAgent",
            description="Custom agent that enriches research data with video sources",
            **kwargs
        )
        object.__setattr__(self, '_mcp_wrapper', mcp_wrapper)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Core implementation of the ProvenanceAgent."""
        logger.info("ProvenanceAgent: Starting video source enrichment")

        try:
            # Access cumulative turtle data from session state
            # Try persistent knowledge first, then fall back to cumulative results
            persistent_turtle = ctx.session.state.get("persistent_knowledge_turtle", "")
            if persistent_turtle:
                cumulative_turtle_results = [persistent_turtle]
            else:
                cumulative_turtle_results = ctx.session.state.get("cumulative_turtle_results", [])

            if not cumulative_turtle_results:
                logger.warning("ProvenanceAgent: No cumulative turtle results found in session state")
                ctx.session.state["enriched_turtle"] = ""
                ctx.session.state["cumulative_enriched_turtle"] = ""

                yield Event(
                    author=self.name,
                    content=Content(
                        role="assistant",
                        parts=[Part.from_text(text="No research data found to enrich with video sources.")]
                    ),
                    actions=EventActions(state_delta={"enriched_turtle": "", "cumulative_enriched_turtle": ""})
                )
                return

            logger.info(f"ProvenanceAgent: Processing {len(cumulative_turtle_results)} turtle datasets")

            # Parse cumulative data into a combined graph
            combined_graph = Graph()
            combined_graph.bind("bbp", BBP)
            combined_graph.bind("schema", SCHEMA)
            combined_graph.bind("prov", PROV)

            parsed_count = 0
            for turtle_data in cumulative_turtle_results:
                try:
                    combined_graph.parse(data=turtle_data, format="turtle")
                    parsed_count += 1
                except Exception as parse_error:
                    logger.warning(f"ProvenanceAgent: Failed to parse turtle dataset: {parse_error}")
                    continue

            if parsed_count == 0:
                logger.warning("ProvenanceAgent: Failed to parse any turtle datasets")
                fallback_data = "\n".join(cumulative_turtle_results)
                ctx.session.state["cumulative_enriched_turtle"] = fallback_data

                yield Event(
                    author=self.name,
                    content=Content(
                        role="assistant",
                        parts=[Part.from_text(text="Failed to parse turtle data for enrichment.")]
                    ),
                    actions=EventActions(state_delta={"cumulative_enriched_turtle": fallback_data})
                )
                return

            logger.info(f"ProvenanceAgent: Successfully parsed {parsed_count} turtle datasets into graph.")

            # Extract BBP URIs (parliamentary entities) from the combined graph
            all_uris = set()
            for subj in combined_graph.subjects():
                if isinstance(subj, URIRef) and str(subj).startswith(str(BBP)):
                    all_uris.add(str(subj))

            if not all_uris:
                logger.warning("ProvenanceAgent: No parliamentary entities found in turtle data")
                enriched_turtle = combined_graph.serialize(format='turtle')
                ctx.session.state["cumulative_enriched_turtle"] = enriched_turtle

                yield Event(
                    author=self.name,
                    content=Content(
                        role="assistant",
                        parts=[Part.from_text(text="Processed research data but found no parliamentary entities to enrich.")]
                    ),
                    actions=EventActions(state_delta={"cumulative_enriched_turtle": enriched_turtle})
                )
                return

            logger.info(f"ProvenanceAgent: Found {len(all_uris)} parliamentary entities. Retrieving video sources...")

            # Call MCP provenance tool directly
            try:
                provenance_result_str = await self._mcp_wrapper.call_tool(
                    "get_provenance_turtle", 
                    {
                        "node_uris": ",".join(all_uris),
                        "include_transcript": True
                    }
                )
            except Exception as e:
                logger.error(f"ProvenanceAgent: Failed to call provenance tool: {e}")
                provenance_result_str = f"Error calling provenance tool: {str(e)}"

            # Add provenance data to the combined graph
            if provenance_result_str and not provenance_result_str.startswith("Error"):
                try:
                    prov_graph = Graph()
                    prov_graph.parse(data=provenance_result_str, format="turtle")
                    combined_graph += prov_graph

                    # Count video sources added
                    video_count = len(list(prov_graph.subjects(predicate=URIRef(SCHEMA["url"]))))
                    total_video_count = len(list(combined_graph.subjects(predicate=URIRef(SCHEMA["url"]))))

                    logger.info(f"ProvenanceAgent: Retrieved {video_count} new video sources. Total videos: {total_video_count}")
                    content_msg = f"Enriched knowledge graph with {video_count} new video sources. Total: {total_video_count} videos."

                except Exception as e:
                    logger.error(f"ProvenanceAgent: Failed to parse provenance result: {e}")
                    content_msg = f"Retrieved provenance data but failed to parse it: {str(e)}"
            else:
                logger.warning(f"ProvenanceAgent: Failed to retrieve video sources: {provenance_result_str}")
                content_msg = f"Failed to retrieve video sources: {provenance_result_str}"

            # Serialize the enriched graph
            enriched_turtle = combined_graph.serialize(format='turtle')
            ctx.session.state["cumulative_enriched_turtle"] = enriched_turtle

            yield Event(
                author=self.name,
                content=Content(
                    role="assistant",
                    parts=[Part.from_text(text=content_msg)]
                ),
                actions=EventActions(state_delta={"cumulative_enriched_turtle": enriched_turtle})
            )

            logger.info("ProvenanceAgent: Video source enrichment completed")

        except Exception as e:
            logger.error(f"ProvenanceAgent failed: {e}")
            
            # Fallback: provide raw data without provenance
            try:
                logger.info("ProvenanceAgent: Attempting fallback without provenance")
                cumulative_turtle_results = ctx.session.state.get("cumulative_turtle_results", [])
                fallback_data = "\n".join(cumulative_turtle_results) if cumulative_turtle_results else ""
                ctx.session.state["cumulative_enriched_turtle"] = fallback_data
                
                error_msg = f"Video source enrichment failed: {str(e)}. Using raw data as fallback."
                
            except Exception as fallback_error:
                logger.error(f"ProvenanceAgent: Fallback also failed: {fallback_error}")
                error_msg = f"Video source enrichment and fallback both failed: {str(e)}"
                ctx.session.state["cumulative_enriched_turtle"] = ""

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

    def __init__(self, model: str, turtle_data: str = "", user_query: str = "", **kwargs):
        # Build the instruction with the turtle data directly in the prompt
        instruction = f"""You are YuhHearDem, a civic AI assistant that helps users understand Barbados Parliament discussions.

Based on the parliamentary data below, provide a comprehensive answer to the user's question.

RESPONSE GUIDELINES:
- Write in clear, accessible language for citizens, journalists, and students
- Structure your response with: Summary → Key Discussion Points with inline citations
- Use INLINE MARKDOWN LINKS for all specific claims, quotes, and references from the parliamentary data.
- The ONLY links must be YouTube URLs derived from the turtle data (`schema:url`), do not hallucinate links.
- Look for provenance statements in the turtle data that have `schema:url`, `schema:text`, `schema:videoTitle`, and `schema:startTime` properties.
- **Specific Link Formatting:** When linking to a statement or claim found in the turtle data, include the timestamp and the video source for context. Format the reference *inline* after the relevant text like this: `([Timestamp] from "[Video Source Hint]")`.
    - The `[Timestamp]` part *itself* should be the markdown link.
    - The link URL should be the `schema:url` from the data with the `schema:startTime` added as a `#t=` fragment (e.g., `https://www.youtube.com/watch?v=videoID&t=startTimeInSeconds`).
    - Convert the `schema:startTime` (in seconds) to a clear MM:SS or H:MM:SS format for the displayed timestamp text.
    - For the "[Video Source Hint]", use a brief, recognizable part of the `schema:videoTitle`, such as the date and "House Session".

- **Example Link Format:** `...the second reading of the Queen Elizabeth Hospital Amendment Bill 2024 ([0:01:21](https://www.youtube.com/watch?v=pyKuPiXNDDo&t=81s) from "Mar 5, 2024 House Session").`
- When referencing parliamentary discussions, ensure the inline link correctly points to the specific statement's timestamp in the video.
- If this relates to previous conversation topics, acknowledge the connection naturally.
- If limited information was found, explain what was searched and suggest related topics.
- Use parliamentary terminology appropriately (e.g., "parliamentary sessions", "Minister", "constituency").
- Be objective and factual, focusing on what was actually discussed in Parliament.
- DO NOT include a separate "Sources" section - all references should be inline markdown links.
- ONLY cite video URLs that are actually present in the turtle data with `schema:url` properties - do not hallucinate links.
- Never include any other types of links or sources, only YouTube URLs from the turtle data.

CRITICAL: Only use video links and the corresponding `startTime` and `videoTitle` that appear in the turtle data. If no relevant video sources are found in the turtle data for a specific claim, state that no video sources were available for that point and explain what information was found instead (e.g., "According to the data provided, [claim], however, the source video was not available.").

If substantial information was found, provide detailed coverage with inline citations formatted as specified. If little was found, clearly explain the research conducted and suggest alternative search approaches.

Remember: You're helping regular citizens understand Parliament. Keep it real, keep it clear, cite your video sources with timestamps and source hints, and make it sound like you're referencing actual parliamentary proceedings!
USER QUERY: {user_query}

PARLIAMENTARY DATA TO ANALYZE:
```turtle
{turtle_data}
```

"""

        super().__init__(
            name="WriterAgent",
            model=model,
            description="Synthesizes parliamentary research into clear, cited responses",
            instruction=instruction,
            **kwargs
        )


class ResearchPipeline(SequentialAgent):
    """Sequential pipeline for research -> provenance -> writing."""

    def __init__(self, researcher: ResearcherAgent, provenance: ProvenanceAgent, writer_model: str, **kwargs):
        super().__init__(
            name="ResearchPipeline",
            description="Sequential pipeline for parliamentary research with video source enrichment",
            sub_agents=[researcher, provenance],
            **kwargs
        )
        object.__setattr__(self, '_writer_model', writer_model)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Override to handle the data flow between sequential agents."""

        # Store the original query for the WriterAgent
        original_query = ctx.session.state.get('original_query', 'parliamentary matters')
        logger.info(f"ResearchPipeline: Processing query: {original_query[:50]}...")

        # 1. Run ResearcherAgent
        logger.info("ResearchPipeline: Running ResearcherAgent...")
        async for event in self.sub_agents[0].run_async(ctx):
            yield event

        # 2. Run ProvenanceAgent
        logger.info("ResearchPipeline: Running ProvenanceAgent...")
        async for event in self.sub_agents[1].run_async(ctx):
            yield event

        # 3. Create WriterAgent with enriched turtle data
        cumulative_enriched_turtle = ctx.session.state.get("cumulative_enriched_turtle", "")
        
        if not cumulative_enriched_turtle:
            # Fallback to raw turtle data
            raw_turtle_results = ctx.session.state.get("cumulative_turtle_results", [])
            if raw_turtle_results:
                logger.info("ResearchPipeline: Using raw turtle data as fallback for WriterAgent")
                cumulative_enriched_turtle = "\n\n# === COMBINED PARLIAMENTARY DATA ===\n\n".join(raw_turtle_results)
            else:
                logger.warning("ResearchPipeline: No turtle data found for WriterAgent")
                cumulative_enriched_turtle = "# No parliamentary data found"

        logger.info(f"ResearchPipeline: Creating WriterAgent with {len(cumulative_enriched_turtle)} characters of turtle data")
        
        # 4. Create and run WriterAgent
        writer_agent = WriterAgent(
            model=self._writer_model,
            turtle_data=cumulative_enriched_turtle,
            user_query=original_query,
            generate_content_config=types.GenerateContentConfig(temperature=0.7),
            planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=1024))
        )
        
        logger.info("ResearchPipeline: Running dynamically created WriterAgent...")
        async for event in writer_agent.run_async(ctx):
            yield event

        logger.info("ResearchPipeline: Pipeline execution completed.")


class ConversationalAgent(LlmAgent):
    """Root agent that handles conversation and delegates to research when needed."""

    def __init__(self, model: str, research_pipeline: ResearchPipeline, yuhheardem_system, **kwargs):
        
        # Define the knowledge graph reset tool as a method
        def clear_knowledge_graph(reason: str = "Topic change detected") -> str:
            """
            Clear the cumulative knowledge graph to start fresh on a new topic.
            
            Args:
                reason: Brief explanation of why you're clearing the knowledge graph
                
            Returns:
                Confirmation message about the clearing operation
            """
            try:
                # Get stats before clearing
                old_stats = yuhheardem_system.knowledge_graph.get_summary_stats()
                
                # Reset the knowledge graph
                yuhheardem_system.knowledge_graph = CumulativeKnowledgeGraph()
                
                # Clear session state related to knowledge graph
                if yuhheardem_system.current_session:
                    yuhheardem_system.current_session.state.pop("cumulative_turtle_results", None)
                    yuhheardem_system.current_session.state.pop("persistent_knowledge_turtle", None)
                    yuhheardem_system.current_session.state.pop("cumulative_enriched_turtle", None)
                    yuhheardem_system.current_session.state.pop("knowledge_graph_stats", None)
                
                logger.info(f"🧹 Knowledge graph cleared by ConversationalAgent. Reason: {reason}")
                logger.info(f"📊 Previous stats: {old_stats['total_triples']} triples from {old_stats['search_count']} searches")
                
                return (f"Knowledge graph cleared successfully. "
                       f"Removed {old_stats['total_triples']} facts from {old_stats['search_count']} searches. "
                       f"Reason: {reason}")
                
            except Exception as e:
                error_msg = f"Failed to clear knowledge graph: {str(e)}"
                logger.error(f"❌ {error_msg}")
                return error_msg
        
        # Set function metadata for the LLM
        clear_knowledge_graph.__name__ = "clear_knowledge_graph"
        clear_knowledge_graph.__doc__ = """
        Clear the cumulative knowledge graph to start fresh on a new topic.
        
        Use this tool when:
        - The user changes to a completely different topic (e.g., from water issues to education)
        - The user explicitly asks to "start over" or "new question"
        - Previous parliamentary knowledge would be confusing for the new topic
        - The conversation shifts from specific detailed topics to general questions
        
        Args:
            reason: Brief explanation of why you're clearing (e.g., "User switched from healthcare to agriculture")
        
        Returns:
            Confirmation that the knowledge graph has been cleared
        """

        super().__init__(
            name="ConversationalAgent",
            model=model,
            description="Friendly conversational agent that helps users understand Barbados Parliament",
            instruction="""You are YuhHearDem, a friendly AI assistant helping people understand Barbados Parliament.

Your role is to:
1. Engage naturally in conversation.
2. Recognize when users need substantive parliamentary information (delegate to the ResearchPipeline).
3. Handle simple conversational queries directly (like "hello", "thank you", "what did I ask?").
4. Maintain context across the conversation based on the provided history.
5. **MANAGE KNOWLEDGE CONTEXT**: Clear the knowledge graph when topics change significantly.

KNOWLEDGE MANAGEMENT:
You have access to a `clear_knowledge_graph` tool. Use it when:
- User switches to a completely different parliamentary topic (water → education → healthcare)
- User says things like "new question", "different topic", "let's talk about something else"
- Previous accumulated knowledge would confuse the new research
- Moving from detailed specific topics to broad general questions

Examples of when to clear:
- User was asking about water infrastructure, now asks about education policy
- User was discussing specific bills, now asks about general parliamentary procedures
- User explicitly indicates a topic change

Examples of when NOT to clear:
- Follow-up questions on the same topic ("tell me more about that bill")
- Related subtopics (water quality → water infrastructure → water management)
- Clarifying questions ("what did you mean by that?")

When to delegate to the ResearchPipeline:
- Substantive questions about Parliament, ministers, policies, bills, water issues, etc.
- Requests for specific information about discussions or events.
- Questions needing details from parliamentary proceedings or requiring factual lookup.

When to handle directly:
- Greetings, farewells, and other social cues.
- Simple questions about your identity or purpose.
- Meta-questions about the conversation itself (e.g., "What did I just say?").
- Requests that do not require looking up information in the parliamentary database.

Always:
- Be warm and conversational.
- Use "YuhHearDem!" as a greeting when appropriate.
- Acknowledge the user's query and previous conversation briefly.
- Never refer to the knowledge graph itself, just refer to "the information we have" or "the parliamentary data".

IMPORTANT: For parliamentary questions, respond with a friendly acknowledgment and then call transfer_to_agent. 
Examples:
- "YuhHearDem! I'll look that up for you in the parliamentary records."
- "Let me search through the parliamentary discussions about that topic."
- "I'll find what Parliament has said about this issue."

If you decide to clear the knowledge graph, do it BEFORE transferring to research so the ResearcherAgent starts with a clean slate.

The research will be conducted separately and the results will be provided as a follow-up response.
""",
            tools=[clear_knowledge_graph],
            sub_agents=[research_pipeline],
            **kwargs
        )

class YuhHearDemADK:
    """Main application using Google ADK with conversational architecture and cumulative knowledge graph."""

    def __init__(self, mcp_endpoint: str, google_api_key: str):
        self.mcp_endpoint = mcp_endpoint
        self.api_key = google_api_key

        # Initialize MCP wrapper
        self.mcp_wrapper = MCPToolWrapper(mcp_endpoint)

        # Initialize cumulative knowledge graph
        self.knowledge_graph = CumulativeKnowledgeGraph()

        # Initialize Google GenAI with API key
        if self.api_key:
            try:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    logger.info("✅ Google Generative AI configured successfully")
                except ImportError:
                    logger.warning("⚠️ google.generativeai not installed, but continuing with ADK")
            except Exception as e:
                logger.error(f"❌ Failed to configure Google Generative AI: {e}")
                self.api_key = None

        # Define models
        researcher_model = os.getenv("RESEARCHER_MODEL", "gemini-2.0-flash")
        writer_model = os.getenv("WRITER_MODEL", "gemini-2.5-flash-preview-05-20")
        conversational_model = os.getenv("CONVERSATIONAL_MODEL", "gemini-2.0-flash")

        # Create the research pipeline agents with shared knowledge graph
        self.researcher = ResearcherAgent(
            model=researcher_model,
            mcp_wrapper=self.mcp_wrapper,
            knowledge_graph=self.knowledge_graph  # Shared knowledge graph
        )

        self.provenance_agent = ProvenanceAgent(
            mcp_wrapper=self.mcp_wrapper
        )

        # Create the research pipeline
        self.research_pipeline = ResearchPipeline(
            researcher=self.researcher,
            provenance=self.provenance_agent,
            writer_model=writer_model
        )

        # Create the conversational root agent (pass self reference for knowledge management)
        self.conversational_agent = ConversationalAgent(
            model=conversational_model,
            research_pipeline=self.research_pipeline,
            yuhheardem_system=self  # Pass reference to self for tool creation
        )

        # Session management
        self.session_service = InMemorySessionService()
        self.current_session = None
        self.conversation_history: List[Dict[str, str]] = []

    async def test_connection(self) -> bool:
        """Test connection to MCP server."""
        try:
            client = Client(self.mcp_endpoint)
            async with client as async_client:
                await async_client.ping()
                return True
        except Exception as e:
            logger.error(f"MCP connection test failed: {e}")
            return False

    async def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process a query through the conversational agent with cumulative knowledge awareness."""
        try:
            logger.info(f"Processing query with cumulative knowledge: {query[:50]}...")

            user_id = "user"

            if not self.current_session:
                self.current_session = await self.session_service.create_session(
                    app_name="YuhHearDem",
                    user_id=user_id
                )
                logger.info(f"Created new session {self.current_session.id[:8]}...")
            else:
                # Restore knowledge graph from previous session if available
                persistent_turtle = self.current_session.state.get("persistent_knowledge_turtle", "")
                if persistent_turtle and len(self.knowledge_graph.graph) == 0:
                    logger.info("Restoring knowledge graph from previous session...")
                    success = self.knowledge_graph.add_turtle_data(persistent_turtle)
                    if success:
                        logger.info(f"Restored knowledge graph: {self.knowledge_graph.get_summary_stats()}")

            # Store conversation history and query in session state
            self.current_session.state['conversation_history'] = self.conversation_history
            self.current_session.state['original_query'] = query

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
                history_slice = self.conversation_history[-4:]
                for exchange in history_slice:
                    context += f"User: {exchange['user']}\n"
                    assistant_text = exchange.get('assistant', '')
                    if len(assistant_text) > 300:
                        assistant_text = assistant_text[:300] + "..."
                    context += f"Assistant: {assistant_text}\n\n"

            # Add knowledge graph summary if available
            kg_summary = self.knowledge_graph.get_knowledge_summary()
            context += f"\n{kg_summary}\n"

            # Create the prompt with context
            full_prompt = f"{context}CURRENT QUERY: {query}"

            # Run the conversational agent
            user_message = Content(role="user", parts=[Part.from_text(text=full_prompt)])

            final_response_text = ""
            knowledge_stats = {}

            events = runner.run(
                user_id=user_id,
                session_id=self.current_session.id,
                new_message=user_message
            )

            # Process events
            for event in events:
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # Only accumulate text content from WriterAgent
                                if event.author == "WriterAgent":
                                    final_response_text += part.text

            # Get final knowledge graph statistics
            knowledge_stats = self.knowledge_graph.get_summary_stats()

            # Update conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": final_response_text
            })

            # Trim history if too long
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                logger.info("Conversation history trimmed.")

            return final_response_text, {
                "success": True, 
                "knowledge_stats": knowledge_stats,
                "cumulative_knowledge": True
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error processing query: {str(e)}", {"success": False}

    def clear_history(self):
        """Clear conversation history but preserve accumulated knowledge."""
        self.conversation_history = []
        logger.info("Conversation history cleared. Knowledge graph preserved.")

    def clear_all(self):
        """Clear everything including accumulated knowledge."""
        self.conversation_history = []
        self.current_session = None
        self.knowledge_graph = CumulativeKnowledgeGraph()  # Reset knowledge graph
        logger.info("All state cleared including cumulative knowledge graph.")

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get current knowledge graph statistics."""
        return self.knowledge_graph.get_summary_stats()
