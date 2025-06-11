#!/usr/bin/env python3
"""
YuhHearDem Core ADK System - Parliamentary Research Components (SIMPLIFIED)
============================================================

Core multi-agent system using Google Agent Development Kit with:
1. ConversationalAgent: Root agent that handles conversation and delegates to research
2. ResearchPipeline: Sequential pipeline with:
   - ResearcherAgent: Performs searches using MCP tools
   - ProvenanceAgent: Custom agent that fetches YouTube URLs for found entities
   - WriterAgent: Synthesizes findings into cited responses (gets turtle data in constructor)
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
    turtle_results: List[str] = field(default_factory=list) # Data from current research step
    cumulative_turtle_results: List[str] = field(default_factory=list) # Data across session
    knowledge_graph: Graph = field(default_factory=Graph) # Cumulative graph
    enriched_turtle: str = "" # Enriched data for current research step
    cumulative_enriched_turtle: str = "" # Enriched data across session


class MCPToolWrapper:
    """Wrapper to call MCP tools."""

    def __init__(self, mcp_endpoint: str):
        self.mcp_endpoint = mcp_endpoint
        # Client is not async-context-managed in __init__
        self.mcp_client = Client(mcp_endpoint)

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





# Synchronous wrappers for async MCP calls for use by LlmAgent tools
def sync_mcp_tool_caller(mcp_wrapper: MCPToolWrapper, tool_name: str):
    """Creates a synchronous function that calls an async MCP tool with proper ADK response format."""
    def sync_tool_func_with_collect(query: str, **kwargs) -> str:
        """Synchronous wrapper for the async MCP tool call with ADK-compatible response."""
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
                        if result:
                            logger.debug(f"📄 First 200 chars of result: {str(result)[:200]}")
                        return result
                    except Exception as e:
                        logger.error(f"❌ Error in thread for {tool_name}: {e}")
                        return f"Error in thread: {str(e)}"
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async_in_thread)
                    result = future.result(timeout=30)
                    
            except RuntimeError:
                logger.info(f"🔄 No event loop running, creating new one for {tool_name}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    args = {"query": query}
                    args.update(kwargs)
                    logger.info(f"🔍 Calling MCP tool {tool_name} with args: {args}")
                    result = loop.run_until_complete(mcp_wrapper.call_tool(tool_name, args))
                    logger.info(f"📥 MCP tool {tool_name} returned: {len(str(result)) if result else 0} characters")
                    if result:
                        logger.debug(f"📄 First 200 chars of result: {str(result)[:200]}")
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
            
            # Process result
            if not result:
                logger.warning(f"⚠️ {tool_name} returned empty/None result")
                success_msg = f"Search completed successfully for query '{query}'. No results found in parliamentary database."
                logger.info(f"🎯 Returning success message for {tool_name}: {success_msg}")
                return success_msg
            
            logger.info(f"✅ {tool_name} completed successfully, result length: {len(str(result))}")
            
            # Check if result looks like turtle data
            result_str = str(result)
            is_turtle = ('@prefix' in result_str or 'bbp:' in result_str or result_str.strip().startswith('#'))
            logger.info(f"🐢 Result appears to be turtle data: {is_turtle}")
            
            if is_turtle:
                sync_tool_func_with_collect.turtle_collector.append(result_str)
                logger.info(f"📚 Added to turtle collector. Total items: {len(sync_tool_func_with_collect.turtle_collector)}")
                
                success_msg = f"Successfully searched parliamentary database for '{query}'. Found relevant parliamentary data."
                logger.info(f"🎯 Returning success message for {tool_name}: {success_msg}")
                return success_msg
            else:
                logger.warning(f"⚠️ Result doesn't look like turtle data: {result_str[:100]}...")
                success_msg = f"Search completed for '{query}'. Data retrieved from parliamentary system."
                logger.info(f"🎯 Returning success message for {tool_name}: {success_msg}")
                return success_msg
            
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

    sync_tool_func_with_collect.turtle_collector = []
    sync_tool_func_with_collect.__name__ = tool_name
    return sync_tool_func_with_collect


def create_mcp_tools(mcp_wrapper: MCPToolWrapper):
    """Create tool functions for ADK agents, wrapping async MCP calls with enhanced diagnostics."""
    
    logger.info("🛠️ Creating MCP tools with enhanced diagnostics")

    # Create synchronous wrappers for each tool
    hybrid_search_tool = sync_mcp_tool_caller(mcp_wrapper, "hybrid_search_turtle")
    authority_search_tool = sync_mcp_tool_caller(mcp_wrapper, "authority_search_turtle")
    topic_search_tool = sync_mcp_tool_caller(mcp_wrapper, "topic_search_turtle")

    # Define tool metadata (args structure) - LLM needs this
    hybrid_search_tool.__name__ = "hybrid_search_turtle"
    hybrid_search_tool.__doc__ = """
        Hybrid search combining PageRank importance with semantic similarity.

        Args:
            query: The search query string
            hops: Number of hops to explore (typically 2 for broader results)
            limit: Maximum number of results to return (typically 5-8)
        """

    authority_search_tool.__name__ = "authority_search_turtle"
    authority_search_tool.__doc__ = """
        Search for authoritative nodes (high PageRank) related to the query.

        Args:
            query: The search query string
            hops: Number of hops to explore (typically 2 for broader results)
            limit: Maximum number of results to return (typically 5-8)
            max_rank: Maximum PageRank value to filter (typically 1000)
        """

    topic_search_tool.__name__ = "topic_search_turtle"
    topic_search_tool.__doc__ = """
        Search for nodes important within the specific topic domain.

        Args:
            query: The search query string
            hops: Number of hops to explore (typically 2 for broader results)
            limit: Maximum number of results to return (typically 5-8)
        """
    
    # Combine collectors from all tools for easy access later
    all_collectors = [hybrid_search_tool.turtle_collector, authority_search_tool.turtle_collector, topic_search_tool.turtle_collector]
    setattr(hybrid_search_tool, '_all_collectors', all_collectors)
    setattr(authority_search_tool, '_all_collectors', all_collectors)
    setattr(topic_search_tool, '_all_collectors', all_collectors)

    logger.info("✅ MCP tools created successfully with enhanced diagnostics")
    return [hybrid_search_tool, authority_search_tool, topic_search_tool]


class ResearcherAgent(LlmAgent):
    """Agent that performs parliamentary research using MCP tools."""

    def __init__(self, model: str, mcp_wrapper: MCPToolWrapper, **kwargs):
        # Create MCP-based tools wrapped for synchronous use by LlmAgent
        tools = create_mcp_tools(mcp_wrapper)

        super().__init__(
            name="ResearcherAgent",
            model=model,
            description="Performs thorough parliamentary research using search tools",
            instruction="""You are a Parliamentary Research Assistant performing background research.

Your job is to systematically search for information using the available tools, taking into account the conversation history above.

ALWAYS USE 1 HOP

HISTORY-AWARE RESEARCH STRATEGY:
1. Consider what has been discussed before - build on previous searches rather than repeat them
2. Look for connections between the current query and previously mentioned entities/topics
3. If this is a follow-up question, focus on aspects not covered in previous searches
4. Use entity names and topics from the conversation history to inform your searches

SEARCH PROCESS:
1. Start with hybrid_search_turtle for the main query
2. Try multiple variations and related terms (synonyms, broader/narrower concepts)
3. Use authority_search_turtle to find what parliamentary leaders said
4. Use topic_search_turtle for focused topical searches
5. Be thorough - make 5-10 searches with different approaches
6. IMPORTANT: Use conversation context to guide search terms and avoid redundant searches

TEMPORAL FOCUS: Prioritize 2025 information when available, but also search for historical context and trends. Try adding "2025", "2024", "recent" to queries when relevant.

CONTEXT-AWARE Search strategy examples:
- If health was discussed before and user asks about "budget", try "health budget", "healthcare funding"
- If a Minister was mentioned before, include their name in relevant searches
- If looking for recent developments, reference previously discussed time periods
- Connect current query to entities from conversation history when relevant

EXAMPLE CONTEXT USAGE:
- Previous context: "Minister of Health discussed COVID-19"
- Current query: "vaccination rates"
- Enhanced searches: "Minister of Health vaccination rates", "COVID-19 vaccination Barbados", "vaccination program 2025"

After completing all searches, respond with exactly: "Research complete."

This single phrase signals completion and prevents infinite loops. 
Do not add any other content that wasn't explicitly instructed.
""",
            tools=tools,
            generate_content_config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.3
            ),
            **kwargs
        )

        # Store tools and mcp_wrapper for internal use
        object.__setattr__(self, '_tools', tools)
        object.__setattr__(self, '_mcp_wrapper', mcp_wrapper)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Override to manage state updates based on collected turtle data."""

        # Get the list of collectors from one of the tools
        all_collectors = getattr(self._tools[0], '_all_collectors', [])
        # Clear collectors before starting the run for this invocation
        for collector in all_collectors:
            collector.clear()

        # CRITICAL FIX: Actually yield the events from the parent LlmAgent
        async for event in super()._run_async_impl(ctx):
            yield event

        # After the LLM is done, collect the raw turtle data from the tools' collectors.
        raw_turtle_data = []
        for collector in all_collectors:
            raw_turtle_data.extend(collector)
            collector.clear()

        if raw_turtle_data:
            logger.info(f"ResearcherAgent: Collected {len(raw_turtle_data)} raw turtle datasets")

            # Get existing cumulative turtle data from session state
            cumulative_turtle_data = ctx.session.state.get("cumulative_turtle_results", [])
            if not isinstance(cumulative_turtle_data, list):
                logger.warning(f"cumulative_turtle_results in state is not a list, resetting. Type: {type(cumulative_turtle_data)}")
                cumulative_turtle_data = []

            # Accumulate new data with existing data
            cumulative_turtle_data.extend(raw_turtle_data)

            logger.info(f"ResearcherAgent: Total accumulated turtle datasets: {len(cumulative_turtle_data)}")

            # Store both the new data (for this turn's provenance) and cumulative data
            ctx.session.state["turtle_results"] = raw_turtle_data
            ctx.session.state["cumulative_turtle_results"] = cumulative_turtle_data

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
            logger.warning("ResearcherAgent: No turtle data collected from tools during this run.")
            yield Event(
                author=self.name,
                content=Content(
                    role="assistant",
                    parts=[Part.from_text(text="No new parliamentary data collected in this step.")]
                ),
                actions=EventActions(state_delta={
                    "turtle_results": [],
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

            logger.info(f"ProvenanceAgent: Processing {len(cumulative_turtle_results)} turtle datasets (cumulative knowledge)")

            # Parse cumulative data into a combined graph
            combined_graph = Graph()
            combined_graph.bind("bbp", BBP)
            combined_graph.bind("schema", SCHEMA)
            combined_graph.bind("prov", PROV)

            parsed_count = 0
            for turtle_data in cumulative_turtle_results:
                 try:
                     temp_graph = Graph()
                     temp_graph.parse(data=turtle_data, format="turtle")
                     combined_graph += temp_graph
                     parsed_count += 1
                 except Exception as parse_error:
                     logger.warning(f"ProvenanceAgent: Failed to parse individual turtle dataset for enrichment: {parse_error}")
                     continue

            if parsed_count == 0:
                logger.warning("ProvenanceAgent: Failed to parse any cumulative turtle datasets")
                ctx.session.state["enriched_turtle"] = str(cumulative_turtle_results)
                ctx.session.state["cumulative_enriched_turtle"] = str(cumulative_turtle_results)

                yield Event(
                    author=self.name,
                    content=Content(
                        role="assistant",
                        parts=[Part.from_text(text="Failed to parse cumulative turtle data for enrichment.")]
                    ),
                    actions=EventActions(state_delta={
                        "enriched_turtle": str(cumulative_turtle_results),
                        "cumulative_enriched_turtle": str(cumulative_turtle_results)
                    })
                )
                return

            logger.info(f"ProvenanceAgent: Successfully parsed {parsed_count} cumulative turtle datasets into graph.")

            # Extract BBP URIs (parliamentary entities) from the combined graph
            all_uris = set()
            for subj in combined_graph.subjects():
                if isinstance(subj, URIRef) and str(subj).startswith(str(BBP)):
                    all_uris.add(str(subj))

            if not all_uris:
                logger.warning("ProvenanceAgent: No parliamentary entities found in cumulative turtle data")
                enriched_turtle = combined_graph.serialize(format='turtle')
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
                prov_graph = Graph()
                try:
                    prov_graph.parse(data=provenance_result_str, format="turtle")
                    combined_graph += prov_graph

                    # Count video sources added
                    video_count = len(list(prov_graph.subjects(predicate=URIRef(SCHEMA["url"]))))
                    total_video_count = len(list(combined_graph.subjects(predicate=URIRef(SCHEMA["url"]))))

                    logger.info(f"ProvenanceAgent: Retrieved {video_count} new video sources. Total videos in knowledge graph: {total_video_count}")

                    content_msg = f"Enriched cumulative knowledge graph with {video_count} new video sources. Total: {total_video_count} videos."

                except Exception as e:
                    logger.error(f"ProvenanceAgent: Failed to parse provenance result: {e}")
                    content_msg = f"Retrieved provenance data but failed to parse it: {str(e)}"
            else:
                logger.warning(f"ProvenanceAgent: Failed to retrieve video sources: {provenance_result_str}")
                content_msg = f"Failed to retrieve video sources: {provenance_result_str}"

            # Serialize the enriched cumulative graph
            enriched_turtle = combined_graph.serialize(format='turtle')

            # Store both the result from this step and the cumulative enriched turtle in session state
            ctx.session.state["enriched_turtle"] = enriched_turtle
            ctx.session.state["cumulative_enriched_turtle"] = enriched_turtle

            yield Event(
                author=self.name,
                content=Content(
                    role="assistant",
                    parts=[Part.from_text(text=content_msg)]
                ),
                actions=EventActions(state_delta={
                    "cumulative_enriched_turtle": enriched_turtle,
                })
            )

            logger.info("ProvenanceAgent: Video source enrichment completed")

        except Exception as e:
            logger.error(f"ProvenanceAgent failed: {e}")
            error_msg = f"Video source enrichment failed: {str(e)}"

            # FALLBACK: If provenance fails, still provide the raw turtle data to WriterAgent
            try:
                logger.info("ProvenanceAgent: Attempting fallback - combining raw turtle data without provenance")
                fallback_graph = Graph()
                fallback_graph.bind("bbp", BBP)
                fallback_graph.bind("schema", SCHEMA)
                fallback_graph.bind("prov", PROV)
                
                cumulative_turtle_results = ctx.session.state.get("cumulative_turtle_results", [])
                
                for turtle_data in cumulative_turtle_results:
                    try:
                        temp_graph = Graph()
                        temp_graph.parse(data=turtle_data, format="turtle")
                        fallback_graph += temp_graph
                    except Exception as parse_error:
                        logger.warning(f"ProvenanceAgent fallback: Failed to parse turtle dataset: {parse_error}")
                        continue
                
                fallback_turtle = fallback_graph.serialize(format='turtle')
                ctx.session.state["cumulative_enriched_turtle"] = fallback_turtle
                logger.info(f"ProvenanceAgent: Stored fallback turtle data ({len(fallback_turtle)} characters)")
                
                error_msg += f" Using raw parliamentary data as fallback ({len(fallback_turtle)} characters)."
                
            except Exception as fallback_error:
                logger.error(f"ProvenanceAgent: Fallback also failed: {fallback_error}")
                error_msg += " Fallback to raw data also failed."

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

    def __init__(self, model: str, turtle_data: str = "", user_query: str = "", **kwargs):
        # Build the instruction with the turtle data directly in the prompt
        instruction = f"""You are YuhHearDem, a civic AI assistant that helps users understand Barbados Parliament discussions.


Based on the parliamentary data below, provide a comprehensive answer to the user's question. 

RESPONSE GUIDELINES:
- Write in clear, accessible language for citizens, journalists, and students
- Structure your response with: Summary → Key Discussion Points with inline citations
- Use INLINE MARKDOWN LINKS for all specific claims, quotes, and references
- The ONLY links must be YouTube URLs from the turtle data, do not hallucinate links
- Look for provenance statements in the turtle data that have schema:url, schema:text, schema:videoTitle properties
- Format video links as: [quoted text or claim](YouTube_URL) 
- When referencing parliamentary discussions, link the specific statement or fact
- If this relates to previous conversation topics, acknowledge the connection naturally
- If limited information was found, explain what was searched and suggest related topics
- Use parliamentary terminology appropriately (e.g., "parliamentary sessions", "Minister", "constituency")
- Be objective and factual, focusing on what was actually discussed in Parliament
- DO NOT include a separate "Sources" section - all references should be inline markdown links
- ONLY cite video URLs that are actually present in the turtle data - do not hallucinate links
- Never include any other types of links or sources, only YouTube URLs from the turtle data

CRITICAL: Only use video links that appear in the turtle data with schema:url properties. If no video sources are found in the turtle data, state that no video sources were available and explain what information was found instead.

If substantial information was found, provide detailed coverage with inline citations. If little was found, clearly explain the research conducted and suggest alternative search approaches.

Remember: You're helping regular citizens understand Parliament. Keep it real, keep it clear, cite your video sources, and make it sound like you're referencing actual parliamentary proceedings!

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
        # Only pass the researcher and provenance agents to the parent
        # WriterAgent will be created dynamically with the turtle data
        super().__init__(
            name="ResearchPipeline",
            description="Sequential pipeline for parliamentary research with video source enrichment",
            sub_agents=[researcher, provenance],  # No writer here - we'll create it dynamically
            **kwargs
        )
        # Store writer_model using object.__setattr__ to avoid SequentialAgent field restrictions
        object.__setattr__(self, '_writer_model', writer_model)

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Override to handle the simplified data flow between sequential agents."""

        # Store the original query for the WriterAgent
        original_query = "parliamentary matters"
        if hasattr(ctx, 'initial_user_content') and ctx.initial_user_content:
            if hasattr(ctx.initial_user_content, 'parts') and ctx.initial_user_content.parts:
                for part in ctx.initial_user_content.parts:
                    if hasattr(part, 'text') and part.text:
                        query_match = re.search(r"CURRENT QUERY:\s*(.*)", part.text, re.DOTALL)
                        if query_match:
                            original_query = query_match.group(1).strip()
                        else:
                             original_query = part.text.strip()
                        break
        ctx.session.state['original_query'] = original_query
        logger.info(f"ResearchPipeline: Stored original query: {original_query[:50]}...")

        # 1. Run ResearcherAgent
        logger.info("ResearchPipeline: Running ResearcherAgent...")
        async for event in self.sub_agents[0].run_async(ctx):
            yield event

        # 2. Run ProvenanceAgent
        logger.info("ResearchPipeline: Running ProvenanceAgent...")
        async for event in self.sub_agents[1].run_async(ctx):
            yield event

        # 3. Get the combined turtle data and create a new WriterAgent with it
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
        
        # 4. Create a new WriterAgent instance with the turtle data
        writer_agent = WriterAgent(
            model=self._writer_model,
            turtle_data=cumulative_enriched_turtle,
            user_query=original_query,
            generate_content_config=types.GenerateContentConfig(
                temperature=0.7,
            ),
            planner=BuiltInPlanner(
                thinking_config=types.ThinkingConfig(thinking_budget=1024)
            )
        )
        
        # 5. Run the WriterAgent
        logger.info("ResearchPipeline: Running dynamically created WriterAgent...")
        async for event in writer_agent.run_async(ctx):
            yield event

        logger.info("ResearchPipeline: Pipeline execution completed.")


class ConversationalAgent(LlmAgent):
    """Root agent that handles conversation and delegates to research when needed."""

    def __init__(self, model: str, research_pipeline: ResearchPipeline, **kwargs):
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

IMPORTANT: For parliamentary questions, respond with a friendly acknowledgment and then call transfer_to_agent. 
Examples:
- "YuhHearDem! I'll look that up for you in the parliamentary records."
- "Let me search through the parliamentary discussions about that topic."
- "I'll find what Parliament has said about this issue."

The research will be conducted separately and the results will be provided as a follow-up response.
""",
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

        # Initialize Google GenAI with API key
        if self.api_key:
             try:
                 try:
                     import google.generativeai as genai
                     genai.configure(api_key=self.api_key)
                     logger.info("✅ Google Generative AI configured successfully")
                 except ImportError:
                     # google.generativeai is not installed, but ADK might still work
                     logger.warning("⚠️ google.generativeai not installed, but continuing with ADK")
             except Exception as e:
                 logger.error(f"❌ Failed to configure Google Generative AI: {e}")
                 self.api_key = None

        # Define models - use environment variables or defaults
        researcher_model = os.getenv("RESEARCHER_MODEL", "gemini-2.0-flash")
        writer_model = os.getenv("WRITER_MODEL", "gemini-2.5-flash-preview-05-20")
        conversational_model = os.getenv("CONVERSATIONAL_MODEL", "gemini-2.0-flash")

        # Create the research pipeline agents
        self.researcher = ResearcherAgent(
            model=researcher_model,
            mcp_wrapper=self.mcp_wrapper
        )

        self.provenance_agent = ProvenanceAgent(
            mcp_wrapper=self.mcp_wrapper
        )

        # Create the research pipeline (SequentialAgent)
        # WriterAgent will be created dynamically with turtle data
        self.research_pipeline = ResearchPipeline(
            researcher=self.researcher,
            provenance=self.provenance_agent,
            writer_model=writer_model
        )

        # Create the conversational root agent (LlmAgent)
        self.conversational_agent = ConversationalAgent(
            model=conversational_model,
            research_pipeline=self.research_pipeline
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
        """Process a query through the conversational agent (non-streaming version)."""
        try:
            logger.info(f"Processing non-stream query: {query[:50]}...")

            user_id = "user"

            if not self.current_session:
                 self.current_session = await self.session_service.create_session(
                    app_name="YuhHearDem",
                    user_id=user_id
                 )
                 logger.info(f"Created new session {self.current_session.id[:8]}... for non-stream query.")

            # Create runner for conversational agent
            runner = Runner(
                agent=self.conversational_agent,
                session_service=self.session_service,
                app_name="YuhHearDem"
            )

            # Build conversation context from global history
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

            # Create the prompt with context
            full_prompt = f"{context}CURRENT QUERY: {query}"

            # Run the conversational agent
            user_message = Content(role="user", parts=[Part.from_text(text=full_prompt)])

            final_response_text = ""
            turtle_results = []

            events = runner.run(
                user_id=user_id,
                session_id=self.current_session.id,
                new_message=user_message
            )

            # Process events (synchronous iteration)
            for event in events:
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                # Only accumulate text content if it's from the WriterAgent
                                if event.author == "WriterAgent":
                                    final_response_text += part.text

                            elif hasattr(part, 'function_response') and part.function_response:
                                if hasattr(part.function_response, 'response'):
                                    fr_response = part.function_response.response

                                    if isinstance(fr_response, dict):
                                        for key in ['result', 'content', 'turtle_results', 'enriched_turtle', 'cumulative_turtle_results', 'cumulative_enriched_turtle']:
                                            if key in fr_response:
                                                value = str(fr_response[key])
                                                if value and ('@prefix' in value or value.strip().startswith('#') or 'bbp:' in value):
                                                    turtle_results.append(value)
                                    elif isinstance(fr_response, str):
                                        if fr_response and ('@prefix' in fr_response or fr_response.strip().startswith('#') or 'bbp:' in fr_response):
                                            turtle_results.append(fr_response)

            # Clean the accumulated WriterAgent text
            final_response_text = final_response_text

            # Update conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": final_response_text
            })

            # Trim history if too long
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                logger.info("Conversation history trimmed in process_query.")

            return final_response_text, {"success": True, "turtle_count": len(turtle_results)}

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error processing query: {str(e)}", {"success": False}

    def clear_history(self):
        """Clear conversation history but preserve accumulated knowledge in session state."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")

    def clear_all(self):
        """Clear everything including accumulated knowledge (by resetting the session)."""
        self.conversation_history = []
        self.current_session = None
        logger.info("All state cleared including accumulated knowledge.")
