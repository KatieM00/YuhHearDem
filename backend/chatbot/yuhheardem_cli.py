#!/usr/bin/env python3
"""
YuhHearDem CLI Interface - Interactive Command Line Interface
============================================================

Command line interface for the YuhHearDem Parliamentary Research System
with colorful output, conversation history, and knowledge graph statistics.
"""

import asyncio
import os
import sys
import logging
from typing import Dict, Any

# CLI-specific imports
try:
    import click
    import colorama
    from colorama import Fore, Back, Style
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required CLI package: {e}")
    print("Please install: pip install click colorama python-dotenv")
    sys.exit(1)

# Import the core system
try:
    from yuhheardem_core import YuhHearDemADK
except ImportError:
    print("Failed to import yuhheardem_core. Make sure yuhheardem_core.py is available.")
    sys.exit(1)

# Initialize colorama for Windows support
colorama.init()

# Load environment variables
load_dotenv()

# Configure logging to be less verbose for CLI
logging.basicConfig(level=logging.WARNING)


class YuhHearDemCLI:
    """CLI wrapper for YuhHearDem with enhanced user interface."""
    
    def __init__(self, mcp_endpoint: str, google_api_key: str):
        self.core_system = YuhHearDemADK(mcp_endpoint, google_api_key)
    
    def print_header(self):
        """Print application header with colorful ASCII art."""
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
    
    def show_history(self):
        """Display conversation history with colors."""
        if not self.core_system.conversation_history:
            print(f"{Fore.CYAN}No conversation history yet.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}📚 Conversation History:{Style.RESET_ALL}")
        for i, exchange in enumerate(self.core_system.conversation_history, 1):
            print(f"\n{Fore.YELLOW}Exchange {i}:{Style.RESET_ALL}")
            print(f"{Fore.BLUE}User:{Style.RESET_ALL} {exchange['user']}")
            print(f"{Fore.GREEN}Assistant:{Style.RESET_ALL} {exchange['assistant'][:200]}...")
    
    def show_knowledge_stats(self):
        """Display statistics about accumulated knowledge."""
        if not self.core_system.current_session:
            print(f"{Fore.CYAN}No active session - no accumulated knowledge.{Style.RESET_ALL}")
            return
        
        print(f"{Fore.CYAN}📊 Knowledge Graph Statistics:{Style.RESET_ALL}")
        print(f"- Session ID: {self.core_system.current_session.id}")
        print(f"- Conversation exchanges: {len(self.core_system.conversation_history)}")
        print(f"- Session active: {self.core_system.current_session is not None}")
        print(f"- Knowledge persists across queries within this session")
        
        # Note: More detailed stats would require access to session state
        print(f"\n{Fore.YELLOW}💡 Tip: Use 'clear' to reset conversation but keep knowledge,")
        print(f"or 'clear_all' to reset everything including accumulated data.{Style.RESET_ALL}")
    
    def print_connection_status(self, success: bool):
        """Print connection test results."""
        if success:
            print(f"{Fore.GREEN}✅ Connected to YuhHearDem MCP server{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ Failed to connect to MCP server{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}⚠️  Some features may not work properly{Style.RESET_ALL}")
    
    def print_query_processing(self, query: str):
        """Show query processing started."""
        print(f"\n{Fore.BLUE}🔍 Processing: {query[:50]}{'...' if len(query) > 50 else ''}{Style.RESET_ALL}")
    
    def print_response(self, response: str, status: Dict[str, Any]):
        """Print the response with formatting."""
        print(f"\n{Fore.CYAN}YuhHearDem:{Style.RESET_ALL}")
        print(response)
        
        # Show processing stats if available
        if status.get("turtle_count", 0) > 0:
            print(f"\n{Fore.MAGENTA}📊 Found {status['turtle_count']} parliamentary datasets{Style.RESET_ALL}")
    
    def print_error(self, error_msg: str):
        """Print error messages with formatting."""
        print(f"\n{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
    
    def print_command_help(self):
        """Print available commands."""
        help_text = f"""
{Fore.CYAN}Available Commands:{Style.RESET_ALL}
{Fore.GREEN}• quit, exit, q{Style.RESET_ALL} - Exit the application
{Fore.GREEN}• clear{Style.RESET_ALL} - Clear conversation history (keep knowledge graph)
{Fore.GREEN}• clear_all{Style.RESET_ALL} - Clear everything including accumulated knowledge
{Fore.GREEN}• stats{Style.RESET_ALL} - Show knowledge graph statistics
{Fore.GREEN}• history{Style.RESET_ALL} - Show conversation history
{Fore.GREEN}• help{Style.RESET_ALL} - Show this help message

{Fore.YELLOW}Tips:{Style.RESET_ALL}
• Ask follow-up questions - the system remembers context
• Knowledge accumulates across questions for richer responses
• Video sources are automatically retrieved and cited
"""
        print(help_text)
    
    async def run_interactive(self):
        """Run the interactive CLI with enhanced UI."""
        self.print_header()
        
        # Test connection
        print(f"{Fore.BLUE}🔧 Testing connection to MCP server...{Style.RESET_ALL}")
        connection_success = await self.core_system.test_connection()
        self.print_connection_status(connection_success)
        
        print(f"\n{Fore.GREEN}Ready! Type your question or 'help' for commands.{Style.RESET_ALL}")
        
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
                    self.core_system.clear_history()
                    print(f"{Fore.YELLOW}🧹 Conversation history cleared. Knowledge graph preserved.{Style.RESET_ALL}")
                    continue
                elif user_input.lower() == 'clear_all':
                    self.core_system.clear_all()
                    print(f"{Fore.YELLOW}🧹 Everything cleared including accumulated knowledge.{Style.RESET_ALL}")
                    continue
                elif user_input.lower() == 'stats':
                    self.show_knowledge_stats()
                    continue
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'help':
                    self.print_command_help()
                    continue
                
                # Process query with visual feedback
                self.print_query_processing(user_input)
                
                # Show agent activity
                print(f"{Fore.MAGENTA}🤖 Conversational agent analyzing...{Style.RESET_ALL}")
                
                response, status = await self.core_system.process_query(user_input)
                
                # Display response
                self.print_response(response, status)
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.CYAN}Thank you for using YuhHearDem! 👋{Style.RESET_ALL}")
                break
            except Exception as e:
                self.print_error(f"An error occurred: {e}")
    
    async def run_single_query(self, query: str):
        """Run a single query and exit."""
        print(f"{Fore.BLUE}🔍 Processing query: {query}{Style.RESET_ALL}")
        
        # Test connection
        if not await self.core_system.test_connection():
            self.print_error("Failed to connect to MCP server")
            return
        
        self.print_connection_status(True)
        
        try:
            response, status = await self.core_system.process_query(query)
            print(f"\n{response}")
            
            if status.get("turtle_count", 0) > 0:
                print(f"\n{Fore.MAGENTA}📊 Processed {status['turtle_count']} parliamentary datasets{Style.RESET_ALL}")
                
        except Exception as e:
            self.print_error(f"Query processing failed: {e}")


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
    """
    YuhHearDem CLI - Conversational Parliamentary Research System
    
    An interactive command-line interface for researching Barbados Parliament
    discussions with AI-powered agents and cumulative knowledge building.
    """
    
    if debug:
        logging.getLogger().setLevel(logging.INFO)
        print(f"{Fore.YELLOW}🐛 Debug logging enabled{Style.RESET_ALL}")
    
    if not api_key:
        # Try to get from environment
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')
        
    if not api_key:
        print(f"{Fore.RED}❌ Google API key required.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Set GOOGLE_API_KEY or GOOGLE_GENAI_API_KEY environment variable{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Or use: {Fore.WHITE}yuhheardem_cli.py --api-key YOUR_KEY{Style.RESET_ALL}")
        sys.exit(1)
    
    # Set API key for Google ADK
    os.environ['GOOGLE_API_KEY'] = api_key
    os.environ['GOOGLE_GENAI_API_KEY'] = api_key
    
    # Create CLI wrapper
    cli = YuhHearDemCLI(endpoint, api_key)
    
    async def run():
        if query:
            # Single query mode
            await cli.run_single_query(query)
        else:
            # Interactive mode
            await cli.run_interactive()
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print(f"\n{Fore.CYAN}Goodbye! 👋{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
