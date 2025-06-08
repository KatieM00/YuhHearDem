#!/usr/bin/env python3
"""
Parliamentary Transcript Processor

This script processes raw AI-transcribed parliamentary session data from Barbados
using LangChain and Google's Gemini model to clean, correct, and format the text
for Knowledge Graph extraction.

Requirements:
- langchain-google-genai
- python-dotenv (optional, for environment variables)

Usage:
    python transcript_processor.py input_file.txt output_file.txt
"""

import sys
import os
from pathlib import Path

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install langchain-google-genai python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

class ParliamentaryTranscriptProcessor:
    def __init__(self, api_key: str = None):
        """
        Initialize the processor with Gemini model.
        
        Args:
            api_key: Google API key. If None, will try to get from environment.
        """
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=api_key,
            temperature=1.0,
            thinking_budget=0  # Disable thinking mode
        )
        
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Return the comprehensive system prompt for transcript processing."""
        return """**Objective:**
The primary objective of this task is to transform raw, AI-transcribed parliamentary session data from Barbados into a clean, accurate, and time-aligned textual format suitable for subsequent Knowledge Graph (KG) extraction. This involves precise sentence segmentation, robust transcription correction, and careful handling of names and parliamentary conventions.

**Processing Instructions:**

1. **Comprehensive Transcription Correction:**
   - **Error Identification:** Thoroughly review the text content of all segments for transcription errors. These typically include spelling mistakes, grammatical inaccuracies, missing punctuation, and mishearings (e.g., homophones, words incorrectly segmented or combined by the AI).
   - **Clarity & Accuracy:** Correct all identified errors to ensure the text is grammatically sound, correctly spelled, and accurately reflects the likely spoken content. The goal is to produce highly readable and semantically coherent sentences.
   - **Barbadian English & Context:** Be mindful of Barbadian English nuances, common parliamentary terminology, and the formal tone expected in parliamentary proceedings. Ensure corrections align with this context (e.g., "Honourable" vs. "Honorable" - prefer "Honourable" as per UK/Commonwealth spelling).

2. **Strict Name & Entity Handling:**
   - **Extreme Caution with Proper Nouns:** Exercise the utmost caution with proper nouns, especially names of individuals (e.g., Members of Parliament, ministers, citizens mentioned), constituencies, specific legislation, or organizations.
   - **Ambiguity Resolution:** If there is *any* doubt regarding the correct spelling of a proper noun, or if the transcribed name sounds ambiguous or could be mistaken, **replace it with `[unknown]`**. This is crucial to prevent the propagation of factual errors into the Knowledge Graph.
   - **Parliamentary Titles:** Accurately identify and correctly spell common parliamentary titles and forms of address, such as "Mr. Speaker," "Madam President," "Honourable Member for [Constituency Name]," "Minister [Name]," "Prime Minister," "Leader of the Opposition," etc., ensuring they are capitalized correctly. Do *not* use `[unknown]` for these if their context is clear.

3. **Accurate Sentence Segmentation (NLP-driven):**
   - **Complete Sentences:** Using advanced Natural Language Processing (NLP) techniques, accurately identify and segment complete sentences. This will often involve merging text segments from multiple input JSON objects to form a single, coherent sentence, or occasionally splitting a single text segment if it contains more than one complete sentence.
   - **Logical Flow:** Ensure that each output line represents a complete, grammatically correct, and logically coherent sentence. Avoid creating fragmented sentences or run-on sentences.
   - **Punctuation:** Insert appropriate punctuation marks (periods, question marks, exclamation points, commas, etc.) to ensure sentence clarity and correctness.

4. **Timecode Assignment:**
   - **Sentence Start Time:** For each identified complete sentence, assign the start time (in integer seconds) of the *first* input JSON segment that contributes to that sentence.
   - **Rounding:** Convert the start time from the original string (e.g., "103.840") to an integer number of seconds, rounding *down* to the nearest whole second (e.g., "103.840" becomes 103, "109.360" becomes 109).

**Output Format:**
The output should be plain text, where each line represents a single, complete, time-aligned, and corrected sentence. The format for each line must be:

`[integer_seconds] [Complete and corrected sentence]`

**Key Considerations:**
- **Precision is Paramount:** Every correction and segmentation decision impacts the quality of the downstream Knowledge Graph.
- **Contextual Understanding:** Leverage the context of parliamentary proceedings to inform decisions, especially regarding names and formal language.
- **Balance of Correction and Preservation:** While fixing errors, avoid altering the fundamental meaning or intent of the speaker's original words.
- **No Interpretation:** Do not summarize, interpret, or add information not explicitly stated. Focus solely on cleaning and structuring the transcript.

Process the following transcript data and return ONLY the formatted output with no additional commentary or explanation:"""

    def load_transcript(self, file_path: str) -> str:
        """
        Load transcript data from a file as raw text.
        
        Args:
            file_path: Path to the input file containing transcript data
            
        Returns:
            Raw file content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading transcript: {str(e)}")

    def process_transcript(self, input_file: str):
        """
        Process the entire transcript file using Gemini's large context window.
        
        Args:
            input_file: Path to input JSON transcript file
        """
        # Generate output filename by replacing .json with .txt
        input_path = Path(input_file)
        if input_path.suffix.lower() != '.json':
            raise ValueError(f"Input file must be a .json file, got: {input_file}")
        
        output_file = str(input_path.with_suffix('.txt'))
        
        print(f"Loading transcript from: {input_file}")
        content = self.load_transcript(input_file)
        print(f"Loaded transcript file ({len(content)} characters)")
        
        # Create messages for the conversation
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Please process this transcript data:\n\n{content}")
        ]
        
        print("Processing transcript with Gemini model...")
        try:
            # Get response from the model
            response = self.llm.invoke(messages)
            
            # Debug: Print response details
            print(f"Response type: {type(response)}")
            print(f"Response content length: {len(response.content) if response.content else 'None'}")
            
            result = response.content.strip() if response.content else ""
            
            if not result:
                print("Warning: Model returned empty content!")
                print("This could be due to:")
                print("- Content filtering by the model")
                print("- Input file being too large")
                print("- API key issues")
                print("- Model response format issues")
                
                # Let's also check the raw response
                print(f"Raw response object: {response}")
                return
            
            # Save to output file
            print(f"Saving processed transcript to: {output_file}")
            print(f"Content preview (first 200 chars): {result[:200]}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"Processing complete! Output saved to: {output_file}")
            print(f"Output file size: {len(result)} characters")
            
        except Exception as e:
            print(f"Exception details: {e}")
            print(f"Exception type: {type(e)}")
            raise Exception(f"Error processing with Gemini model: {str(e)}")

def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python transcript_processor.py <input_file.json>")
        print("\nExample:")
        print("python transcript_processor.py transcript.json")
        print("Output will be saved as: transcript.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize processor
        processor = ParliamentaryTranscriptProcessor()
        
        # Process the transcript
        processor.process_transcript(input_file)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo set up Google API key:")
        print("1. Get an API key from Google AI Studio: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable: export GOOGLE_API_KEY='your-api-key'")
        print("3. Or create a .env file with: GOOGLE_API_KEY=your-api-key")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()