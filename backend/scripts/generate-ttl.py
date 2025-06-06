#!/usr/bin/env python3
"""
Parliamentary Transcript to RDF Converter

This script converts processed parliamentary transcripts into RDF Turtle format,
with validation and error correction using LangChain and Google's Gemini model.

Requirements:
- langchain-google-genai
- rdflib
- python-dotenv (optional, for environment variables)

Usage:
    python rdf_converter.py input_transcript.txt
"""

import sys
import os
from pathlib import Path
from typing import Optional

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    from dotenv import load_dotenv
    import rdflib
    from rdflib import Graph
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install langchain-google-genai rdflib python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

class RDFConverter:
    def __init__(self, api_key: str = None):
        """
        Initialize the RDF converter with Gemini models.
        
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
        
        # Primary LLM for initial conversion (temperature 0.0 for consistency)
        self.llm_convert = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=api_key,
            temperature=0.0,
            thinking_budget=0  # Disable thinking mode
        )
        
        # Secondary LLM for error correction (temperature 1.0 for creativity in fixing)
        self.llm_correct = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=api_key,
            temperature=1.0,
            thinking_budget=0  # Disable thinking mode
        )
        
        self.conversion_prompt = None  # Will be set dynamically
        self.correction_prompt = self._get_correction_prompt()
    
    def _get_conversion_prompt(self, video_id: str) -> str:
        """Return the system prompt for transcript to RDF conversion."""
        return f"""Your task is to convert the provided parliamentary session transcript into a comprehensive RDF knowledge graph in Turtle (.ttl) format. The graph should capture speakers, their roles, constituencies, political affiliations, stances on the debated Bill, key arguments, mentioned organizations, legislation, events, and other relevant entities and their relationships. Crucially, each generated RDF claim must include provenance linking it to the specific time segment in the video that corresponds to the supporting text in the transcript.

**Source Video ID:** `{video_id}` (YouTube video)

**Input Transcript Format:**
Each line of the transcript begins with a numerical time offset in seconds from the start of the video, followed by the spoken text.
Example:
`54 This is [unknown] H, Member for St. Peter.`
`61 Thank you very much, Mr. Speaker.`
`62 I rise to join this debate to continue the serious, mature approach that was taken by he who led in this debate, the Honourable Member for St. Michael South.`

**General Instructions:**

1.  **Output Format:** Generate a single RDF graph in Turtle (.ttl) syntax.
2.  **Prefixes:** Begin the output with the following RDF prefixes:
    ```turtle
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix schema: <http://schema.org/> .
    @prefix org: <http://www.w3.org/ns/org#> .
    @prefix bbp: <http://example.com/barbados-parliament-ontology#> .
    @prefix sess: <http://example.com/barbados-parliament-session/> .
    @prefix prov: <http://www.w3.org/ns/prov#> .
    ```
3.  **Main Entity (`bbp:VideoRecording`):** All data generated from this transcript should ultimately be linked to a single `bbp:VideoRecording` instance. Use the URI `sess:video_{video_id}` for this video. The `bbp:hasVideoId` should reference the YouTube video ID.
    ```turtle
    sess:video_{video_id} a bbp:VideoRecording ;
        schema:identifier "{video_id}" ;
        bbp:hasVideoId "{video_id}" .
    ```
4.  **Parliamentary Session (`bbp:ParliamentarySession`):** Create an instance for the parliamentary session itself, linking it to the video and the main Bill being debated.
    ```turtle
    sess:{video_id}_session a bbp:ParliamentarySession ;
        schema:name "Barbados Parliament Session - Video {video_id}" ;
        bbp:recordedIn sess:video_{video_id} ;
        bbp:debatesBill bbp:Bill_ChildProtectionBill .
    ```
5.  **Provenance (Time Offset for each claim):**
    For *each generated RDF triple* (e.g., `subject predicate object .`), you *must* also generate an `rdf:Statement` instance (as a blank node) that represents that triple. This `rdf:Statement` blank node *must* include the following properties:
    *   `rdf:type rdf:Statement`
    *   `rdf:subject` (pointing to the subject of the original triple)
    *   `rdf:predicate` (pointing to the predicate of the original triple)
    *   `rdf:object` (pointing to the object of the original triple)
    *   `prov:wasDerivedFrom` (pointing to a nested blank node representing the transcript segment).

    The nested blank node representing the transcript segment *must* have:
    *   `a bbp:TranscriptSegment`
    *   `bbp:fromVideo` (pointing to `sess:video_{video_id}`)
    *   `bbp:startTimeOffset` (the exact start time offset in seconds from the beginning of the *supporting line(s)* from the transcript, as `xsd:decimal`).
    *   `bbp:endTimeOffset` (the inferred end time offset in seconds for the supporting text. This should be the `startTimeOffset` of the *next* relevant line in the transcript, or a reasonable estimate if it's the very last statement extracted from the transcript. For statements spanning multiple lines, the `endTimeOffset` should be based on the start time of the line *following* the last supporting line.)

    **Example of Provenance Structure:**
    ```turtle
    bbp:Parliamentarian_UnknownH bbp:hasRole "Member for St. Peter" .
    _:stmt_UnknownH_Role_{video_id}_54 rdf:type rdf:Statement ;
        rdf:subject bbp:Parliamentarian_UnknownH ;
        rdf:predicate bbp:hasRole ;
        rdf:object "Member for St. Peter" ;
        prov:wasDerivedFrom [
            a bbp:TranscriptSegment ;
            bbp:fromVideo sess:video_{video_id} ;
            bbp:startTimeOffset "54.0"^^xsd:decimal ;
            bbp:endTimeOffset "61.0"^^xsd:decimal # Inferred from the start time of the next line (61)
        ] .

    bbp:Bill_ChildProtectionBill schema:description "designed to protect that vulnerable group in our society, our children" .
    _:stmt_ChildProtectionBill_Desc_{video_id}_459 rdf:type rdf:Statement ;
        rdf:subject bbp:Bill_ChildProtectionBill ;
        rdf:predicate schema:description ;
        rdf:object "designed to protect that vulnerable group in our society, our children" ;
        prov:wasDerivedFrom [
            a bbp:TranscriptSegment ;
            bbp:fromVideo sess:video_{video_id} ;
            bbp:startTimeOffset "459.0"^^xsd:decimal ;
            bbp:endTimeOffset "478.0"^^xsd:decimal # Inferred from the start time of the line after the supporting text ends (478)
        ] .
    ```
    *Ensure a unique blank node identifier (e.g., `_:stmt_SubjectPredicate_VideoID_StartTime`) for each `rdf:Statement` to prevent collisions when combining graphs.*

6.  **Handling `[unknown]`:** If a name or specific detail is marked as `[unknown]` (e.g., `[unknown] H`), use a placeholder like "Unknown H" for the name portion in the URI and `schema:name` property. Do not invent information.
7.  **URI Naming Convention:**
    *   **Parliamentarians:** `bbp:Parliamentarian_FirstNameLastName` (e.g., `bbp:Parliamentarian_UnknownH`, `bbp:Parliamentarian_StMichaelSouth`, `bbp:Parliamentarian_StPhilipWest`, `bbp:Parliamentarian_ChristChurchEast`, `bbp:Parliamentarian_StJohn`, `bbp:Parliamentarian_StGeorgeNorth`). Use the last name if known, otherwise the constituency, or 'Unknown' for unnamed speakers.
    *   **Constituencies:** `bbp:Constituency_Name` (e.g., `bbp:Constituency_StPeter`, `bbp:Constituency_StMichaelSouth`).
    *   **Political Parties:** `bbp:PoliticalParty_Name` (e.g., `bbp:PoliticalParty_BarbadosLabourParty`, `bbp:PoliticalParty_DemocraticLabourParty`).
    *   **Bills/Legislation:** `bbp:Bill_BillName` (e.g., `bbp:Bill_ChildProtectionBill`), `bbp:Legislation_LegislationName_Year` (e.g., `bbp:Legislation_PreventionOfCruelty_1904`).
    *   **Organizations (General):** `bbp:Org_Name` (e.g., `bbp:Org_ChildCareBoard`, `bbp:Org_UNICEF`, `bbp:Org_InterAmericanDevelopmentBank`).
    *   **Schools:** `bbp:School_Name` (e.g., `bbp:School_AllSaintsPrimary`, `bbp:School_SpringerMemorial`).
    *   **Locations:** `schema:Place_Name` (e.g., `schema:Place_Barbados`, `schema:Place_Washington`, `schema:Place_BankHall`, `schema:Place_StPeter`). Note: Prefer `schema:Place` for general locations and `bbp:Constituency` for specific parliamentary constituencies.
    *   **Concepts/Topics:** `bbp:Concept_TopicName` (e.g., `bbp:Concept_ChildProtection`, `bbp:Concept_MandatoryReporting`, `bbp:Concept_ViolenceInSchools`).
    *   **Events:** `schema:Event_EventName_Year` (e.g., `schema:Event_SpringerMemorialDrill_2023`, `schema:Event_IDBSurveyControversy_2017`).
    *   **Conventions/Treaties:** `bbp:Convention_Name` (e.g., `bbp:Convention_UNCRC`).

**Entities and Relationships to Extract (Be comprehensive for *all* speakers and entities mentioned):**

1.  **Parliamentary Session (`sess:dR-eoAEvPH4_session`):**
    *   Identify the main Bill being debated (`bbp:debatesBill`).
    *   Link to the `bbp:VideoRecording` via `bbp:recordedIn`.
2.  **Parliamentarians:**
    *   Create a `schema:Person` and `bbp:Parliamentarian` instance for each speaker.
    *   Extract their `schema:name` (e.g., "Unknown H", "Member for St. Michael South").
    *   Extract their `bbp:hasRole` (e.g., "Member for St. Peter", "Minister of Education", "Minister of Youth", "Leader of the Opposition", "Chief Parliamentary Counsel acting", "Attorney General").
    *   Extract which `bbp:Constituency` they `bbp:representsConstituency`.
    *   Extract their `org:memberOf` `bbp:PoliticalParty` if stated or strongly implied (e.g., "proud member of" for BLP).
    *   Record their `bbp:supportsBill` or `bbp:opposesBill` (the main Bill being debated).
    *   Note any `bbp:criticizes` or `bbp:commends` relationships with other parties, organizations, or individuals.
    *   Note any `bbp:discussesConcept` (key topics mentioned, e.g., "child protection", "mandatory reporting", "discrimination", "family unit", "violence in schools", "consultation", "democracy").
    *   Note any `bbp:refersToLegislation` or `bbp:citesConvention`.
    *   Note any `bbp:describesEvent`.
    *   Capture explicit familial relationships (`bbp:sonOf`, `bbp:husbandOf`, `bbp:fatherOf`, `bbp:seesAs`) where mentioned.
3.  **The Bill under Debate:**
    *   Create a `bbp:Bill` instance (identified as "Child Protection Bill").
    *   Capture its `schema:name` and stated `schema:description` or `bbp:hasPurpose`.
    *   Note its age or the age of legislation it replaces, using `bbp:replacesLegislation`.
4.  **Other Legislation/Laws Mentioned:**
    *   Create `bbp:Legislation` instances for each named law.
    *   Capture their `schema:name` and `bbp:enactedYear` (if specified, use `xsd:gYear`).
5.  **Organizations:**
    *   Create `org:Organization` (or more specific types like `bbp:PoliticalParty`, `bbp:Ministry`, `bbp:School`) instances for all named organizations (e.g., All Saints Primary, Barbados Secondary School's Entrance Examination, Child Care Board, International Labour Organisation, UNICEF, IDB, Holder Foundation, Lane Trust, Barbados Union of Teachers, Ministry of Public Service, Juvenile Justice, Bar Association, Criminal Justice Reform Committee, Parents' Association, Students' Council).
    *   Capture their `schema:name`.
    *   Note any `bbp:isLocatedIn` relationships.
    *   Note any `bbp:involvedInEvent` or `bbp:responsibleFor` relationships.
    *   Note any specific initiatives or roles mentioned (e.g., "Digital Literacy Curriculum" for Ministry of Education).
6.  **Locations:**
    *   Create `schema:Place` instances for all named places (e.g., St. Peter, St. Michael South, Christ Church East, St. Philip West, St. John, St. George North, Bank Hall, Washington, Government Hill, Mount Tabor, Station Hill).
    *   Capture `schema:name`.
7.  **Key Concepts/Topics:**
    *   Create `bbp:Concept` instances for significant themes.
    *   Use `rdfs:label` for the concept name.
8.  **Events:**
    *   Create `schema:Event` (or `bbp:ParliamentaryEvent` for specific parliamentary-related incidents) instances for specific incidents or historical occurrences (e.g., 1890s hunger uprising, Springer Memorial School drill, IDB survey).
    *   Capture `schema:name`, `schema:startDate` (if year or full date is mentioned, use `xsd:gYear` or `xsd:date`), `schema:description`, `bbp:involvedParty`.
9.  **Conventions/Treaties:**
    *   Create `bbp:Convention` instances for international agreements.
    *   Capture `schema:name` and `bbp:adoptedYear` (if mentioned, use `xsd:gYear`).

**Note on Verbosity:** Due to the requirement for reified statements for each triple to capture provenance, the generated graph will be significantly more verbose than a graph without this detailed provenance.

video_id: {video_id}

Convert the following transcript:"""

    def _get_correction_prompt(self) -> str:
        """Return the system prompt for RDF correction."""
        return """You are an expert Semantic Web engineer. The following RDF Turtle content has syntax errors that prevent it from being parsed by rdflib.

You have access to:
1. The original transcript that was being converted
2. The RDF content with syntax errors
3. The specific error message from the parser

Please fix ONLY the syntax errors while preserving all the semantic content and structure. Do not add, remove, or modify any triples - only fix syntax issues like:
- Missing periods at end of statements
- Incorrect punctuation
- Malformed URIs
- Invalid literal syntax
- Missing prefixes
- Incorrect use of blank nodes
- Quote escaping issues
- Namespace declaration problems

Use the original transcript as context to ensure that:
- Entity names and labels accurately reflect the content
- Relationships make sense in context
- URIs are properly formed and consistent
- All semantic meaning is preserved

If you see multiple errors, fix them all in one go. Do not output any additional commentary or explanations.

Return ONLY the corrected Turtle syntax with no additional commentary or explanations."""

    def load_transcript(self, file_path: str) -> str:
        """
        Load processed transcript from file.
        
        Args:
            file_path: Path to the input processed transcript file
            
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

    def clean_turtle_output(self, rdf_content: str) -> str:
        """
        Clean Turtle output by removing markdown code fences and extra formatting.
        
        Args:
            rdf_content: Raw RDF content that might contain code fences
            
        Returns:
            Clean Turtle content
        """
        # Remove markdown code fences
        lines = rdf_content.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for code fence start/end
            if stripped.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                else:
                    in_code_block = False
                continue
            
            # If we're in a code block or not in any block, include the line
            if in_code_block or not any(stripped.startswith(fence) for fence in ['```']):
                cleaned_lines.append(line)
        
        # Join back and strip extra whitespace
        cleaned = '\n'.join(cleaned_lines).strip()
        
        # Remove any remaining markdown artifacts
        cleaned = cleaned.replace('```turtle', '').replace('```ttl', '').replace('```', '')
        
        return cleaned.strip()

    def convert_to_rdf(self, transcript: str, video_id: str) -> str:
        """
        Convert transcript to RDF Turtle format.
        
        Args:
            transcript: Processed transcript content
            video_id: Video ID extracted from filename
            
        Returns:
            RDF Turtle content
        """
        conversion_prompt = self._get_conversion_prompt(video_id)
        messages = [
            SystemMessage(content=conversion_prompt),
            HumanMessage(content=transcript)
        ]
        
        try:
            response = self.llm_convert.invoke(messages)
            raw_content = response.content.strip()
            
            # Clean any markdown formatting
            cleaned_content = self.clean_turtle_output(raw_content)
            
            return cleaned_content
        except Exception as e:
            raise Exception(f"Error converting to RDF: {str(e)}")

    def correct_rdf(self, rdf_content: str, error_message: str, original_transcript: str) -> str:
        """
        Correct RDF syntax errors using original transcript as context.
        
        Args:
            rdf_content: RDF Turtle content with errors
            error_message: Error message from rdflib
            original_transcript: Original transcript for context
            
        Returns:
            Corrected RDF Turtle content
        """
        
        correction_message = f"""{self.correction_prompt}

ORIGINAL TRANSCRIPT (for context):
{original_transcript}

PARSER ERROR MESSAGE:
{error_message}

RDF CONTENT TO FIX:
{rdf_content}"""
        
        messages = [
            HumanMessage(content=correction_message)
        ]
        
        try:
            response = self.llm_correct.invoke(messages)
            corrected_content = response.content.strip()
            
            # Clean any markdown formatting that might have been added
            corrected_content = self.clean_turtle_output(corrected_content)
            
            return corrected_content
        except Exception as e:
            raise Exception(f"Error correcting RDF: {str(e)}")

    def validate_rdf(self, rdf_content: str) -> tuple[bool, Optional[str]]:
        """
        Validate RDF Turtle syntax using rdflib.
        
        Args:
            rdf_content: RDF Turtle content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            g = Graph()
            g.parse(data=rdf_content, format='turtle')
            return True, None
        except Exception as e:
            return False, str(e)

    def process_transcript(self, input_file: str):
        """
        Process transcript file and convert to validated RDF Turtle.
        
        Args:
            input_file: Path to input processed transcript file
        """
        # Generate output filename by replacing extension with .ttl
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.ttl'))
        
        # Extract video_id from filename (basename without extension)
        video_id = input_path.stem
        
        print(f"Loading processed transcript from: {input_file}")
        print(f"Using video_id: {video_id}")
        transcript = self.load_transcript(input_file)
        print(f"Loaded transcript ({len(transcript)} characters)")
        
        print("Converting transcript to RDF Turtle...")
        rdf_content = self.convert_to_rdf(transcript, video_id)
        
        print("Validating RDF syntax...")
        is_valid, error_message = self.validate_rdf(rdf_content)
        
        # Correction loop (up to 5 attempts)
        correction_attempts = 0
        max_attempts = 5
        
        while not is_valid and correction_attempts < max_attempts:
            correction_attempts += 1
            print(f"RDF validation failed (attempt {correction_attempts}/{max_attempts})")
            print(f"Error: {error_message}")
            print("Attempting to correct syntax errors with original transcript context...")
            
            try:
                # Pass the original transcript as context for correction
                rdf_content = self.correct_rdf(rdf_content, error_message, transcript)
                is_valid, error_message = self.validate_rdf(rdf_content)
                
                if is_valid:
                    print(f"RDF syntax corrected successfully after {correction_attempts} attempt(s)")
                    break
                else:
                    print(f"Correction attempt {correction_attempts} still has errors: {error_message}")
                    
            except Exception as e:
                print(f"Error during correction attempt {correction_attempts}: {e}")
                if correction_attempts == max_attempts:
                    break
        
        if not is_valid:
            print(f"Failed to correct RDF syntax after {max_attempts} attempts")
            print(f"Final error: {error_message}")
            print("Saving RDF content with syntax errors for manual review...")
            output_file = str(input_path.with_suffix('.ttl.error'))
        else:
            print("RDF syntax validation successful!")
        
        # Save the RDF content
        print(f"Saving RDF Turtle to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(rdf_content)
        
        print(f"Conversion complete! RDF saved to: {output_file}")
        
        if is_valid:
            # Final validation with statistics
            g = Graph()
            g.parse(data=rdf_content, format='turtle')
            print(f"Final RDF graph contains {len(g)} triples")
        else:
            print("Note: Output file contains syntax errors and requires manual correction")

def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python rdf_converter.py <input_transcript.txt>")
        print("\nExample:")
        print("python rdf_converter.py processed_transcript.txt")
        print("Output will be saved as: processed_transcript.ttl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize converter
        converter = RDFConverter()
        
        # Process the transcript
        converter.process_transcript(input_file)
        
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