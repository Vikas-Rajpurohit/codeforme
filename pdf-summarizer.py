import os
import re
import PyPDF2
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import json

class PDFSummarizer:
    def __init__(self, llm_api_wrapper, chunk_size=2000, overlap=200, max_workers=4):
        """
        Initialize the PDF summarizer with the LLM API wrapper.
        
        Args:
            llm_api_wrapper: A callable that takes a prompt string and returns a response string
            chunk_size: The approximate size of chunks to process
            overlap: The amount of overlap between chunks
            max_workers: Maximum number of parallel workers for processing
        """
        self.llm_api = llm_api_wrapper
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_workers = max_workers
        
    def extract_text_and_metadata(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extract text and metadata from a PDF file."""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            metadata = {
                'title': reader.metadata.get('/Title', 'Unknown'),
                'author': reader.metadata.get('/Author', 'Unknown'),
                'subject': reader.metadata.get('/Subject', ''),
                'keywords': reader.metadata.get('/Keywords', ''),
                'creation_date': reader.metadata.get('/CreationDate', ''),
                'modification_date': reader.metadata.get('/ModDate', ''),
                'page_count': len(reader.pages)
            }
            
            # Extract text with page numbers
            text_with_page = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text_with_page.append(f"[Page {i+1}] {page_text}")
            
            full_text = "\n\n".join(text_with_page)
            
            return full_text, metadata
    
    def detect_section_boundaries(self, text: str) -> List[int]:
        """
        Detect section boundaries in the text based on common section header patterns.
        Returns list of indices where sections start.
        """
        # Patterns for financial document section headers
        patterns = [
            r'\n\s*(?:SECTION|Section)\s+\d+[.:]\s+\w+',  # Section 1: Introduction
            r'\n\s*\d+(?:\.\d+)*\s+[A-Z][a-zA-Z\s]+',     # 1.2 Important Terms
            r'\n\s*[IVXLCDM]+\.\s+[A-Z][a-zA-Z\s]+',      # IV. Risk Factors
            r'\n\s*[A-Z][A-Z\s]{2,}(?:\s*\([^)]+\))?\s*$', # RISK FACTORS (all caps)
            r'\n\s*(?:ARTICLE|Article)\s+[IVXLCDM]+[.:]\s+\w+', # Article IV: Terms
        ]
        
        # Find all matches
        boundaries = [0]  # Always include the start of the document
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())
                
        # Sort and deduplicate
        boundaries = sorted(set(boundaries))
        return boundaries
        
    def create_semantic_chunks(self, text: str) -> List[str]:
        """
        Split text into semantic chunks based on detected section boundaries.
        Falls back to size-based chunking if no clear sections are found.
        """
        boundaries = self.detect_section_boundaries(text)
        
        # If we have a reasonable number of sections, use them
        if len(boundaries) > 1 and len(boundaries) < 30:  # Avoid over-chunking
            chunks = []
            for i in range(len(boundaries) - 1):
                chunk = text[boundaries[i]:boundaries[i+1]]
                chunks.append(chunk)
            # Add the last section
            chunks.append(text[boundaries[-1]:])
            
            # Further split any chunks that are too big
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > self.chunk_size * 1.5:
                    # Split by paragraphs if chunk is too big
                    paragraphs = chunk.split('\n\n')
                    current_chunk = ""
                    for para in paragraphs:
                        if len(current_chunk) + len(para) < self.chunk_size:
                            current_chunk += para + "\n\n"
                        else:
                            if current_chunk:
                                final_chunks.append(current_chunk)
                            current_chunk = para + "\n\n"
                    if current_chunk:
                        final_chunks.append(current_chunk)
                else:
                    final_chunks.append(chunk)
            return final_chunks
            
        # Fallback: Split by approximate size with paragraph boundaries
        else:
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < self.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    chunks.append(current_chunk)
                    current_chunk = para + "\n\n"
                    
            if current_chunk:  # Add the last chunk
                chunks.append(current_chunk)
                
            return chunks
    
    def summarize_chunk(self, chunk: str, metadata: Dict, is_financial: bool = True) -> str:
        """Summarize a single chunk using the LLM API."""
        # Create a prompt that emphasizes financial terms and policies
        if is_financial:
            prompt = f"""Summarize the following section of a financial document. 
Focus on preserving:
1. Important financial terms, metrics, and numbers
2. Policy statements and legal obligations
3. Risk factors and warnings
4. Key dates, timelines, and deadlines

Document Title: {metadata.get('title', 'Unknown')}
Document Section:
{chunk}

Provide a concise yet comprehensive summary that maintains all critical financial information and regulatory details:"""
        else:
            prompt = f"""Summarize the following section of a document.
Document Title: {metadata.get('title', 'Unknown')}
Document Section:
{chunk}

Provide a concise yet comprehensive summary:"""
        
        # Call the LLM API
        response = self.llm_api(prompt)
        return response
    
    def summarize_summaries(self, summaries: List[str], metadata: Dict, is_financial: bool = True) -> str:
        """Combine the individual summaries into a coherent final summary."""
        all_summaries = "\n\n".join([f"Section {i+1} Summary:\n{summary}" for i, summary in enumerate(summaries)])
        
        if is_financial:
            prompt = f"""You are creating a final comprehensive summary of a financial document.
Below are summaries of each section of the document.

Document Title: {metadata.get('title', 'Unknown')}
Document Author: {metadata.get('author', 'Unknown')}
Document Keywords: {metadata.get('keywords', '')}
Total Pages: {metadata.get('page_count', 'Unknown')}

{all_summaries}

Create a well-structured final summary of the entire document that:
1. Preserves all critical financial information, terms, and metrics
2. Maintains all policy statements and compliance requirements
3. Highlights key risk factors and warnings
4. Organizes information logically by sections
5. Includes important dates, deadlines, and timelines

The summary should be comprehensive enough to serve as a reliable reference to the original document:"""
        else:
            prompt = f"""You are creating a final comprehensive summary of a document.
Below are summaries of each section of the document.

Document Title: {metadata.get('title', 'Unknown')}
Document Author: {metadata.get('author', 'Unknown')}
Document Keywords: {metadata.get('keywords', '')}
Total Pages: {metadata.get('page_count', 'Unknown')}

{all_summaries}

Create a well-structured final summary of the entire document that captures all the key information and maintains the logical flow:"""
        
        # Call the LLM API
        response = self.llm_api(prompt)
        return response

    def summarize_pdf(self, pdf_path: str, is_financial: bool = True) -> Dict:
        """
        Summarize a PDF using a map-reduce approach.
        
        Args:
            pdf_path: Path to the PDF file
            is_financial: Whether to use financial-specific prompting
            
        Returns:
            Dict containing the summary and metadata
        """
        # Extract text and metadata
        text, metadata = self.extract_text_and_metadata(pdf_path)
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(text)
        
        # Map phase: Summarize each chunk in parallel
        chunk_summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.summarize_chunk, chunk, metadata, is_financial): i 
                for i, chunk in enumerate(chunks)
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    summary = future.result()
                    chunk_summaries.append((chunk_idx, summary))
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                    chunk_summaries.append((chunk_idx, f"[Error summarizing this section: {e}]"))
        
        # Sort summaries by original order
        chunk_summaries.sort(key=lambda x: x[0])
        summaries = [summary for _, summary in chunk_summaries]
        
        # Reduce phase: Combine summaries
        final_summary = self.summarize_summaries(summaries, metadata, is_financial)
        
        # Return results
        return {
            "metadata": metadata,
            "chunk_count": len(chunks),
            "section_summaries": summaries,
            "final_summary": final_summary
        }


# Example usage with a simple LLM API wrapper
def example_llm_api(prompt: str) -> str:
    """
    Replace this with your actual LLM API call.
    This is just a placeholder function.
    """
    # In a real implementation, you would call your LLM API here
    return f"This is where the LLM response would be for: {prompt[:50]}..."

# Usage example
if __name__ == "__main__":
    # Create the summarizer with your LLM API function
    summarizer = PDFSummarizer(example_llm_api)
    
    # Summarize a PDF
    result = summarizer.summarize_pdf("example_financial_report.pdf")
    
    # Output the result
    print(f"Document: {result['metadata']['title']}")
    print(f"Total sections processed: {result['chunk_count']}")
    print("\nFinal Summary:")
    print(result['final_summary'])
    
    # Save the full result to a JSON file
    with open("summary_result.json", "w") as f:
        json.dump(result, f, indent=2)
