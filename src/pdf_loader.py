"""
PDF Loader module for the DSA RAG Chatbot.
Extracts text and page metadata from DSA PDF files using PyPDF.
"""

from pathlib import Path
from typing import List, Optional
import pypdf
from langchain_core.documents import Document


def load_single_pdf(file_path: Path) -> List[Document]:
    """
    Extract text and metadata from a single PDF document.
    
    Args:
        file_path: Path to the target PDF file.
        
    Returns:
        List of LangChain Document objects with page text and metadata.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")
        
    documents: List[Document] = []
    
    try:
        reader = pypdf.PdfReader(str(file_path))
        total_pages = len(reader.pages)
        
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            # Clean up trailing and leading spaces
            cleaned_text = page_text.strip()
            
            # Skip completely empty pages
            if not cleaned_text:
                continue
                
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path.resolve()),
                "page": page_idx + 1,  # 1-indexed for user readability
                "total_pages": total_pages,
            }
            documents.append(Document(page_content=cleaned_text, metadata=metadata))
            
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF '{file_path.name}': {exc}") from exc
        
    return documents


def get_pdf_paths(directory: Path) -> List[Path]:
    """
    Find all .pdf files in the directory without duplicates across case-insensitive filesystems.
    """
    if not directory.exists() or not directory.is_dir():
        return []
    
    unique_files = {p.resolve(): p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"}
    return sorted(list(unique_files.values()), key=lambda p: p.name.lower())


def load_all_pdfs(directory: Path) -> List[Document]:
    """
    Scan a directory for all .pdf files and extract their contents.
    
    Args:
        directory: Directory containing PDF files.
        
    Returns:
        List of LangChain Document objects across all discovered PDFs.
    """
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {directory}")
        
    pdf_files = get_pdf_paths(directory)
    
    if not pdf_files:
        return []
        
    all_documents: List[Document] = []
    for pdf_file in pdf_files:
        docs = load_single_pdf(pdf_file)
        all_documents.extend(docs)
        
    return all_documents


def list_available_pdfs(directory: Path) -> List[dict]:
    """
    Return a list of available PDF filenames with file sizes and page counts.
    Useful for UI displays and diagnostic checks.
    """
    if not directory.exists() or not directory.is_dir():
        return []
        
    pdf_files = get_pdf_paths(directory)
    details = []
    
    for pdf_file in pdf_files:
        try:
            reader = pypdf.PdfReader(str(pdf_file))
            details.append({
                "name": pdf_file.name,
                "size_kb": round(pdf_file.stat().st_size / 1024, 2),
                "pages": len(reader.pages),
            })
        except Exception:
            details.append({
                "name": pdf_file.name,
                "size_kb": round(pdf_file.stat().st_size / 1024, 2),
                "pages": 0,
            })
            
    return details
