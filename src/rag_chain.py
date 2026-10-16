"""
RAG Chain module for the DSA RAG Chatbot.
Constructs the LangChain LCEL pipeline with Google Gemini and grounded prompts.
"""

from typing import Dict, Any, List, Optional
import config
from src.retriever import format_docs_for_prompt, extract_source_citations

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None
    StrOutputParser = None
    RunnablePassthrough = None


DSA_SYSTEM_PROMPT = """You are an expert Data Structures and Algorithms (DSA) tutor.
Your role is to help students learn algorithms and data structures clearly, rigorously, and accurately.

CRITICAL INSTRUCTIONS:
1. Knowledge Grounding:
   - Answer the question relying ONLY on the information provided in the Context below.
   - Do NOT invent facts, methods, or algorithms that are not grounded in the context.
   - If the context does not contain sufficient information to answer the question thoroughly, clearly state:
     "The retrieved DSA documents do not contain sufficient information to answer this question."
   - Do not make unsupported guesses.

2. Structure and Clarity:
   - When explaining an algorithm or data structure:
     a. Provide an intuitive overview.
     b. Give a clear step-by-step walkthrough of the logic.
     c. Provide idiomatic, clean, and well-commented Python code when relevant.
     d. Explicitly state the Time Complexity (best, average, worst) and Space Complexity using Big-O notation, with a brief justification.

3. Citation and Attribution:
   - Cite the relevant source document name(s) and page number(s) present in the context headers.

---
Context:
{context}
---
"""


def get_llm(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> "ChatGoogleGenerativeAI":
    """
    Instantiate and return the Google Gemini Chat model.
    """
    if ChatGoogleGenerativeAI is None:
        raise ImportError(
            "langchain-google-genai is not installed. Run `pip install langchain-google-genai`."
        )

    key = config.GEMINI_API_KEY if api_key is None else api_key
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured in your .env file.")

    model = model_name or config.LLM_MODEL
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=temp,
    )


def create_dsa_prompt_template() -> "ChatPromptTemplate":
    """
    Create the chat prompt template enforcing strict context grounding for DSA.
    """
    return ChatPromptTemplate.from_messages([
        ("system", DSA_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])


def execute_rag_query(
    question: str,
    retriever,
    llm: Optional["ChatGoogleGenerativeAI"] = None,
) -> Dict[str, Any]:
    """
    Execute a full RAG query:
    1. Retrieve relevant chunks using the retriever.
    2. Check if any context documents were retrieved.
    3. Format context and pass to Gemini LLM.
    4. Return the answer along with extracted source citations.
    
    Args:
        question: User query string.
        retriever: LangChain VectorStoreRetriever.
        llm: Optional ChatGoogleGenerativeAI instance.
        
    Returns:
        dict: {
            "question": str,
            "answer": str,
            "sources": List[Dict[str, Any]],
            "raw_documents": List[Document],
            "has_context": bool
        }
    """
    if llm is None:
        llm = get_llm()

    # Step 1: Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "question": question,
            "answer": (
                "The retrieved DSA documents do not contain sufficient information to answer this question. "
                "No matching content was found in the indexed knowledge base."
            ),
            "sources": [],
            "raw_documents": [],
            "has_context": False,
        }

    # Step 2: Format context for prompt
    context_text = format_docs_for_prompt(retrieved_docs)
    sources = extract_source_citations(retrieved_docs)

    # Step 3: Run LLM generation
    prompt = create_dsa_prompt_template()
    chain = prompt | llm | StrOutputParser()
    
    response_text = chain.invoke({
        "context": context_text,
        "question": question,
    })

    return {
        "question": question,
        "answer": response_text,
        "sources": sources,
        "raw_documents": retrieved_docs,
        "has_context": True,
    }
