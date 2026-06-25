# Modern core layout imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# IMPORT FROM LANGCHAIN_CLASSIC HERE
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from app.config import Config
from typing import cast
class LLMService:
    def __init__(self, vector_store):
        # 1. Initialize modern OpenAI Client
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            api_key=lambda: Config.OPENAI_API_KEY or ""
        )
        
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        self.session_store = {}
        self.rag_chain = self._build_conversational_rag()

    def _get_session_history(self, session_id: str):
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def _build_conversational_rag(self):
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # QA prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        base_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return RunnableWithMessageHistory(
            base_rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def ask(self, session_id: str, question: str) -> str:
        """Call this method to query the RAG system with context memory."""
        config = cast(RunnableConfig, {"configurable": {"session_id": session_id}})
        response = self.rag_chain.invoke({"input": question}, config=config)
        return response["answer"]
    

# --- TEST EXECUTION BLOCK ---
if __name__ == "__main__":
    import os
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

    print("\n--- Initializing Mock Data and Services ---")
    
    # 1. Create a quick in-memory vector database with sample data
    # Make sure your OPENAI_API_KEY is available in your environment or config
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY or ""
    
    sample_documents = [
        Document(page_content="The Knowledge Intelligence System project is built by Vardhan."),
        Document(page_content="LangChain combined with Memory allows a RAG pipeline to remember context."),
        Document(page_content="The system uses a virtual environment managed by uv to run services safely.")
    ]
    
    print("Building temporary vector store...")
    embeddings = OpenAIEmbeddings(api_key=lambda: os.environ["OPENAI_API_KEY"])
    mock_vector_store = Chroma.from_documents(sample_documents, embeddings)
    
    # 2. Instantiate your service
    print("Initializing LLMService...")
    service = LLMService(vector_store=mock_vector_store)
    
    # 3. Simulate a conversation session
    session_id = "test_user_session_99"
    print(f"\n--- Starting Conversation (Session: {session_id}) ---")
    
    # Turn 1: Direct question about data in the vector store
    q1 = "Who built the Knowledge Intelligence System?"
    print(f"User: {q1}")
    a1 = service.ask(session_id=session_id, question=q1)
    print(f"AI: {a1}\n")
    
    # Turn 2: Follow-up question relying COMPLETELY on memory/context
    q2 = "What tools does he use to manage its virtual environment?"
    print(f"User: {q2}")
    a2 = service.ask(session_id=session_id, question=q2)
    print(f"AI: {a2}\n")
    
    print("--- Test Complete ---") 
    