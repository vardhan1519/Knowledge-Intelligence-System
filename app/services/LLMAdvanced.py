from typing import Dict, Any, List
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END, MessagesState  # <-- Everything grouped here
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# 1. Correctly declare custom keys alongside the internal MessagesState
class CustomState(MessagesState):
    context: str
    sources: List[str]

class AutomatedLLMService:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        self.app = self._build_graph()

    def _build_graph(self):
        # Node 1: Contextualize
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history."),
            MessagesPlaceholder("messages"),  
        ])
        rewrite_chain = contextualize_q_prompt | self.llm
        
        def retrieve_context(state: CustomState) -> Dict[str, Any]:
            if len(state["messages"]) > 1:
                rewritten_q = rewrite_chain.invoke({"messages": state["messages"]}).content
            else:
                rewritten_q = state["messages"][-1].content
                
            docs = self.retriever.invoke(str(rewritten_q))
            context_text = "\n\n".join(doc.page_content for doc in docs)
            
            # Extract unique source filenames
            sources = []
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                if source and ("/" in source or "\\" in source):
                    source = os.path.basename(source)
                if source not in sources:
                    sources.append(source)
            
            return {"context": context_text, "sources": sources}

        # Node 2: Answer
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following context to answer:\n\n{context}"),
            MessagesPlaceholder("messages"),
        ])
        
        def answer_question(state: CustomState) -> Dict[str, Any]:
            qa_chain = qa_prompt | self.llm
            response = qa_chain.invoke({
                "messages": state["messages"], 
                "context": state.get("context", "")
            })
            return {"messages": [response]}

        # Build graph explicitly matching CustomState type
        workflow = StateGraph(CustomState)
        
        workflow.add_node("retrieve", retrieve_context)
        workflow.add_node("answer", answer_question)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def ask(self, session_id: str, question: str) -> Dict[str, Any]:
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        
        # Pass input as standard payload matching the graph expects
        result = self.app.invoke(
            {"messages": [HumanMessage(content=question)]},  # pyright: ignore[reportArgumentType]
            config=config
        )
        
        return {
            "answer": result["messages"][-1].content,
            "sources": result.get("sources", [])
        }


# --- TEST EXECUTION BLOCK ---
if __name__ == "__main__":
    import os
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from dotenv import load_dotenv
    
    load_dotenv()

    print("\n--- Initializing Mock Data and Services ---")
    
    sample_documents = [
        Document(page_content="The Knowledge Intelligence System project is built by Vardhan."),
        Document(page_content="LangChain combined with Memory allows a RAG pipeline to remember context."),
        Document(page_content="The system uses a virtual environment managed by uv to run services safely.")
    ]
    
    print("Building temporary vector store...")
    embeddings = OpenAIEmbeddings()
    mock_vector_store = Chroma.from_documents(sample_documents, embeddings)
    
    print("Initializing AutomatedLLMService...")
    service = AutomatedLLMService(vector_store=mock_vector_store)
    
    session_id = "test_user_session_99"
    print(f"\n--- Starting Conversation (Session: {session_id}) ---")
    
    q1 = "Who built the Knowledge Intelligence System?"
    print(f"User: {q1}")
    a1 = service.ask(session_id=session_id, question=q1)
    print(f"AI: {a1}\n")
    
    q2 = "What tools does he use to manage its virtual environment?"
    print(f"User: {q2}")
    a2 = service.ask(session_id=session_id, question=q2)
    print(f"AI: {a2}\n")
    
    print("--- Test Complete ---")