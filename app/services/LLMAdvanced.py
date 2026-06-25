from typing import Dict, List, Any
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Define the Modern State Type
class State(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    context: str
    answer: str

class ModernLLMService:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Build the graph
        self.app = self._build_graph()

    def _build_graph(self):
        # Node 1: Contextualize the question using pure LCEL
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question... formulate a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        # A simple modern chain to rewrite the question
        rewrite_chain = contextualize_q_prompt | self.llm
        
        def retrieve_context(state: State) -> Dict[str, Any]:
            # Use history to rewrite query if history exists, else search directly
            if state["chat_history"]:
                rewritten_q = rewrite_chain.invoke(state).content
            else:
                rewritten_q = state["input"]
                
            docs = self.retriever.invoke(rewritten_q)
            # Combine the document texts into a single context string
            context_text = "\n\n".join(doc.page_content for doc in docs)
            return {"context": context_text}

        # Node 2: Answer the question
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following context to answer:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = qa_prompt | self.llm

        def answer_question(state: State) -> Dict[str, Any]:
            response = qa_chain.invoke(state)
            return {"answer": response.content}

        # 2. Build the Graph Workflow
        workflow = StateGraph(State)
        workflow.add_node("retrieve", retrieve_context)
        workflow.add_node("answer", answer_question)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        
        # 3. Add the Checkpointer (Modern Memory)
        # This automatically handles saving/loading messages based on thread_id
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def ask(self, session_id: str, question: str) -> str:
        config = {"configurable": {"thread_id": session_id}}
        
        # Input messages are passed to the graph
        # LangGraph automatically appends messages to chat_history if configured, 
        # but here we manually pass the input state.
        result = self.app.invoke({"input": question}, config=config)
        
        # Manually updating history state is crystal clear now
        # No hidden magic under the hood
        current_history = self.app.get_state(config).values.get("chat_history", [])
        from langchain_core.messages import HumanMessage, AIMessage
        updated_history = current_history + [HumanMessage(content=question), AIMessage(content=result["answer"])]
        self.app.update_state(config, {"chat_history": updated_history})
        
        return result["answer"]