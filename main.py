import os 
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wikipedia_query, arxiv_tool, semantic_scholar_tool
import streamlit as st

# API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Response Model
class ResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# LLM & Parser
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=gemini_api_key)
parser = PydanticOutputParser(pydantic_object=ResponseModel)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a highly knowledgeable research assistant specializing in generating comprehensive, detailed, and well-structured research papers.  
            Your response should be at least **500 words**, including **key findings, methodologies, references, and citations**.  
            Use all necessary tools and provide an in-depth analysis. Do NOT summarize excessively.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Tools & Agent
tools = [search_tool, wikipedia_query, arxiv_tool, semantic_scholar_tool]
agents = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

# Executor
agent_executor = AgentExecutor(agent=agents, tools=tools, verbose=True)

# Streamlit UI
st.set_page_config(page_title="Research Agent", page_icon="🧠")
st.title("🧠 Research Agent")

query = st.text_input("What can I help you research? ")

if st.button("Research", use_container_width=True):
    with st.spinner("Processing..."):
        raw_response = agent_executor.invoke({"query": query})
        
        print("Raw Response:", raw_response)  # Debugging
        
        try:
            structured_response = parser.parse(raw_response.get("output", ""))
            st.subheader(structured_response.topic)
            st.write(structured_response.summary)
            with st.expander("Sources"):
                for source in structured_response.sources:
                    st.write(source)
            with st.expander("Tools Used"):
                for tool in structured_response.tools_used:
                    st.write(tool)
        except Exception as e:
            st.error("Error parsing response:", e)
            st.error("Raw Response - ", raw_response)
            
        if structured_response:
            st.download_button(
                "Download Paper",
                structured_response.summary,
                file_name=f"{structured_response.topic}.txt",
                mime="text/plain",
                use_container_width=True,
            )
