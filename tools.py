from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
    
search_tool = Tool(
    name="search",
    func=DuckDuckGoSearchRun().run,
    description="Search the web for information"
)

arxiv_tool = Tool(
    name="arxiv",
    func=ArxivQueryRun().run,
    description="Search the arxiv for information"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia_query = WikipediaQueryRun(api_wrapper=api_wrapper)