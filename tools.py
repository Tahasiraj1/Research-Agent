from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
import requests
    
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

def search_semantic_scholar(query, max_results: int = 5):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,authors,url,abstract&limit={max_results}"
    response = requests.get(url)
    return response.json()

semantic_scholar_tool = Tool(
    name="semantic_scholar",
    func=search_semantic_scholar,
    description="Fetches research papers from Semantic Scholar based on a given topic."
)
