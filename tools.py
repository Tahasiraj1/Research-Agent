from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_file(data: str, filename: str = "research_paper.txt"):
    """Saves the research paper to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"--- Reasearch Paper ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    
    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text)
    
    return f"Data saved to {filename}"

save_tool = Tool(
    name="save",
    func=save_to_file,
    description="Saves structured research data to a text file."
)
    
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_query = WikipediaQueryRun(api_wrapper=api_wrapper)