import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

def web_tool(query: str) -> str:
    """
    Performs a live web search to retrieve up-to-date information, facts, or guidelines.
    Returns structured results including content snippets and the source URLs.
    """
    try:
        tool_instance = TavilySearch(
            max_results=5,
            topic="general",
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            include_image_descriptions=False,
            search_depth="basic",
        )

        output = tool_instance.invoke({"query": query})
        result = output.get('results', [])

        if not result:
            print("No relevant URL found")
        
        format_result = []
        for index, item in enumerate(result, 1):
            url = item.get("url", "No url found")
            title = item.get("title", "No title found")
            content = item.get("content", "No content found")

            format_result.append(
                f"Result {index}:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {content}\n"
            )
        
        return "\n".join(format_result)

    except Exception as e:
        print(f"Error in searching {e}")
        return ""

if __name__ == "__main__":
    query = input()
    search_results = web_tool(query)
    print(search_results)