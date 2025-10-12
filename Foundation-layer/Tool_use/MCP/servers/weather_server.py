from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    # This is a mock implementation - in production, we would call a real weather API
    return "Hot as hell"

if __name__ == "__main__":
    mcp.run(transport="stdio")