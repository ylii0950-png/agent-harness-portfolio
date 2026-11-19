"""Safe fictional read-only MCP example, actual stdio protocol (optional SDK)."""

import asyncio
from typing import Any


def main() -> None:
    from mcp.server.fastmcp import Context, FastMCP

    server = FastMCP("Fictional Harness MCP")

    @server.tool()
    def lookup(code: str, ctx: Context[Any, Any]) -> dict[str, object]:
        """Lookup a fictional public demonstration item."""
        return {
            "code": code,
            "label": "SYNTHETIC-ALPHA",
            "meta": ctx.request_context.meta.model_dump() if ctx.request_context.meta else None,
        }

    @server.tool()
    async def slow(seconds: int = 1) -> str:
        """Read-only timeout fixture."""
        await asyncio.sleep(seconds)
        return "SYNTHETIC-SLOW"

    @server.tool()
    def fail() -> str:
        """Read-only deliberate server failure fixture."""
        raise ValueError("synthetic_failure")

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
