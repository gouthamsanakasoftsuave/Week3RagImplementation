"""A different program from the contract agent.

It only knows the server file. It discovers tools, then calls one.
This is the stand-in for another person's agent.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

SERVER = ROOT / "mcp_server" / "contract_repository.py"


async def main() -> None:
    async with Client(SERVER) as client:
        tools = await client.list_tools()
        names = [tool.name for tool in tools]
        print("discovered:", ", ".join(names))
        no_arg = []
        for tool in tools:
            required = (tool.input_schema or {}).get("required") or []
            if not required:
                no_arg.append(tool.name)
        target = no_arg[0] if no_arg else names[0]
        result = await client.call_tool(target, {})
        print("called", target, "->", result.data)


if __name__ == "__main__":
    asyncio.run(main())
