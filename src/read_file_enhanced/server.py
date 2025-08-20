import contextlib
import enum
import sys
import os
from collections.abc import AsyncIterator
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from markitdown import MarkItDown
import uvicorn
from openai import OpenAI
import asyncio
from typing import Optional
from typing import Annotated
import camelot
from plugin import register_converters
import toml

# Initialize FastMCP server for MarkItDown (SSE)
mcp = FastMCP("file_reader_enhanced")

class ProcessTypeEnum(str, enum.Enum):
    audio = "audio"
    video = "video"
    image_ocr = "image_ocr"
    image_description = "image_description"
    others = "others"

md = MarkItDown()
register_converters(md)

@mcp.tool()
async def read_file_enhanced(
    uri: str = Field(
        description="The URI of the file to convert to markdown. Supports http://, https://, file://, or data: schemes. Examples: 'file:///path/to/document.pdf', 'https://example.com/image.jpg'",
    ),
    llm_process_prompt: Optional[str] = Field(
        description="Specific requirements for processing images, videos and audios. Provide",
        default=None
    ),
    process_type: ProcessTypeEnum = Field(
    description="The type of the file to process. Can be 'audio', 'video', 'image_ocr', 'image_description', 'others'",
    default="others"
),
) -> str:
    """
    This tool can process various file types including office documents, pdfs, images, audios, zip files, and wikipedia pages to markdown format.
    For images, it uses AI to analyze and describe the content based on the provided requirements.
    Use this tool to read all kinds of file or web resource.
    
    Args:
        uri: The location of the file to process
        llm_process_prompt: Instructions for how to handle images, videos and audios in the content
        
    Returns:
        The file content converted to markdown format
    """
    # Execute synchronous MarkItDown conversion in thread pool to avoid blocking
    if uri.startswith("file://") and uri.endswith(".pdf"):
        markdown = mineru_pdf_to_markdown(uri.replace("file://", ""))
    else:
        markdown = await asyncio.to_thread(
            lambda: MarkItDown(
                    enable_plugins=check_plugins_enabled(),
                    llm_prompt=llm_process_prompt,
                    process_type = process_type.value,
                    **llm_config,
                ).convert_uri(uri).markdown
        )
    return markdown

def check_plugins_enabled() -> bool:
    return os.getenv("MARKITDOWN_ENABLE_PLUGINS", "false").strip().lower() in (
        "true",
        "1",
        "yes",
    )


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    sse = SseServerTransport("/messages/")
    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        event_store=None,
        json_response=True,
        stateless=True,
    )

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager."""
        async with session_manager.run():
            print("Application started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                print("Application shutting down...")

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/mcp", app=handle_streamable_http),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        lifespan=lifespan,
    )

# TODO: mineru pdf to markdown, use camelot to replace first
def mineru_pdf_to_markdown(file_path: str) -> str:

    print(f"Reading PDF file: {file_path}")
    # 读取 PDF 表格
    tables = camelot.read_pdf(file_path)
    # 取第一个表格，转为 pandas DataFrame
    df = tables[0].df
    # DataFrame 转为 Markdown 格式字符串
    md_str = df.to_markdown(index=False)

    return md_str

def load_llm_config(config_path):
    import toml
    config = toml.load(config_path)
    return {
        "llm_client": OpenAI(
            base_url=config.get("llm_base_url"),
            api_key=config.get("llm_api_key"),
        ),
        "llm_model": config.get("llm_model") or "qwen-vl-plus-latest",
        "ocr_llm_model": config.get("ocr_llm_model") or "qwen-vl-ocr-latest",
        "image_llm_model": config.get("image_llm_model") or "qwen-vl-plus-latest",
        "audio_llm_model": config.get("audio_llm_model") or "qwen-omni-turbo-latest",
        "video_llm_model": config.get("video_llm_model") or "qwen-omni-turbo-latest",
    }

# Main entry point
def main():
    import argparse

    mcp_server = mcp._mcp_server

    parser = argparse.ArgumentParser(description="Run a MarkItDown MCP server")

    parser.add_argument(
        "--http",
        action="store_true",
        help="Run the server with Streamable HTTP and SSE transport rather than STDIO (default: False)",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="(Deprecated) An alias for --http (default: False)",
    )
    parser.add_argument(
        "--host", default=None, help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on (default: 3001)"
    )
    parser.add_argument(
        "--config",
        default="assets/config.toml",
        help="Path to config.toml file (default: assets/config.toml)",
    )

    args = parser.parse_args()

    global llm_config
    llm_config = load_llm_config(args.config)

    use_http = args.http or args.sse

    if not use_http and (args.host or args.port):
        parser.error(
            "Host and port arguments are only valid when using streamable HTTP or SSE transport (see: --http)."
        )
        sys.exit(1)

    if use_http:
        starlette_app = create_starlette_app(mcp_server, debug=True)
        uvicorn.run(
            starlette_app,
            host=args.host if args.host else "127.0.0.1",
            port=args.port if args.port else 3001,
        )
    else:
        mcp.run()

if __name__ == "__main__":
    main()
