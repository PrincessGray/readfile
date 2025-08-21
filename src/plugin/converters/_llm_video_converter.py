import base64
import mimetypes
import locale
import subprocess
import json
from typing import Any, BinaryIO, Union


from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    StreamInfo,
)

ACCEPTED_MIME_TYPE_PREFIXES = [
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
]
ACCEPTED_FILE_EXTENSIONS = [
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
]

def exiftool_metadata(
    file_stream: BinaryIO,
    *,
    exiftool_path: Union[str, None],
) -> Any:
    if not exiftool_path:
        return {}
    cur_pos = file_stream.tell()
    try:
        output = subprocess.run(
            [exiftool_path, "-json", "-"],
            input=file_stream.read(),
            capture_output=True,
            text=False,
        ).stdout
        return json.loads(
            output.decode(locale.getpreferredencoding(False)),
        )[0]
    finally:
        file_stream.seek(cur_pos)

class LLMVideoConverter(DocumentConverter):
    """
    Converts video files to markdown via LLM analysis.
    """
    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> bool:
        return kwargs.get("process_type") == "video"
        
    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        md_content = ""

        # Add metadata
        metadata = exiftool_metadata(
            file_stream, exiftool_path=kwargs.get("exiftool_path")
        )
        if metadata:
            for f in [
                "Title", "Artist", "Author", "Director", "Album", "Genre",
                "Track", "DateTimeOriginal", "CreateDate", "Duration",
                "VideoFrameRate", "ImageWidth", "ImageHeight"
            ]:
                if f in metadata:
                    md_content += f"{f}: {metadata[f]}\n"

        # Check video format
        video_format = None
        if stream_info.extension:
            video_format = stream_info.extension.lstrip(".")
        elif stream_info.mimetype:
            if "mp4" in stream_info.mimetype:
                video_format = "mp4"
            elif "mov" in stream_info.mimetype:
                video_format = "mov"
            elif "avi" in stream_info.mimetype:
                video_format = "avi"
            elif "mkv" in stream_info.mimetype:
                video_format = "mkv"

        # LLM analysis
        llm_client = kwargs.get("llm_client")
        llm_model = kwargs.get("llm_model")
        llm_description = self._get_llm_description(
            file_stream,
            stream_info,
            client=llm_client,
            model=llm_model,
            prompt=kwargs.get("llm_prompt"),
        )

        md_content += "\n# Description:\n" + llm_description.strip() + "\n"

        return DocumentConverterResult(markdown=md_content.strip())

    def _get_llm_description(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        *,
        client,
        model,
        prompt=None,
    ) -> Union[None, str]:
        if prompt is None or prompt.strip() == "":
            prompt = "Summarize the video content."

        # 获取视频格式
        video_format = None
        if stream_info.extension:
            video_format = stream_info.extension.lstrip(".")
        elif stream_info.mimetype:
            if "mp4" in stream_info.mimetype:
                video_format = "mp4"
            elif "mov" in stream_info.mimetype:
                video_format = "mov"
            elif "avi" in stream_info.mimetype:
                video_format = "avi"
            elif "mkv" in stream_info.mimetype:
                video_format = "mkv"

        # 读视频并编码为 base64
        cur_pos = file_stream.tell()
        try:
            base64_video = base64.b64encode(file_stream.read()).decode("utf-8")
        except Exception as e:
            return None
        finally:
            file_stream.seek(cur_pos)

        data_uri = f"data:;base64,{base64_video}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_video",
                        "input_video": {
                            "data": data_uri,
                            # "format": video_format or "mp4",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        result = ""
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    result += delta.content
        return result.strip()
