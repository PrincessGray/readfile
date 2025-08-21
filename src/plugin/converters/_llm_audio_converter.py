from typing import Any, BinaryIO, Union
import json
import subprocess
import locale
import base64
import mimetypes

from markitdown import (
    DocumentConverter,
    DocumentConverterResult,
    StreamInfo,
)

ACCEPTED_MIME_TYPE_PREFIXES = [
    "audio/x-wav",
    "audio/mpeg",
    "video/mp4",
]

ACCEPTED_FILE_EXTENSIONS = [
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
]

def exiftool_metadata(
    file_stream: BinaryIO,
    *,
    exiftool_path: Union[str, None],
) -> Any:  # Need a better type for json data
    # Nothing to do
    if not exiftool_path:
        return {}

    # Run exiftool
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


class LLMAudioConverter(DocumentConverter):
    """
    Converts audio files to markdown via speech transcription (if `speech_recognition` is installed).
    """

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> bool:
        if kwargs.get("process_type") != "audio":
            return False
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in ACCEPTED_FILE_EXTENSIONS:
            return True

        for prefix in ACCEPTED_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False
    
    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        md_content = ""

        # Add metadata
        metadata = exiftool_metadata(
            file_stream, exiftool_path=kwargs.get("exiftool_path")
        )
        if metadata:
            for f in [
                "Title",
                "Artist",
                "Author",
                "Band",
                "Album",
                "Genre",
                "Track",
                "DateTimeOriginal",
                "CreateDate",
                # "Duration", -- Wrong values when read from memory
                "NumChannels",
                "SampleRate",
                "AvgBytesPerSec",
                "BitsPerSample",
            ]:
                if f in metadata:
                    md_content += f"{f}: {metadata[f]}\n"

        # Figure out the audio format for transcription
        if stream_info.extension == ".wav" or stream_info.mimetype == "audio/x-wav":
            audio_format = "wav"
        elif stream_info.extension == ".mp3" or stream_info.mimetype == "audio/mpeg":
            audio_format = "mp3"
        elif (
            stream_info.extension in [".mp4", ".m4a"]
            or stream_info.mimetype == "video/mp4"
        ):
            audio_format = "mp4"
        else:
            audio_format = None

        # Transcribe
        if audio_format:
            llm_client = kwargs.get("llm_client")
            llm_model = kwargs.get("llm_model")
            llm_description = self._get_llm_description(
                file_stream,
                stream_info,
                client=llm_client,
                model=llm_model,
                prompt=kwargs.get("llm_prompt"),
            )

            md_content += "\n# Audio Transcription:\n" + llm_description.strip() + "\n"
        # Return the result
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
            prompt = "Summarize the audio content."

        # 获取音频格式
        audio_format = None
        if stream_info.extension:
            audio_format = stream_info.extension.lstrip(".")
        elif stream_info.mimetype:
            if "wav" in stream_info.mimetype:
                audio_format = "wav"
            elif "mp3" in stream_info.mimetype:
                audio_format = "mp3"
            elif "mp4" in stream_info.mimetype or "m4a" in stream_info.mimetype:
                audio_format = "mp4"

        # 读音频并编码为 base64
        cur_pos = file_stream.tell()
        try:
            base64_audio = base64.b64encode(file_stream.read()).decode("utf-8")
        except Exception as e:
            return None
        finally:
            file_stream.seek(cur_pos)

        # 构造 data URI
        data_uri = f"data:;base64,{base64_audio}"

        # 构造消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": data_uri,
                            "format": audio_format or "mp3",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # 调用 Qwen API
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
            # 可选：处理 usage 信息
        return result.strip()
