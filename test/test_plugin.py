import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from plugin import register_converters
from markitdown import MarkItDown
from openai import OpenAI

md = MarkItDown()
register_converters(md)
print(md.convert_uri("file:///home/gongziqin/mcp-dockers/feature-server/data/GAIA/2023/validation/6359a0b1-8f7b-499b-9336-840f9ab90688.png", process_type="image_description",llm_client=OpenAI(
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        api_key="sk-0237d0b3d163451586a7e01394d995e5",
                    )
                    ,llm_model="qwen-vl-plus-latest",
                    llm_prompt="Only output the description of the image, no other text.",
))

