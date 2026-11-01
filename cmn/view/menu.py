"""App menu: the pages shown in the sidebar and on the home page. See MENU.md."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MenuPage:
    path: str
    title: str
    icon: str
    summary: str
    featured: bool = False


MENU_SECTIONS: dict[str, list[MenuPage]] = {
    "Chat & agents": [
        MenuPage("pages/3_5_13_converse_demo.py", "Chat with any model", ":material/chat:",
                 "Multi-turn chat with Claude, GPT, Llama, Nova and more through the Converse streaming API. "
                 "Attach images and documents.", featured=True),
        MenuPage("pages/3_7_5_converse_tool_v3.py", "Tool-using assistant", ":material/construction:",
                 "An agent loop that calls tools (sales analytics, calculator, Wikipedia, PDF and PowerPoint "
                 "generation) and renders the results as charts, tables and downloads.", featured=True),
        MenuPage("pages/3_7_4_converse_tool_demo.py", "Tool use, step by step", ":material/build:",
                 "The first, single-file version of the tool loop. Easier to read if you want to see how "
                 "tool use works."),
        MenuPage("pages/3_8_01_bedrock_mantle.py", "OpenAI-compatible API", ":material/api:",
                 "GPT and Claude through Bedrock's OpenAI-compatible endpoint (Responses API), with tool "
                 "calling and web search.", featured=True),
        MenuPage("pages/3_9_01_bedrock_harness_chat.py", "AgentCore harness chat", ":material/smart_toy:",
                 "Chat with an agent hosted on an Amazon Bedrock AgentCore harness."),
        MenuPage("pages/3_4_chat_demo.py", "Chat with native requests", ":material/record_voice_over:",
                 "Streaming chat using each provider's own request format (InvokeModel), with Amazon Polly "
                 "read-aloud."),
    ],
    "Search & knowledge (RAG)": [
        MenuPage("pages/14_1_1_session_rag.py", "Chat with your documents", ":material/description:",
                 "Upload documents, index them with Titan embeddings in a local FAISS store, and ask "
                 "questions answered from them.", featured=True),
        MenuPage("pages/3_9_50_bedrock_managed_kb.py", "Managed knowledge bases", ":material/library_books:",
                 "Query Bedrock Knowledge Bases with Retrieve and RetrieveAndGenerate."),
        MenuPage("pages/2_s3_vector.py", "S3 Vectors store", ":material/database:",
                 "Add and search documents in an Amazon S3 Vectors bucket with Nova multimodal embeddings."),
        MenuPage("pages/2_s3_vector_kb_3.py", "Multi-tenant knowledge base", ":material/groups:",
                 "One knowledge base on S3 Vectors shared by several tenants, kept apart with metadata filters."),
        MenuPage("pages/6_2_4_filekb_demo.py", "Ask about a file", ":material/upload_file:",
                 "Ask questions about an uploaded file with RetrieveAndGenerate; images go to Converse."),
    ],
    "Images & vision": [
        MenuPage("pages/9_1_8_image_generator.py", "Generate images", ":material/image:",
                 "Text-to-image with Stability AI: SD3.5 Large, Stable Image Core and Ultra.", featured=True),
        MenuPage("pages/9_5_7_image_stability_edit.py", "Edit images", ":material/auto_fix_high:",
                 "Remove backgrounds, erase objects, inpaint, outpaint, search-and-replace and recolor.",
                 featured=True),
        MenuPage("pages/9_5_7_image_stability_control.py", "Sketch & style control", ":material/brush:",
                 "Turn sketches into images, keep an image's structure, or transfer a style."),
        MenuPage("pages/9_5_7_image_stability_upscale.py", "Upscale images", ":material/high_quality:",
                 "Fast, conservative and creative upscaling."),
        MenuPage("pages/8_1_6_image_query.py", "Ask about an image", ":material/image_search:",
                 "Question an image with Claude, alongside Amazon Rekognition labels and dominant colors."),
        MenuPage("pages/8_5_7_image_color_compare.py", "Compare image colors", ":material/palette:",
                 "Compare two images' colors, brightness and sharpness with Rekognition and Claude."),
        MenuPage("pages/99_01_multimodal_search.py", "Multimodal product search", ":material/manage_search:",
                 "Search a product catalog by text or by image (OpenSearch)."),
    ],
    "Prompts & tools": [
        MenuPage("pages/1_count_token.py", "Token counter", ":material/tag:",
                 "Count a prompt's tokens before sending it, with the free CountTokens API."),
        MenuPage("pages/2_prompt_template.py", "Prompt manager", ":material/edit_note:",
                 "Browse and run prompts saved in Bedrock Prompt Management."),
        MenuPage("pages/13_1_1_resformat.py", "Structured output", ":material/data_object:",
                 "Turn vehicle telemetry into a driver risk report, comparing ways to get reliable JSON "
                 "(prompting, schema-enforced tool use)."),
        MenuPage("pages/12_1_1_dataset.py", "Table row selection", ":material/table_rows:",
                 "Streamlit dataframe row selection. UI only, no AWS calls."),
    ],
}
