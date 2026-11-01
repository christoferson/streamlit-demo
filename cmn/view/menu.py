"""App menu: the pages shown in the sidebar and on the home page. See MENU.md."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MenuPage:
    path: str
    title: str
    icon: str
    summary: str                       # one or two sentences, shown on cards and in the list
    details: tuple[str, ...] = ()      # "what you can do" bullets
    services: tuple[str, ...] = ()     # AWS services / APIs, shown as badges
    try_this: str = ""                 # a one-minute thing to try
    needs: str = ""                    # setup beyond Bedrock model access; empty if none
    differs: str = ""                  # how it differs from similar-looking demos
    featured: bool = False


MENU_SECTIONS: dict[str, list[MenuPage]] = {
    "Chat & agents": [
        MenuPage(
            "pages/3_5_13_converse_demo.py", "Chat with any model", ":material/chat:",
            "A full-featured chat that works with every model in the catalog (Claude, GPT, Llama, Nova, "
            "Mistral, DeepSeek and more) through one API.",
            details=(
                "Pick a model; only the settings it accepts (temperature, top P, top K) are shown",
                "Attach images or documents (PDF, TXT, CSV, MD), depending on what the model supports",
                "Live token counts and latency in a side panel",
                "Save a conversation and load it back later; saved files are signed so edits are detected",
            ),
            services=("Bedrock Converse API", "CloudWatch Logs"),
            try_this="Attach a PDF or a photo and ask \"Summarize this\", then switch to another model and ask again.",
            differs="The general-purpose chat. No tools; see Tool-using assistant for that.",
            featured=True,
        ),
        MenuPage(
            "pages/3_7_5_converse_tool_v3.py", "Tool-using assistant", ":material/construction:",
            "A business-analyst assistant that decides which tools to call, chains several calls in a row, "
            "and shows the results as charts, tables and downloadable reports.",
            details=(
                "14 tools: sales data, KPIs, forecasts, anomaly detection, product search (text to SQL), "
                "charts, calculator, date/time, Wikipedia, AWS docs, web pages, PDF and PowerPoint",
                "Results are drawn as interactive charts, KPI cards and tables, with download links for files",
                "Per-turn metrics: tokens, latency, number of model calls and the order tools were used",
                "Built-in example questions to get started",
            ),
            services=("Bedrock Converse API", "Tool use", "DuckDB"),
            try_this="Ask \"Compare 2024 vs 2023 sales performance. What caused any underperformance?\"",
            differs="The production version of the tool loop, with rich rendering and no limit on tool rounds.",
            featured=True,
        ),
        MenuPage(
            "pages/3_7_4_converse_tool_demo.py", "Tool use, step by step", ":material/build:",
            "A simpler, single-file tool-use chat that shows each tool call and result as raw JSON, so you "
            "can see exactly how the model and tools talk to each other.",
            details=(
                "5 tools: calculator, date/time, acronyms, Wikipedia, web page reader",
                "Tick tools on or off and watch how the model's choices change",
                "A Tool Use Log lists every call with its input and result",
                "Stops after 5 tool rounds per question",
            ),
            services=("Bedrock Converse API", "Tool use"),
            try_this="Ask \"What is today's date, and what is 17% of 2,340?\" and watch the Tool Use Log.",
            differs="The readable prototype; Tool-using assistant is the full version with 14 tools and charts.",
        ),
        MenuPage(
            "pages/3_8_01_bedrock_mantle.py", "OpenAI-compatible API", ":material/api:",
            "Chat with OpenAI GPT models (and Claude) through Bedrock's OpenAI-compatible endpoint, with "
            "web search and cited sources.",
            details=(
                "GPT models use the OpenAI SDK and Responses API; Claude uses the Anthropic SDK",
                "Web search three ways: off, client-side (DuckDuckGo), or Bedrock's built-in server-side search",
                "Built-in search shows what was searched and read, and lists the sources",
                "Attach images and documents",
            ),
            services=("Bedrock Mantle", "OpenAI Responses API", "Bedrock web search"),
            try_this="Pick a GPT model, choose \"Bedrock built-in\" search, and ask for today's top AWS news.",
            needs="Built-in web search needs the bedrock-websearch IAM permissions. Mantle runs in us-east-1.",
            differs="Uses the OpenAI-style API instead of Bedrock's own Converse API.",
            featured=True,
        ),
        MenuPage(
            "pages/3_9_01_bedrock_harness_chat.py", "AgentCore harness chat", ":material/smart_toy:",
            "Chat with a managed agent running on Amazon Bedrock AgentCore, and look inside it: its memory, "
            "past sessions and the machine it runs on.",
            details=(
                "Live status while the agent calls its tools",
                "Browse short-term memory events and search long-term memory",
                "List past sessions and resume one",
                "Run shell commands in the agent's sandbox (no model tokens used)",
            ),
            services=("Bedrock AgentCore", "AgentCore Memory"),
            try_this="Pick a harness, run the \"OS info\" command, then chat and resume the session later.",
            needs="An AgentCore harness in the account; memory features need memory configured on it.",
        ),
        MenuPage(
            "pages/3_4_chat_demo.py", "Chat with native requests", ":material/code:",
            "A plain text chat that sends each provider's own request format (Claude, Nova, Llama, Mistral) "
            "instead of a common API, to show what Converse saves you from.",
            details=(
                "Hand-built request bodies and stream parsing per provider",
                "Shows the parameters sent, tokens, latency and time to first byte for each reply",
            ),
            services=("Bedrock InvokeModel", "CloudWatch metrics"),
            try_this="Ask the same question with a Claude, a Nova and a Llama model and compare the timings.",
            differs="The low-level way to call models. Chat with any model does the same through one API.",
        ),
    ],
    "Search & knowledge (RAG)": [
        MenuPage(
            "pages/6_2_4_filekb_demo.py", "Ask about a file", ":material/upload_file:",
            "Upload one file and ask questions about it, with numbered citations back to the text. Nothing "
            "to set up: no knowledge base or database.",
            details=(
                "Works with PDF, Word, Excel, CSV, HTML, Markdown and text files, and images",
                "Follow-up questions reuse the same file",
                "Answers cite the passages they came from",
            ),
            services=("Bedrock RetrieveAndGenerate", "Bedrock Converse API"),
            try_this="Attach a PDF, ask \"What are the key points?\", then ask a follow-up.",
            differs="The quickest way to chat with a document: one file, no setup.",
            featured=True,
        ),
        MenuPage(
            "pages/14_1_1_session_rag.py", "Chat with your documents", ":material/description:",
            "Build your own small search engine: upload documents, split and embed them into a local "
            "vector index, then ask questions answered from them.",
            details=(
                "Create sessions; each keeps its own documents and index, saved to disk",
                "Choose chunk size, overlap and how many passages to retrieve",
                "\"View Sources\" shows which passages were used and how close they matched",
            ),
            services=("Titan Text Embeddings", "FAISS", "Bedrock Converse API"),
            try_this="Create a session, upload a .txt or .md file, click Process, then ask about it.",
            differs="Retrieval-augmented generation built by hand, without a managed knowledge base.",
        ),
        MenuPage(
            "pages/3_9_50_bedrock_managed_kb.py", "Managed knowledge bases", ":material/library_books:",
            "A console for the knowledge bases in your account: browse their documents, then search them "
            "or get answers with citations.",
            details=(
                "Three modes: Retrieve (ranked passages with scores), Retrieve & Generate (answer with "
                "citations), and Agentic Retrieve (shows the agent's search steps)",
                "Filter by document metadata",
                "See each knowledge base's storage, embedding model and per-document sync status",
            ),
            services=("Bedrock Knowledge Bases", "Retrieve / RetrieveAndGenerate"),
            try_this="Pick a knowledge base, compare \"Retrieve\" passages with a \"Retrieve & Generate\" answer.",
            needs="At least one Bedrock Knowledge Base with synced documents.",
        ),
        MenuPage(
            "pages/2_s3_vector.py", "S3 Vectors store", ":material/database:",
            "Store and search text by meaning in Amazon S3 Vectors, and compare embedding models. Retrieval "
            "only, no chat answer.",
            details=(
                "Add documents one at a time, in bulk, or load 10 sample AWS service descriptions",
                "Search with metadata filters; results show similarity scores",
                "Choose the embedding model (Nova, Titan, Cohere) and vector size",
                "Inspect, preview and reset indexes",
            ),
            services=("Amazon S3 Vectors", "Nova Multimodal Embeddings"),
            try_this="Click \"Add Sample Data\", then search \"What is a serverless compute service?\"",
            needs="An S3 Vectors bucket and index (VECTOR_BUCKET_NAME, VECTOR_INDEX_NAME).",
        ),
        MenuPage(
            "pages/2_s3_vector_kb_3.py", "Multi-tenant knowledge base", ":material/groups:",
            "One shared knowledge base serving many users, where each user only ever sees their own "
            "documents.",
            details=(
                "Each user creates their own knowledge bases and uploads files",
                "Every search is filtered by user and knowledge base, so tenants stay isolated",
                "Switch users to check the isolation; a debug tab compares results with and without the filter",
            ),
            services=("Bedrock Knowledge Bases", "S3 Vectors", "DynamoDB", "S3"),
            try_this="As user-alice, upload a file and query it; switch to user-bob and the file is gone.",
            needs="A knowledge base, S3 buckets, two DynamoDB tables and an ingestion Lambda (CMN_KB_* settings).",
            differs="An architecture pattern for SaaS apps rather than a single-user demo.",
        ),
    ],
    "Images & vision": [
        MenuPage(
            "pages/9_1_8_image_generator.py", "Generate images", ":material/image:",
            "Describe a picture and get it, with three Stability AI models to compare.",
            details=(
                "Models: Stable Image Ultra, Stable Image Core, SD3.5 Large",
                "Aspect ratio, negative prompt, output format and a fixed seed for repeatable results",
            ),
            services=("Stability AI on Bedrock",),
            try_this="Try \"a lighthouse on a cliff at sunset, cinematic\" at 16:9 on Ultra, then on Core.",
            differs="Creates images from text only. The three pages below change an image you upload.",
            featured=True,
        ),
        MenuPage(
            "pages/9_5_7_image_stability_edit.py", "Edit images", ":material/auto_fix_high:",
            "Change a photo you upload: remove the background, erase or replace objects, recolor, or extend "
            "the edges.",
            details=(
                "Six tools: Remove Background, Erase, Inpaint, Outpaint, Search and Replace, Search and Recolor",
                "Side-by-side before and after, with a download button",
            ),
            services=("Stability AI on Bedrock",),
            try_this="Upload a product photo, run Remove Background and download the transparent PNG.",
            differs="Edits your image. Sketch & style control makes a new image guided by yours.",
            featured=True,
        ),
        MenuPage(
            "pages/9_5_7_image_stability_control.py", "Sketch & style control", ":material/brush:",
            "Make a new image guided by one you upload: turn a sketch into a picture, keep a photo's layout, "
            "or copy an image's style.",
            details=(
                "Four services: Control Sketch, Control Structure, Style Guide, Style Transfer",
                "Sliders for how closely to follow the input image",
            ),
            services=("Stability AI on Bedrock",),
            try_this="Pick Control Sketch, upload a rough line drawing and keep the default prompt.",
        ),
        MenuPage(
            "pages/9_5_7_image_stability_upscale.py", "Upscale images", ":material/high_quality:",
            "Make a small image bigger and sharper.",
            details=(
                "Fast (4x), Conservative (keeps detail) and Creative (up to 4K, can add detail from a prompt)",
            ),
            services=("Stability AI on Bedrock",),
            try_this="Upload a small JPEG, run Fast Upscale and compare.",
        ),
        MenuPage(
            "pages/8_1_6_image_query.py", "Ask about an image", ":material/image_search:",
            "Upload a picture and chat about it: describe it, read text from it, find objects, judge its "
            "composition.",
            details=(
                "Multi-turn questions about one image",
                "One click adds Amazon Rekognition's labels and dominant colors for comparison",
            ),
            services=("Bedrock Converse API", "Amazon Rekognition"),
            try_this="Upload a photo of a sign or receipt and ask \"Extract all visible text\".",
            differs="Open-ended chat about one image. Compare image colors compares two.",
        ),
        MenuPage(
            "pages/8_5_7_image_color_compare.py", "Compare image colors", ":material/palette:",
            "Upload two images and get a written comparison of their colors and quality, with a similarity "
            "score from 1 to 10.",
            details=(
                "Measures brightness, sharpness, contrast and dominant colors with Rekognition and locally",
                "\"Compare with Meta\" feeds those measurements to Claude and answers in English and Japanese",
            ),
            services=("Bedrock InvokeModel", "Amazon Rekognition"),
            try_this="Upload two photos of one product in different colors and click \"Compare with Meta\".",
        ),
        MenuPage(
            "pages/99_01_multimodal_search.py", "Multimodal product search", ":material/manage_search:",
            "A product catalog you can search with a photo or with words, because images and text share "
            "one embedding space.",
            details=(
                "Register products with a photo, title and description; edit or delete them",
                "Search by image, by text (falls back to keyword search), or by title",
            ),
            services=("Titan Multimodal Embeddings", "OpenSearch Serverless"),
            try_this="Register 2 or 3 product photos, then search with a similar photo.",
            needs="An OpenSearch Serverless vector collection (OPENSEARCH_ENDPOINT).",
        ),
    ],
    "Prompts & tools": [
        MenuPage(
            "pages/1_count_token.py", "Token counter", ":material/tag:",
            "See how many input tokens a prompt or whole conversation will cost before you send it.",
            details=("Count a single message, or a conversation with a system prompt and several turns",),
            services=("Bedrock CountTokens",),
            try_this="Use the pre-filled conversation and click Count Tokens. Counting is free.",
        ),
        MenuPage(
            "pages/2_prompt_template.py", "Prompt manager", ":material/edit_note:",
            "Run prompts saved in Bedrock Prompt Management: fill in their variables, override settings, "
            "and invoke.",
            details=(
                "Variables like {{topic}} are detected and turned into input fields",
                "Lists the prompts in your account; includes setup docs and the IAM policy",
            ),
            services=("Bedrock Prompt Management", "Bedrock Converse API"),
            try_this="Open List Prompts, copy an ID into Invoke Prompt, fill the variables and run it.",
            needs="At least one prompt saved in Bedrock Prompt Management.",
        ),
        MenuPage(
            "pages/13_1_1_resformat.py", "Structured output", ":material/data_object:",
            "Four ways to make a model return reliable JSON, compared side by side on the same driving data "
            "to produce a driver risk score.",
            details=(
                "Streaming JSON from a prompt, schema-enforced tool use, a schema in the prompt, and native tool use",
                "Each shows the risk score, the raw JSON, latency, tokens and estimated cost",
                "A guide tab explains when to use which approach",
            ),
            services=("Bedrock Converse API", "Tool use", "InvokeModel"),
            try_this="Keep the defaults and run the Streaming and Tool Use tabs; compare speed and output.",
        ),
        MenuPage(
            "pages/12_1_1_dataset.py", "Table row selection", ":material/table_rows:",
            "A Streamlit table where clicking a row shows its details. A UI pattern, no AWS calls.",
            details=("Row selection, progress bars and dropdowns inside the table",),
            try_this="Click any row, then generate new data.",
        ),
    ],
}
