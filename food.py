import os, re, base64, json
from pathlib import Path
from typing import Annotated, Optional, TypedDict
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DataFrameLoader, UnstructuredExcelLoader
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ingest import ingest_nutrition_excels, test_retriever
from model import MODEL_NAME, client
from dotenv import load_dotenv
from prompt import VISION_PROMPT, system_prompt

load_dotenv()

app = FastAPI(title="Kittchens AI API")

# ======================================================
# LOAD PDF KNOWLEDGE BASE (ONCE)
# ======================================================
PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DOCS_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "db" / "chroma_db"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# Load documents
documents = ingest_nutrition_excels(DOCS_DIR)

# Chroma + embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
vector_store = Chroma(
    persist_directory=str(PERSIST_DIR), 
    embedding_function=embeddings,
    collection_name="food",
)


# --- Create collection if it doesn't exist ---
collection_name = "food"
chroma_client = vector_store._client  # underlying chromadb client
try:
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Found existing collection '{collection_name}' with {collection.count()} documents")
except Exception:
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Created new collection '{collection_name}'")

# --- Add documents if collection is empty ---
if collection.count() == 0:
    print(f"Embedding {len(documents)} nutrition documents...")
    vector_store.add_documents(documents)
else:
    print(f"Using existing Chroma index with {collection.count()} documents")

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Test retrieval
query = "nutrition of chicken and palm oil"
docs = test_retriever(retriever=retriever, text=query)


CACHE = {}  # Simple in-memory cache
# ======================================================
# Function 1
# ======================================================
def call_vision_model(image_bytes: bytes) -> str:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    img_hash = image_base64
    if img_hash and img_hash in CACHE:
        return CACHE[img_hash]

    image_content = [
        {"type": "text", "text": "Describe the food in this image."},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {"role": "user", "content": image_content}
        ]
    )

    raw_output = completion.choices[0].message.content


    if img_hash:
        CACHE[img_hash] = raw_output
    return  raw_output


# ======================================================
# Function 2
# ======================================================
def clean_json_markdown(raw: str) -> str:
    """
    Remove ```json and ``` fences if present
    """
    # Remove code fences (```json or ```)
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.DOTALL)
    return cleaned


# ======================================================
# Function 3
# ======================================================
def extract_image_bytes(messages: list) -> bytes | None:
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            for part in msg.content:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]

                    # Expected format: data:image/jpeg;base64,XXXX
                    if url.startswith("data:image"):
                        base64_data = url.split(",", 1)[1]
                        return base64.b64decode(base64_data)

    return None

# ======================================================
# Graph State
# ======================================================
class MessagesState(TypedDict):
    """
    TypedDict state for LangGraph workflow.
    Holds:
    - messages: list of exchanged messages (HumanMessage, SystemMessage, etc.)
    - context: vision description or other textual context
    """
    messages: Annotated[list[AnyMessage], add_messages]
    context: Optional[str]
    image: Optional[bytes]
    tool_called: Optional[bool]
    llm_text: Optional[str]     # raw LLM output
    llm_json: Optional[dict]    # refined structured output


# TOOL 1
def retrieve_docs(query: str) -> str:
    """
    Retrieve trusted internal nutrition knowledge from the Kitchens AI food database.

    Use this tool to look up authoritative nutritional information for Nigerian and African foods,
    including calories, macronutrients, glycemic index, and health notes.
    Always call this tool when answering food, nutrition, calorie, or health-related questions
    before relying on general knowledge.

    If no relevant internal data is found, the tool returns the string "NO_INTERNAL_DATA_FOUND".
    """
    docs = retriever.invoke(query)
    if not docs:
        return "NO_INTERNAL_DATA_FOUND"
    print(f"\n\n\nThis is the response from retrieval: {docs}\n\n\n")
    return "\n\n".join(d.page_content for d in docs)

# TOOL 2
web_search = TavilySearch(max_results=3)

tools = [retrieve_docs, web_search]
tool_node = ToolNode(tools)



# ======================================================
# LLM
# ======================================================
llm_without_tools = ChatGroq(
    model=os.getenv("LLAMA_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

llm = ChatGroq(
    model=os.getenv("LLAMA_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
).bind_tools(tools)




# ======================================================
# 1. VISION NODE
# ======================================================
def call_vision(state: MessagesState):
    """Invokes the vision model on the current image passed and returns image description."""
    image_bytes = state["image"]
    # image_bytes = extract_image_bytes(state["messages"])

    if not image_bytes:
        print("No image found in messages")
        return {"context": None}

    response = call_vision_model(image_bytes)
    return {"context": response}


# ======================================================
# 2. AGENT NODE
# ======================================================
def call_agent(state: MessagesState):
    """Invokes the LLM agent on the current message state and returns updated messages."""
    messages = [
        SystemMessage(content=system_prompt(state["context"]))
    ] + state["messages"]

    response = llm.invoke(messages)
    print(f"\n\n\nThe llm response after calling tools: {response}\n\n\n")

    return {
        "messages": [response],
        "llm_text": response.content,
        "tool_called": True
    }



# ======================================================
# 3. AGENT NODE
# ======================================================
def validate_tool_usage(state: MessagesState):
    """Validates that at least one tool was used; updates state accordingly."""
    if not state.get("tool_called", False):
        state["messages"] = [
            AIMessage(content="I do not have enough verified data to provide an accurate nutrition analysis.")
        ]
        # Keep llm_text if available
        state["llm_json"] = {}
    return state

# ======================================================
# 4. AGENT NODE
# ======================================================
def refine_to_json(state: MessagesState):
    if not state.get("llm_text"):
        return {"llm_json": None, "error": "No LLM output to refine"}

    # Prompt for LLM to produce enriched JSON based on your nutrition Excel
    refinement_prompt = """
    Convert the meal analysis into valid JSON with the following enriched structure:

    {
      "meal_analysis": {
        "meal_description": "...",
        "vision_description": "...",
        "meal_components": [
          {
            "name": "...",
            "type": "...",
            "portion_size": "...",
            "ingredients": ["...", "..."],
            "macronutrients": {
              "calories": ...,
              "carbohydrates_g": ...,
              "protein_g": ...,
              "fat_g": ...,
              "fiber_g": ...
            },
            "micronutrients": ["...", "..."],
            "glycemic_index": "...",
            "health_benefits": "...",
            "health_risks": "...",
            "medical_notes": "...",
            "primary_source": "...",
            "source": "...",
            "secondary_source": "...",
            "confidence_level": "..."
          }
        ],
        "total_nutrition": {
          "calories": ...,
          "carbohydrates_g": ...,
          "protein_g": ...,
          "fat_g": ...,
          "fiber_g": ...,
          "sodium": "low/medium/high"
        },
        "health_advice": {
          "for_diabetes": "...",
          "for_hypertension": "...",
          "general": "..."
        },
        "frontend_info": {
          "safe_meal_score": "...",
          "meal_type": "...",
          "serving_suggestions": "..."
        }
      }
    }

    Include all available information from your knowledge base (Excel) for each ingredient.
    Return ONLY valid JSON, no commentary.
    """

    response = llm_without_tools.invoke([
        SystemMessage(content=refinement_prompt),
        HumanMessage(content=state["llm_text"])
    ])

    json_text = clean_json_markdown(response.content).strip()

    enriched_json = {
        "raw_summary": state["llm_text"],
        "meal_analysis": None 
    }

    try:
        parsed_json = json.loads(json_text)
        enriched_json["meal_analysis"] = parsed_json.get("meal_analysis")
    except json.JSONDecodeError as e:
        enriched_json["error"] = f"JSON parse error: {str(e)}"

    state["llm_json"] = enriched_json
    return state


# ======================================================
# 7. LANGGRAPH
# ======================================================
workflow = StateGraph(MessagesState)
workflow.add_node("vision", call_vision)
workflow.add_node("agent", call_agent)
workflow.add_node("json", refine_to_json)
workflow.add_node("tools", tool_node)
workflow.add_node("validate", validate_tool_usage)


workflow.add_edge(START, "vision")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: "validate"
    }
)
workflow.add_edge("vision", "agent")
workflow.add_edge("tools", "agent")
workflow.add_edge("validate", "json")

food = workflow.compile()

# ======================================================
# 8. API ENDPOINT
# ======================================================
@app.post("/analyze-meal")
async def analyze_meal(
    query: Optional[str] = Form(None),
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()

        #  User intent
        user_query = query or "Analyze the nutritional value of this meal."

        #  Agent input
        message = HumanMessage(content=user_query)

        events = list(
            food.stream({
                "image": image_bytes,
                "messages": [message],
                "context": None,
                "tool_called": False,
                "llm_text": None,
                "llm_json": None
            })
        )

        final_event = events[-1]

        llm_json = list(final_event.values())[0].get("llm_json")

        try:
            return JSONResponse(llm_json)
        except json.JSONDecodeError:
            json_output = {"raw_output": llm_json, "error": "Failed to parse JSON"}

        return JSONResponse(json_output)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
