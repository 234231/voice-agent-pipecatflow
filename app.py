
import os
import logging
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.base_transport import TransportParams


import os

from dotenv import load_dotenv
from loguru import logger
import asyncio
import logging 
from pipecat.services.deepgram.stt import DeepgramSTTService 
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService 
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport 

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)

load_dotenv(override=True)
import logging
from loguru import logger

# Set up logging
logging.basicConfig(level=logging.INFO)

# -------------------
# Flow nodes with logging
# -------------------

def create_initial_node() -> NodeConfig:
    logger.info("Creating initial node")
    
    record_name_func = FlowsFunctionSchema(
        name="record_name_func",
        description="Record the name the user provides.",
        required=["name"],
        handler=record_name_and_set_next_node,
        properties={"name": {"type": "string"}},
    )

    return NodeConfig(
        name="initial",
        role_messages=[{
            "role": "system",
            "content": "You are an inquisitive child. Use simple language. Always use one of the available functions."
        }],
        task_messages=[{
            "role": "system",
            "content": "Say 'Hello!' and ask what the user's name is."
        }],
        functions=[record_name_func],
    )

async def record_name_and_set_next_node(args: FlowArgs, flow_manager: FlowManager) -> tuple[str, NodeConfig]:
    user_name = args.get("name", "<unknown>")
    logger.info(f"[Initial Node] User's name received: {user_name}")
    
    # Prepare second node
    second_node = create_second_node(user_name)
    logger.info("[Initial Node] Transitioning to second node")
    return user_name, second_node

def create_second_node(user_name: str) -> NodeConfig:
    logger.info(f"Creating second node for user: {user_name}")
    
    ask_help_func = FlowsFunctionSchema(
        name="ask_help_func",
        description="Ask what the user wants help with.",
        required=["request"],
        handler=handle_user_request,
        properties={"request": {"type": "string"}},
    )

    return NodeConfig(
        name="second",
        role_messages=[{
            "role": "system",
            "content": "You are an inquisitive child. Speak in simple words. Always use available functions to progress."
        }],
        task_messages=[{
            "role": "system",
            "content": f"Hello {user_name}! What can I help you with today?"
        }],
        functions=[ask_help_func],
        
    )

async def handle_user_request(args, flow_manager):
    user_request = args.get("request")
    logger.info(f"[Second Node] User request received: {user_request}")

    # Check if user wants to end the conversation
    if user_request.lower() in ["bye", "goodbye", "nothing", "exit"]:
        # Create end node and transition
        end_node = create_end_node()
        logger.info("[Second Node] User requested to end conversation.")
        return None, end_node

    # Otherwise, process help request here (e.g., call LLM, TTS, or log)
    # For now, we just acknowledge and stay in the same node
    logger.info(f"[Second Node] Processing help request: {user_request}")
    # Optionally: send TTS or LLM response using flow_manager
    return None, None  # Stay in the second node

def create_end_node(user_name: str = None) -> NodeConfig:
    content = f"Thank you, {user_name}, for answering. Goodbye!" if user_name else "Thank you for answering. Goodbye!"
    logger.info(f"Creating end node. Content: {content}")

    return NodeConfig(
        name="end_node",
        task_messages=[{"role": "system", "content": content}],
        post_actions=[{"type": "end_conversation"}],
    )

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # --- Initialize services ---
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="32b3f3c5-7171-46aa-abe7-b598964aa793",
        text_filters=[MarkdownTextFilter()],
    )
    
    llm = OpenRouterLLMService(api_key=os.getenv("OPENROUTER_API_KEY"), model="gpt-4o-mini")

    # --- Context setup ---
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    # --- Build pipeline ---
    pipeline = Pipeline(
        [
            transport.input(),          # User input
            stt,                        # Deepgram STT
            context_aggregator.user(),  # Track user context
            llm,                        # OpenRouter LLM
            tts,                        # Cartesia TTS
            transport.output(),         # Output to user
            context_aggregator.assistant(),  # Track assistant context
            
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # --- Initialize flow manager ---
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    # --- Transport event handlers ---
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Start the conversation flow
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        # Cancel the pipeline task
        await task.cancel()

    # --- Run the pipeline ---
    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)

load_dotenv(override=True)

# -------------------
# FastAPI setup
# -------------------
app = FastAPI()

# Serve static HTML page to connect
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    # This page should contain your WebRTC JS client to connect to /api/offer
    return FileResponse("voice.html")

# -------------------
# WebRTC offer endpoint
# -------------------
@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    """
    WebRTC offer/answer endpoint.
    request should contain {"sdp": ..., "type": ...} from the browser
    """
    # Create WebRTC connection
    webrtc_connection = SmallWebRTCConnection()
    await webrtc_connection.initialize(sdp=request["sdp"], type=request["type"])

    # Wrap it in SmallWebRTCTransport
    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        webrtc_connection=webrtc_connection,
    )

    # Run your bot in background
    background_tasks.add_task(run_bot, transport, RunnerArguments())

    # Return WebRTC answer
    answer = webrtc_connection.get_answer()
    return {"sdp": answer["sdp"], "type": answer["type"]}

# -------------------
# Health check endpoint
# -------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Pipecat SmallWebRTC server is running"}

# -------------------
# Run FastAPI server
# -------------------
if __name__ == "__main__":
    import uvicorn

    required_env_vars = ["DEEPGRAM_API_KEY", "OPENROUTER_API_KEY", "CARTESIA_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logging.error(f"Missing required environment variables: {missing_vars}")
        exit(1)

    logging.info("Starting Pipecat SmallWebRTC FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
