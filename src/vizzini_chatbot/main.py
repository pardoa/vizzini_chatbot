#!/usr/bin/env python
import sys
import warnings

from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangSmith + OpenTelemetry instrumentation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from langsmith.integrations.otel import OtelSpanProcessor
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor

# Set up tracing before importing CrewAI crew
tracer_provider = trace.get_tracer_provider()
if not isinstance(tracer_provider, TracerProvider):
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)

tracer_provider.add_span_processor(OtelSpanProcessor())

# Instrument CrewAI and OpenAI
CrewAIInstrumentor().instrument()
OpenAIInstrumentor().instrument()

from vizzini_chatbot.crew import VizziniChatbot

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs',
        'current_year': str(datetime.now().year)
    }

    try:
        VizziniChatbot().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        VizziniChatbot().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        VizziniChatbot().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        VizziniChatbot().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = VizziniChatbot().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
