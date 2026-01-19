"""
Intelligent Interruption Handling for LiveKit Voice Agents

This module implements state-aware interruption filtering to distinguish between:
- Passive acknowledgements ("yeah", "ok", "hmm") → IGNORE while agent speaking
- Active interruptions ("stop", "wait", "no") → INTERRUPT immediately  
- Real content (substantive input) → INTERRUPT and process

The key insight is that VAD fires BEFORE STT produces a transcript.
We use the framework's `resume_false_interruption` feature to pause (not stop)
on potential interruptions, then analyze the transcript to decide the action.

Usage:
    python -m examples.voice_agents.intelligent_interruption_agent

For interactive testing:
    python -m examples.voice_agents.intelligent_interruption_agent console
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    JobContext,
    JobProcess,
    RunContext,
    UserInputTranscribedEvent,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice.events import AgentFalseInterruptionEvent
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logger = logging.getLogger("intelligent-interruption-agent")
logger.setLevel(logging.INFO)

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class InterruptionConfig:
    """Configuration for intelligent interruption handling.

    Attributes:
        backchannel_phrases: Set of phrases to ignore when agent is speaking.
            These are passive acknowledgements that don't require a response.
        explicit_stop_phrases: Set of phrases that ALWAYS interrupt the agent.
            These indicate the user wants the agent to stop speaking.
        min_words_for_interruption: Minimum number of words required to consider
            speech as a real interruption (not a backchannel).
        backchannel_grace_period: Seconds after agent stops speaking during which
            backchannels are still ignored (handles timing edge cases).
    """

    backchannel_phrases: set[str] = field(
        default_factory=lambda: {
            # Single word acknowledgements
            "yeah",
            "yep",
            "yes",
            "yea",
            "ya",
            "ok",
            "okay",
            "k",
            "sure",
            "right",
            "true",
            "correct",
            "alright",
            # Vocal fillers / backchannels
            "uh-huh",
            "uh huh",
            "uhuh",
            "mm-hmm",
            "mmhmm",
            "mm hmm",
            "mhm",
            "hmm",
            "hm",
            "um",
            "uh",
            "ah",
            # Confirmations
            "got it",
            "i see",
            "all right",
            "i understand",
            "understood",
            "makes sense",
        }
    )

    explicit_stop_phrases: set[str] = field(
        default_factory=lambda: {
            # Direct stop commands
            "stop",
            "wait",
            "no",
            "hold on",
            "pause",
            "stop it",
            "stop please",
            "please stop",
            # Stronger requests
            "be quiet",
            "shut up",
            "quiet",
            "enough",
            "stop talking",
            "hold",
            "wait wait",
            "no no",
            "hang on",
            # Polite versions
            "one moment",
            "just a moment",
            "excuse me",
        }
    )

    min_words_for_interruption: int = 2
    backchannel_grace_period: float = 0.3  # seconds


# Global config instance (can be customized)
INTERRUPTION_CONFIG = InterruptionConfig()


# ============================================================================
# State Tracking
# ============================================================================


class AgentStateTracker:
    """Tracks agent speaking state for interruption filtering decisions.

    This class monitors agent state changes to know when the agent is speaking,
    which is critical for deciding whether to ignore backchannels.
    """

    def __init__(self) -> None:
        self.is_speaking: bool = False
        self.last_speaking_time: float = 0.0
        self.speaking_start_time: float = 0.0
        self._state_history: list[tuple[float, str]] = []

    def on_agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        """Handle agent state change events."""
        now = time.time()
        self._state_history.append((now, ev.new_state))

        if ev.new_state == "speaking":
            self.is_speaking = True
            self.speaking_start_time = now
            logger.debug(f"🎤 Agent started speaking at {now:.2f}")
        else:
            if self.is_speaking:
                self.last_speaking_time = now
                duration = now - self.speaking_start_time
                logger.debug(
                    f"🔇 Agent stopped speaking at {now:.2f} (spoke for {duration:.2f}s)"
                )
            self.is_speaking = False

    def was_speaking_recently(self, threshold_seconds: float = 0.5) -> bool:
        """Check if agent was speaking within the threshold window.

        This handles race conditions where VAD fires slightly after
        the agent technically stopped but before transcript arrives.
        """
        if self.is_speaking:
            return True
        return (time.time() - self.last_speaking_time) < threshold_seconds

    def get_speaking_duration(self) -> float:
        """Get how long the agent has been speaking (0 if not speaking)."""
        if not self.is_speaking:
            return 0.0
        return time.time() - self.speaking_start_time


# ============================================================================
# Transcript Classification
# ============================================================================


class TranscriptClassifier:
    """Classifies user transcripts for interruption decision making."""

    def __init__(self, config: InterruptionConfig) -> None:
        self.config = config

    def classify(
        self, transcript: str
    ) -> Literal["backchannel", "explicit_stop", "real_content"]:
        """Classify a transcript into actionable categories.

        Args:
            transcript: The user's speech transcript from STT.

        Returns:
            - "backchannel": Passive acknowledgement, should be ignored if agent speaking
            - "explicit_stop": User wants agent to stop, always interrupt
            - "real_content": Substantive input, should interrupt if agent speaking
        """
        normalized = transcript.lower().strip()

        # Remove common punctuation for matching
        normalized = normalized.rstrip(".,!?")

        # 1. Check for explicit stop phrases first (highest priority)
        for stop_phrase in self.config.explicit_stop_phrases:
            if stop_phrase in normalized:
                return "explicit_stop"

        # 2. Check if entire transcript is a known backchannel
        if normalized in self.config.backchannel_phrases:
            return "backchannel"

        # 3. Check word count - very short phrases are likely backchannels
        words = normalized.split()
        word_count = len(words)

        if word_count < self.config.min_words_for_interruption:
            # Single words not in explicit_stop_phrases are likely backchannels
            return "backchannel"

        # 4. Default to real content
        return "real_content"


# ============================================================================
# Intelligent Interruption Handler
# ============================================================================


class IntelligentInterruptionHandler:
    """Handles interruption decisions based on transcript classification and agent state."""

    def __init__(
        self,
        config: InterruptionConfig,
        state_tracker: AgentStateTracker,
        session: AgentSession,
    ) -> None:
        self.config = config
        self.state_tracker = state_tracker
        self.session = session
        self.classifier = TranscriptClassifier(config)

        # Statistics for debugging
        self.stats = {
            "backchannels_ignored": 0,
            "explicit_stops": 0,
            "real_interruptions": 0,
            "normal_inputs": 0,
        }

    def should_process_as_interruption(
        self, transcript: str, is_final: bool
    ) -> bool:
        """Determine if this transcript should interrupt the agent.

        Args:
            transcript: The user's speech transcript.
            is_final: Whether this is a final or interim transcript.

        Returns:
            True if the transcript should trigger an interruption/response,
            False if it should be ignored (backchannel during agent speech).
        """
        # Get agent speaking state
        agent_speaking = self.state_tracker.is_speaking
        agent_was_speaking_recently = self.state_tracker.was_speaking_recently(
            self.config.backchannel_grace_period
        )

        # If agent is not speaking and wasn't recently, always process normally
        if not agent_speaking and not agent_was_speaking_recently:
            self.stats["normal_inputs"] += 1
            logger.info(f"📝 Normal input (agent silent): '{transcript}'")
            return True

        # Agent is/was speaking - classify the transcript
        classification = self.classifier.classify(transcript)

        if classification == "explicit_stop":
            self.stats["explicit_stops"] += 1
            logger.info(f"🛑 INTERRUPT - Explicit stop detected: '{transcript}'")
            return True

        if classification == "backchannel":
            self.stats["backchannels_ignored"] += 1
            logger.info(
                f"🔇 IGNORED - Backchannel while agent speaking: '{transcript}'"
            )
            return False

        if classification == "real_content":
            self.stats["real_interruptions"] += 1
            logger.info(f"⏹️ INTERRUPT - Real content detected: '{transcript}'")
            return True

        # Default: process as interruption (safety)
        return True

    def log_stats(self) -> None:
        """Log statistics about interruption handling."""
        logger.info(f"📊 Interruption Statistics: {self.stats}")


# ============================================================================
# Agent Implementation
# ============================================================================


class IntelligentInterruptionAgent(Agent):
    """Voice agent with intelligent interruption handling.

    This agent distinguishes between passive acknowledgements (backchannels)
    and real interruptions, allowing natural conversation flow.
    """

    def __init__(self, config: InterruptionConfig | None = None) -> None:
        super().__init__(
            instructions="""You are a helpful voice assistant named Alex.
You speak in a conversational, friendly manner.
When asked to tell stories or give long explanations, take your time and speak naturally.
Keep responses concise but informative.
Do not use emojis, asterisks, markdown, or special formatting - speak naturally.""",
        )
        self.config = config or INTERRUPTION_CONFIG
        self.state_tracker = AgentStateTracker()
        self.handler: IntelligentInterruptionHandler | None = None

    async def on_enter(self) -> None:
        """Called when the agent enters the session."""
        # Set up the interruption handler
        self.handler = IntelligentInterruptionHandler(
            config=self.config,
            state_tracker=self.state_tracker,
            session=self.session,
        )

        # Subscribe to agent state changes
        self.session.on("agent_state_changed", self.state_tracker.on_agent_state_changed)

        # Subscribe to user input for filtering
        self.session.on("user_input_transcribed", self._on_user_input_transcribed)

        # Subscribe to false interruption events for debugging
        self.session.on("agent_false_interruption", self._on_false_interruption)

        logger.info("🤖 Intelligent Interruption Agent started")
        logger.info(f"📋 Backchannel phrases: {len(self.config.backchannel_phrases)} configured")
        logger.info(f"🛑 Stop phrases: {len(self.config.explicit_stop_phrases)} configured")

        # Greet the user
        self.session.generate_reply()

    def _on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        """Handle user input transcription events.

        This is where we apply our intelligent filtering logic.
        Note: The framework has already made an interruption decision by this point
        based on VAD. We use this for logging and the resume_false_interruption
        feature handles the actual pause/resume logic.
        """
        if not self.handler:
            return

        # Log the decision our handler would make
        should_process = self.handler.should_process_as_interruption(
            ev.transcript, ev.is_final
        )

        if ev.is_final:
            logger.debug(
                f"📝 Final transcript: '{ev.transcript}' "
                f"| Agent speaking: {self.state_tracker.is_speaking} "
                f"| Should process: {should_process}"
            )

    def _on_false_interruption(self, ev: AgentFalseInterruptionEvent) -> None:
        """Handle false interruption events from the framework."""
        logger.info(
            f"🔄 False interruption detected by framework - "
            f"Resumed: {ev.resumed}"
        )

    @function_tool
    async def tell_story(self, context: RunContext, topic: str = "adventure") -> str:
        """Tell a story about a given topic. Use this to demonstrate interruption handling.

        Args:
            topic: The topic of the story to tell.
        """
        logger.info(f"📖 Telling story about: {topic}")
        return f"""Here's a story about {topic}:

Once upon a time, in a land far away, there lived a brave explorer who loved {topic}.
Every day, they would venture into the unknown, discovering new wonders and facing exciting challenges.
The journey was long and filled with many twists and turns.
Along the way, they met interesting characters who taught them valuable lessons.
And in the end, they learned that the greatest adventures are the ones we share with others.
The end."""

    @function_tool
    async def get_interruption_stats(self, context: RunContext) -> str:
        """Get statistics about interruption handling during this session."""
        if self.handler:
            stats = self.handler.stats
            return f"""Interruption Statistics:
- Backchannels ignored: {stats['backchannels_ignored']}
- Explicit stops: {stats['explicit_stops']}
- Real interruptions: {stats['real_interruptions']}
- Normal inputs: {stats['normal_inputs']}"""
        return "No statistics available yet."


# ============================================================================
# Server Setup
# ============================================================================

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Prewarm the VAD model for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Main entry point for the voice agent session."""

    session = AgentSession(
        # Speech-to-text configuration
        stt="deepgram/nova-3",
        # LLM configuration
        llm="openai/gpt-4.1-mini",
        # Text-to-speech configuration
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # VAD configuration
        vad=ctx.proc.userdata["vad"],
        # Turn detection
        turn_detection=MultilingualModel(),
        # =====================================================================
        # KEY CONFIGURATION FOR INTELLIGENT INTERRUPTION HANDLING
        # =====================================================================
        # Allow the user to interrupt the agent
        allow_interruptions=True,
        # CRITICAL: Pause instead of hard-stop on potential interruptions
        # This gives us time to analyze the transcript
        resume_false_interruption=True,
        # Time window to wait for transcript before deciding if false interruption
        false_interruption_timeout=1.5,
        # Minimum speech duration to consider as potential interruption
        min_interruption_duration=0.4,
        # Minimum words required (we also implement our own word-based filtering)
        min_interruption_words=0,
        # Allow preemptive generation for lower latency
        preemptive_generation=True,
    )

    # Create and start the agent
    agent = IntelligentInterruptionAgent()

    await session.start(
        agent=agent,
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
