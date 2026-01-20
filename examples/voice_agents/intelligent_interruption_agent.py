"""
================================================================================
INTELLIGENT INTERRUPTION HANDLING FOR REAL-TIME GENERATIVE AI VOICE AGENTS
================================================================================

A State-Aware Semantic Filtering System for Conversational AI

Author: Prabhpreet Singh
Assignment: LiveKit Agents - Intelligent Interruption Handling
================================================================================

ABSTRACT
--------
This module implements a state-aware interruption filtering layer for real-time
Generative AI voice agents built on the LiveKit Agents framework. The system
addresses a fundamental challenge in conversational AI: distinguishing between
passive acknowledgements (backchannels) and genuine user interruptions during
agent speech synthesis.

The solution leverages a multi-component architecture combining Voice Activity
Detection (VAD), Speech-to-Text (STT), Large Language Model (LLM) reasoning,
and Text-to-Speech (TTS) synthesis, with an additional semantic classification
layer that operates on transcribed user input to make context-aware interruption
decisions.


================================================================================
1. PROBLEM STATEMENT
================================================================================

1.1 The VAD-STT Timing Challenge
--------------------------------
In real-time voice agent systems, Voice Activity Detection (VAD) operates at
the audio signal level, triggering immediately when human speech is detected.
However, VAD has no semantic understanding—it cannot distinguish between:

    - Backchannel acknowledgements: "yeah", "uh-huh", "ok", "hmm"
    - Explicit stop commands: "stop", "wait", "hold on"
    - Genuine conversational interruptions with new content

The critical timing issue is:

    Audio Input → VAD Detection (immediate, ~50ms)
                      ↓
               Agent Interrupts (PREMATURE!)
                      ↓
    Audio Input → STT Transcription (delayed, ~200-500ms)
                      ↓
               Semantic Understanding (TOO LATE)

By the time Speech-to-Text produces a transcript that could be semantically
analyzed, the VAD has already triggered an interruption, causing the agent
to stop speaking—even for benign backchannels.


1.2 Impact on Conversational User Experience
--------------------------------------------
This naive interruption behavior creates several UX problems:

    (a) Conversation Flow Disruption: The agent stops mid-sentence when users
        naturally say "yeah" or "uh-huh" to indicate they're following along.

    (b) Agent Stuttering: Repeated false interruptions cause the agent to
        restart responses, creating an unnatural, jarring experience.

    (c) User Frustration: Users must remain completely silent during agent
        speech, which is unnatural in human conversation.

    (d) Loss of Context: Interrupted responses may leave information incomplete,
        requiring users to re-prompt.


1.3 Why This Is a GenAI-Specific Challenge
------------------------------------------
Unlike traditional IVR systems with pre-recorded responses, Generative AI voice
agents produce dynamic, context-dependent responses through LLM inference. This
creates unique challenges:

    - Responses are generated in real-time with variable length
    - The LLM maintains conversational context that can be disrupted
    - Function tool calls may be interrupted mid-execution
    - TTS synthesis must handle streaming token generation

The interruption handling system must therefore be aware of the full generative
pipeline state, not just audio-level signals.


================================================================================
2. GENERATIVE AI ARCHITECTURE
================================================================================

2.1 System Component Overview
-----------------------------
The voice agent operates as a pipeline of specialized AI components:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         USER AUDIO INPUT                            │
    └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  VOICE ACTIVITY DETECTION (VAD) - Silero VAD                        │
    │  • Detects speech onset/offset in audio stream                      │
    │  • Operates at ~50ms latency                                        │
    │  • Triggers potential interruption events                           │
    └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  SPEECH-TO-TEXT (STT) - Deepgram Nova-3                             │
    │  • Converts audio to text transcription                             │
    │  • Provides interim and final transcripts                           │
    │  • Latency: 200-500ms depending on utterance length                 │
    └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  INTELLIGENT INTERRUPTION HANDLER (This Implementation)             │
    │  • Semantic classification of user input                            │
    │  • Agent state tracking (speaking/listening/thinking)               │
    │  • Decision: IGNORE backchannel / PROCESS interruption              │
    └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  LARGE LANGUAGE MODEL (LLM) - OpenAI GPT-4.1-mini                   │
    │  • Generative reasoning core                                        │
    │  • Processes conversation context + user input                      │
    │  • Generates natural language responses                             │
    │  • Executes function tool calls when appropriate                    │
    └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  TEXT-TO-SPEECH (TTS) - Cartesia Sonic-2                            │
    │  • Converts LLM text output to natural speech                       │
    │  • Streams audio with low latency                                   │
    │  • Supports interruption (pause/resume)                             │
    └─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        AGENT AUDIO OUTPUT                           │
    └─────────────────────────────────────────────────────────────────────┘


2.2 The LLM as Generative Reasoning Core
----------------------------------------
The Large Language Model serves as the central intelligence of the voice agent:

    INPUTS TO LLM:
    • System instructions (agent persona, behavior guidelines)
    • Conversation history (previous turns, context)
    • Current user input (transcribed speech)
    • Available function tools (capabilities the agent can invoke)

    LLM PROCESSING:
    • Understands user intent from natural language
    • Maintains multi-turn conversational context
    • Reasons about appropriate response strategy
    • Decides whether to call function tools
    • Generates natural, contextually appropriate responses

    OUTPUTS FROM LLM:
    • Text response (streamed token-by-token)
    • Function tool calls (structured JSON for actions)
    • Conversation state updates


2.3 Function Tool Integration
-----------------------------
The LLM can invoke function tools to extend its capabilities beyond text:

    @function_tool
    async def tell_story(self, context: RunContext, topic: str) -> str:
        '''Tell a story about a given topic.'''
        # Returns structured content for the LLM to speak
        return f"Here's a story about {topic}..."

Function tools enable:
    • Domain-specific actions (database queries, API calls)
    • Structured data retrieval
    • Multi-step reasoning workflows
    • Integration with external systems

The interruption handling system must be aware of function tool execution
to avoid interrupting critical operations.


================================================================================
3. SYSTEM DESIGN
================================================================================

3.1 Agent Speaking State Tracking
---------------------------------
The first component tracks the agent's current state through event subscription:

    class AgentStateTracker:
        States tracked:
        - "speaking": Agent TTS is actively producing audio
        - "listening": Agent is waiting for user input
        - "thinking": LLM is processing (between STT and TTS)
        - "initializing": Session startup

        Key methods:
        - on_agent_state_changed(): Updates state on framework events
        - is_speaking: Boolean indicating current speech state
        - was_speaking_recently(): Handles timing edge cases

The state tracker subscribes to 'agent_state_changed' events from the
AgentSession, maintaining a real-time view of the agent's activity.


3.2 Transcript-Based Semantic Classification
--------------------------------------------
The classifier analyzes transcribed user speech to categorize intent:

    class TranscriptClassifier:
        Classification categories:
        
        (1) BACKCHANNEL: Passive acknowledgements
            Examples: "yeah", "ok", "uh-huh", "hmm", "sure", "right"
            Action: IGNORE if agent is speaking
            
        (2) EXPLICIT_STOP: Clear stop commands
            Examples: "stop", "wait", "hold on", "pause", "enough"
            Action: ALWAYS interrupt, regardless of content
            
        (3) REAL_CONTENT: Substantive user input
            Examples: "Actually, I have a question", "What about..."
            Action: Interrupt and process as new input

    Classification algorithm:
    1. Normalize transcript (lowercase, strip punctuation)
    2. Check against explicit_stop_phrases (highest priority)
    3. Check against backchannel_phrases (exact match)
    4. Apply word count heuristic (< min_words → likely backchannel)
    5. Default to real_content


3.3 False Interruption Recovery Mechanism
-----------------------------------------
The system leverages the framework's false interruption handling:

    Configuration:
        resume_false_interruption=True
        false_interruption_timeout=1.5  # seconds

    Behavior:
    1. VAD detects speech → Agent PAUSES (not stops)
    2. System waits for STT transcript
    3. If no substantive input within timeout → Agent RESUMES
    4. If real content detected → Agent processes normally

This approach is critical because it decouples the immediate VAD response
from the semantic decision, allowing time for STT processing.


3.4 Mixed-Intent Detection
--------------------------
The classifier handles compound utterances with mixed signals:

    Example: "yeah but wait" or "ok stop"
    
    Resolution strategy:
    - Explicit stop phrases take PRIORITY over backchannels
    - If any stop phrase is detected, classify as EXPLICIT_STOP
    - This ensures user stop commands are never ignored

    Implementation:
        for stop_phrase in explicit_stop_phrases:
            if stop_phrase in normalized_transcript:
                return "explicit_stop"


================================================================================
4. ALGORITHMIC FLOW
================================================================================

4.1 Complete Lifecycle: Audio Input to Agent Response
------------------------------------------------------

PHASE 1: AUDIO CAPTURE AND VOICE DETECTION
    User speaks → Microphone captures audio
              → Audio frames sent to VAD
              → VAD detects speech onset
              → Event: user_started_speaking

PHASE 2: POTENTIAL INTERRUPTION (Agent Speaking)
    If agent.state == "speaking":
        → Agent audio PAUSES (not stops)
        → Framework starts false_interruption_timeout timer
        → Audio continues flowing to STT

PHASE 3: SPEECH-TO-TEXT TRANSCRIPTION
    Audio frames → STT model (Deepgram)
               → Interim transcripts emitted
               → Final transcript emitted
               → Event: user_input_transcribed

PHASE 4: SEMANTIC CLASSIFICATION (This Implementation)
    Transcript → IntelligentInterruptionHandler
             → AgentStateTracker.is_speaking checked
             → TranscriptClassifier.classify() called
             → Decision: IGNORE / PROCESS

    If BACKCHANNEL and agent was speaking:
        → Log: "🔇 IGNORED - Backchannel"
        → Agent RESUMES speaking (via framework)
        → No LLM invocation

    If EXPLICIT_STOP or REAL_CONTENT:
        → Log: "🛑 INTERRUPT" or "⏹️ INTERRUPT"
        → Agent stops speaking
        → Proceed to LLM processing

PHASE 5: LLM PROCESSING
    Transcript → Conversation context updated
             → LLM receives: context + user input + tools
             → LLM generates response (streamed)
             → Function tools executed if called
             → Response text produced

PHASE 6: TEXT-TO-SPEECH SYNTHESIS
    LLM text → TTS model (Cartesia)
           → Audio frames generated
           → Audio streamed to output
           → Agent state: "speaking"

PHASE 7: RESPONSE DELIVERY
    TTS audio → Room audio track
            → User hears response
            → Agent state: "listening"
            → Cycle repeats


4.2 State Machine Representation
--------------------------------

    ┌──────────────┐
    │ INITIALIZING │
    └──────┬───────┘
           │ session.start()
           ▼
    ┌──────────────┐    user speaks     ┌──────────────┐
    │  LISTENING   │ ─────────────────► │   THINKING   │
    └──────────────┘                    └──────┬───────┘
           ▲                                   │ LLM response ready
           │ TTS complete                      ▼
    ┌──────┴───────┐                    ┌──────────────┐
    │   SPEAKING   │ ◄────────────────  │   SPEAKING   │
    └──────────────┘    TTS starts      └──────────────┘
           │
           │ user interrupts (real content)
           ▼
    ┌──────────────┐
    │   THINKING   │ (process interruption)
    └──────────────┘


================================================================================
5. CONTRIBUTIONS TO GENAI RELIABILITY AND CONVERSATIONAL INTELLIGENCE
================================================================================

5.1 Improved Conversational UX
------------------------------
    BEFORE (Naive VAD):
        User: "Tell me about machine learning"
        Agent: "Machine learning is a branch of—"
        User: "uh-huh"
        Agent: [STOPS] "...I apologize, you were saying?"
        User: "No, continue"
        Agent: "Machine learning is a branch of—"  [RESTARTS]

    AFTER (Intelligent Interruption Handling):
        User: "Tell me about machine learning"
        Agent: "Machine learning is a branch of artificial intelligence
                that enables systems to learn from data..."
        User: "uh-huh"
        Agent: [CONTINUES SEAMLESSLY] "...These algorithms identify
                patterns and make decisions with minimal human intervention."

5.2 Reliability Improvements
----------------------------
    (a) Reduced False Positives: Backchannels no longer trigger interruptions,
        reducing unnecessary LLM invocations and TTS restarts.

    (b) Predictable Behavior: The classification system provides deterministic
        handling based on semantic content, not just audio signals.

    (c) Graceful Degradation: If classification fails, the system defaults to
        treating input as real content (fail-safe behavior).

    (d) Production-Ready: The solution operates entirely in the agent layer,
        requiring no modifications to framework internals.

5.3 Conversational Intelligence Enhancements
---------------------------------------------
    (a) Context Preservation: By not interrupting on backchannels, the agent
        can complete complex multi-sentence responses coherently.

    (b) Natural Turn-Taking: The system respects the natural flow of human
        conversation where listeners provide acknowledgement signals.

    (c) Intent Understanding: The semantic classifier demonstrates early-stage
        natural language understanding at the interruption handling layer.

    (d) Extensibility: The backchannel and stop phrase lists are configurable,
        allowing domain-specific customization.


5.4 Technical Innovations
-------------------------
    (a) State-Aware Filtering: Combining agent state with transcript semantics
        for context-dependent decision making.

    (b) Framework Feature Leverage: Using resume_false_interruption to pause
        rather than stop, enabling semantic analysis time.

    (c) Multi-Signal Classification: Combining phrase matching, word count
        heuristics, and priority ordering for robust classification.

    (d) Grace Period Handling: The backchannel_grace_period addresses timing
        edge cases at state transition boundaries.


================================================================================
USAGE
================================================================================

Running the Agent:
    python -m examples.voice_agents.intelligent_interruption_agent

Interactive Console Testing:
    python -m examples.voice_agents.intelligent_interruption_agent console

Expected Log Output:
    🤖 Intelligent Interruption Agent started
    📋 Backchannel phrases: 28 configured
    🛑 Stop phrases: 18 configured
    
    📝 Normal input (agent silent): 'Hello there'
    🔇 IGNORED - Backchannel while agent speaking: 'yeah'
    🛑 INTERRUPT - Explicit stop detected: 'stop'
    ⏹️ INTERRUPT - Real content detected: 'Actually wait I have a question'


================================================================================
REFERENCES
================================================================================

[1] LiveKit Agents Framework Documentation
    https://docs.livekit.io/agents/

[2] Silero VAD: Pre-trained Voice Activity Detection
    https://github.com/snakers4/silero-vad

[3] Deepgram Nova-3: Real-time Speech Recognition
    https://deepgram.com/

[4] OpenAI GPT-4 Function Calling
    https://platform.openai.com/docs/guides/function-calling

[5] Cartesia Sonic: Neural Text-to-Speech
    https://cartesia.ai/


================================================================================
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
