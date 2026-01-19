# Intelligent Interruption Handling for LiveKit Voice Agents

## My Approach

When I started working on this assignment, I noticed something annoying about the default behavior - the agent would stop talking every time I said "yeah" or "uh-huh" while listening to it. In real conversations, these are just acknowledgements, not requests to stop!

So I dug into how LiveKit's interruption system works and came up with a solution that feels much more natural.

## The Problem I Solved

The default VAD (Voice Activity Detection) is too aggressive. It treats *any* speech as an interruption. But in natural conversation:

- **"yeah", "ok", "hmm"** → Just nodding along, not trying to interrupt
- **"stop", "wait", "hold on"** → Actually wants the agent to stop

My solution distinguishes between these two cases based on **what** the user says and **when** they say it.

## How It Works

I track whether the agent is currently speaking. When user speech is detected:

1. If agent is **silent** → process everything normally
2. If agent is **speaking** → classify the transcript:
   - Backchannel phrase? → Ignore it, let agent continue
   - Stop command? → Interrupt immediately
   - Real content? → Interrupt and respond

```
User says "yeah" while agent talking  →  🔇 Ignored, agent continues seamlessly
User says "yeah" while agent silent   →  📝 Processed normally
User says "stop" while agent talking  →  🛑 Agent stops immediately
```

## Key Design Decisions

### 1. Using Framework Features

Instead of fighting the framework, I leveraged `resume_false_interruption=True`. This makes the agent **pause** instead of hard-stop when it detects speech. That gives me time to analyze the transcript before deciding.

### 2. Configurable Phrase Lists

I made the backchannel and stop phrases configurable so they can be tuned for different use cases:

```python
backchannel_phrases = {"yeah", "ok", "hmm", "uh-huh", "sure", "right", ...}
explicit_stop_phrases = {"stop", "wait", "no", "hold on", "pause", ...}
```

### 3. Word Count Heuristic

Single words during agent speech are almost always backchannels. I use `min_words_for_interruption=2` as an additional filter.

### 4. Grace Period for Timing

There's a small window after the agent stops where backchannels are still filtered out. This handles edge cases where the user's "yeah" overlaps with the agent finishing.

## Files

Everything is in one file to keep it simple:

- `examples/voice_agents/intelligent_interruption_agent.py`

Components inside:
- `InterruptionConfig` - Configuration dataclass
- `AgentStateTracker` - Tracks speaking state from events
- `TranscriptClassifier` - Categorizes user input
- `IntelligentInterruptionHandler` - Decision logic
- `IntelligentInterruptionAgent` - Main agent class

## Running It

```bash
# Install dependencies first
pip install -e ./livekit-agents
pip install -e ./livekit-plugins/livekit-plugins-silero
pip install -e ./livekit-plugins/livekit-plugins-turn-detector
pip install python-dotenv

# Set up your .env with API keys

# Run in console mode for testing
python examples/voice_agents/intelligent_interruption_agent.py console
```

## Testing

I tested these scenarios:

| Test | What I Did | Expected | Result |
|------|------------|----------|--------|
| Backchannel during speech | Asked for a story, said "yeah" mid-way | Agent keeps talking | ✓ |
| Backchannel when silent | Waited for agent to finish, said "yeah" | Agent responds | ✓ |
| Stop command | Asked for story, said "stop" | Agent stops immediately | ✓ |
| Normal conversation | Said "hello" | Normal greeting | ✓ |

## What I Learned

- VAD fires *before* STT produces a transcript - that's the root cause of false interruptions
- The `agent_state_changed` event is perfect for tracking speaking state
- LiveKit's `resume_false_interruption` feature is underrated - it saved me from having to hack around the framework

## Tradeoffs

- There's a slight delay before real interruptions are recognized (waiting for STT)
- The phrase lists need tuning for different languages/contexts
- Very fast speakers might get their backchannels cut off before the full word is recognized

## Future Improvements

If I had more time, I'd add:
- Confidence scoring from STT to better identify backchannels
- Learning from user patterns ("this user says 'right' a lot")
- Integration with turn detection models for smarter decisions

---

Built by Prabh for the LiveKit Agents assignment.
