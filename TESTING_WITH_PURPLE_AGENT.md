# Testing Your Green Agent with a Purple Agent

## Understanding the System

### Agent Roles

**Green Agent (Your AQA Benchmark):**
- Acts as the evaluator/benchmark
- Orchestrates the assessment by sending questions to purple agents
- Grades answers using LLM-as-judge (GPT-4o-mini)
- Produces final results: pass rate, scores, latency metrics

**Purple Agent (Agent Under Test):**
- The AI agent being evaluated
- Receives questions via A2A protocol
- Responds with answers
- Can be any A2A-compatible agent (coding assistants, QA bots, research agents, etc.)

### How They Interact

```
┌─────────────────┐                    ┌──────────────────┐
│  Green Agent    │                    │  Purple Agent    │
│  (AQA Benchmark)│                    │  (Being Tested)  │
└────────┬────────┘                    └────────┬─────────┘
         │                                      │
         │  1. Send Question                    │
         │  "What is the capital of France?"    │
         │─────────────────────────────────────>│
         │                                      │
         │  2. Receive Answer                   │
         │  "Paris"                             │
         │<─────────────────────────────────────│
         │                                      │
         │  3. Evaluate (using GPT-4o-mini)     │
         │     Compare with reference answer    │
         │     Calculate score                  │
         │                                      │
         │  4. Repeat for all questions         │
         │                                      │
         │  5. Generate final report            │
         │     - Pass rate: 80%                 │
         │     - Avg score: 0.85                │
         │     - By difficulty breakdown        │
         │                                      │
```

## Local Testing

### Option 1: Manual Testing (Recommended for Development)

Test the interaction step-by-step to understand how it works.

#### Step 1: Start the Purple Agent

In terminal 1:
```bash
cd test-purple-agent
export OPENAI_API_KEY="your-key"
uv sync
uv run src/server.py
```

The purple agent should start on `http://localhost:9010`

#### Step 2: Start the Green Agent

In terminal 2:
```bash
export OPENAI_API_KEY="your-key"
uv run src/server.py
```

The green agent should start on `http://localhost:9009`

#### Step 3: Send an Assessment Request

In terminal 3, create a test script or use the A2A client:

```bash
export OPENAI_API_KEY="your-key"

# Create a test request file
cat > test_request.json << 'EOF'
{
  "participants": {
    "agent": "http://localhost:9010"
  },
  "config": {
    "num_tasks": 5,
    "difficulty": ["easy", "medium"],
    "seed": 42
  }
}
EOF

# Send the request to the green agent
python3 << 'PYTHON'
import asyncio
import json
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
import httpx
from uuid import uuid4

async def test_assessment():
    # Load request
    with open("test_request.json") as f:
        request_data = json.load(f)

    # Connect to green agent
    async with httpx.AsyncClient(timeout=600) as client:
        resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:9009")
        card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=client, streaming=True)
        factory = ClientFactory(config)
        a2a_client = factory.create(card)

        # Send assessment request
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=json.dumps(request_data)))],
            message_id=uuid4().hex,
        )

        print("Starting assessment...")
        async for event in a2a_client.send_message(msg):
            print(f"Event: {event}")

asyncio.run(test_assessment())
PYTHON
```

#### What to Expect

You should see:
1. Green agent loading questions from the dataset
2. Green agent sending questions to purple agent (watch purple agent logs)
3. Purple agent responding with answers
4. Green agent evaluating each answer
5. Final assessment results with pass rate and scores

### Option 2: Using Python Test Script

Create a simpler test script:

```bash
cat > test_interaction.py << 'PYTHON'
"""Test green agent + purple agent interaction."""
import asyncio
import json
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
import httpx
from uuid import uuid4


async def run_assessment():
    """Run a test assessment."""
    # Assessment request
    request = {
        "participants": {"agent": "http://localhost:9010"},
        "config": {
            "num_tasks": 3,
            "difficulty": ["easy"],
            "seed": 42
        }
    }

    # Connect to green agent
    async with httpx.AsyncClient(timeout=600) as client:
        resolver = A2ACardResolver(httpx_client=client, base_url="http://localhost:9009")
        card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=client, streaming=True)
        factory = ClientFactory(config)
        a2a_client = factory.create(card)

        # Send request
        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=json.dumps(request)))],
            message_id=uuid4().hex,
        )

        print("🚀 Starting AQA assessment...")
        print(f"   Testing purple agent: {request['participants']['agent']}")
        print(f"   Number of questions: {request['config']['num_tasks']}")
        print()

        async for event in a2a_client.send_message(msg):
            # Print status updates
            if hasattr(event, 'status') and hasattr(event.status, 'message'):
                if event.status.message:
                    from a2a.utils import get_message_text
                    status_text = get_message_text(event.status.message)
                    print(f"📊 {status_text}")

            # Print final results
            if hasattr(event, 'artifacts') and event.artifacts:
                print("\n✅ Assessment Complete!")
                print("=" * 60)
                for artifact in event.artifacts:
                    for part in artifact.parts:
                        if hasattr(part.root, 'text'):
                            print(part.root.text)
                        elif hasattr(part.root, 'data'):
                            result = part.root.data
                            agg = result.get('aggregate', {})
                            print(f"\n📈 Results:")
                            print(f"   Pass Rate: {agg.get('pass_rate', 0):.1%}")
                            print(f"   Average Score: {agg.get('avg_score', 0):.3f}")
                            print(f"   Total Time: {agg.get('total_time_ms', 0)}ms")
                            print(f"   Avg Latency: {agg.get('avg_latency_ms', 0)}ms")
                print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_assessment())
PYTHON

# Run it
python test_interaction.py
```

### Option 3: Docker Compose (Production-like Testing)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  green-agent:
    build: .
    ports:
      - "9009:9009"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    networks:
      - agent-network

  purple-agent:
    build: ./test-purple-agent
    ports:
      - "9010:9010"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    networks:
      - agent-network

networks:
  agent-network:
    driver: bridge
```

Then run:
```bash
export OPENAI_API_KEY="your-key"
docker-compose up
```

## Submission to AgentBeats

Once your green agent works locally, you need to:

### 1. Register Your Green Agent

Submit via the [AgentBeats submission form](https://docs.google.com/forms/d/e/1FAIpQLSdtqxWcGl2Qg5RPuNF2O3_N07uD0HMJpWBCwZWZbD3dxTuWmg/viewform) by **January 15, 2026**.

You'll receive an `agentbeats_id` for your agent.

### 2. Publish Docker Image

Push your Docker image to a public registry (GitHub Container Registry recommended):

```bash
# Build and tag
docker build -t ghcr.io/YOUR_USERNAME/aqa-green-agent:latest .

# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Push
docker push ghcr.io/YOUR_USERNAME/aqa-green-agent:latest
```

### 3. Update Agent Card URL

Make sure your agent card URL points to your publicly accessible deployment (or use AgentBeats hosting).

### 4. Set Up CI/CD (Optional)

The included `.github/workflows/test-and-publish.yml` automates:
- Running tests on every push
- Building and publishing Docker images
- Tagging versions

Add your `OPENAI_API_KEY` to GitHub Secrets for CI tests.

## What Happens on AgentBeats

1. **Purple agents** (various AI agents) register on the platform
2. **Your green agent** evaluates them by:
   - Sending questions from your dataset
   - Collecting their answers
   - Grading with LLM-as-judge
   - Reporting scores
3. **Leaderboard** displays results comparing all purple agents on your benchmark
4. The community can see which agents perform best on AQA tasks

## Testing Checklist

Before submitting, verify:

- [ ] Green agent passes A2A conformance tests (`pytest --agent-url`)
- [ ] Green agent can successfully assess a purple agent locally
- [ ] Assessment produces valid results with pass rate and scores
- [ ] Docker image builds and runs correctly
- [ ] Environment variables (OPENAI_API_KEY) work in Docker
- [ ] Agent card is properly configured with your skill description
- [ ] Dataset is appropriate and well-balanced
- [ ] LLM evaluator produces reasonable scores

## Resources

- **AgentBeats Documentation**: https://docs.agentbeats.dev/
- **A2A Protocol**: https://a2a-protocol.org/latest/
- **AgentBeats Tutorial**: https://github.com/RDI-Foundation/agentbeats-tutorial
- **Submission Form**: https://docs.google.com/forms/d/e/1FAIpQLSdtqxWcGl2Qg5RPuNF2O3_N07uD0HMJpWBCwZWZbD3dxTuWmg/viewform

## Competition Deadline

**Phase 1 (Green Agent) Deadline**: January 15, 2026 at 11:59pm PT
