# Setup and Testing Guide

## Prerequisites

This agent requires an **OpenAI API key** to run evaluations. The agent uses GPT-4o-mini to evaluate agent answers against reference answers.

## Setting Up the OpenAI API Key

### Option 1: Environment Variable (Recommended for local development)

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option 2: Using an `.env` file

1. Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

2. **Important:** Add `.env` to your `.gitignore` to avoid committing your API key:
```bash
echo ".env" >> .gitignore
```

### Option 3: Docker with Environment Variable

Pass the API key when running the Docker container:

```bash
# Method 1: Direct pass
docker run -e OPENAI_API_KEY="your-api-key-here" -p 9009:9009 my-agent

# Method 2: From your shell environment
export OPENAI_API_KEY="your-api-key-here"
docker run -e OPENAI_API_KEY -p 9009:9009 my-agent

# Method 3: Using an env file
docker run --env-file .env -p 9009:9009 my-agent
```

## Running the Agent

### Locally with uv

```bash
# Install dependencies
uv sync

# Set your API key (if not already set)
export OPENAI_API_KEY="your-api-key-here"

# Run the server
uv run src/server.py
```

The agent will start on `http://localhost:9009`

### With Docker

```bash
# Build the image
docker build -t my-agent .

# Run with API key
docker run -e OPENAI_API_KEY="your-api-key-here" -p 9009:9009 my-agent
```

## Running Tests

The test suite validates A2A protocol conformance to ensure the agent properly implements the Agent-to-Agent (A2A) protocol.

### What the Tests Check

#### 1. `test_agent_card` - Agent Card Validation
Validates that the agent exposes a properly formatted agent card at `/.well-known/agent-card.json`:
- **Required fields**: name, description, url, version, capabilities, defaultInputModes, defaultOutputModes, skills
- **URL format**: Must be an absolute URL (http:// or https://)
- **Capabilities**: Must be a valid object
- **Input/Output modes**: Must be arrays of strings
- **Skills**: Must be a non-empty array with at least one skill

#### 2. `test_message[True]` - Streaming Message Handling
Tests the agent's streaming response capability:
- Sends a simple "Hello" message to the agent
- Validates that the agent responds using Server-Sent Events (SSE)
- Checks that all response events conform to A2A message format
- Validates message structure (non-empty parts array, correct role)
- Ensures task and status updates have required fields

#### 3. `test_message[False]` - Non-Streaming Message Handling
Tests the agent's synchronous response capability:
- Sends a simple "Hello" message to the agent
- Validates that the agent responds with a complete response
- Checks the same A2A format validation as streaming
- Ensures the agent can handle both modes of communication

### Running the Tests

1. **Start the agent** (in one terminal):
```bash
export OPENAI_API_KEY="your-api-key-here"
uv run src/server.py
```

2. **Run the tests** (in another terminal):
```bash
# Install test dependencies
uv sync --extra test

# Set API key in this terminal too
export OPENAI_API_KEY="your-api-key-here"

# Run tests
uv run pytest --agent-url http://localhost:9009
```

### Expected Output

```
platform linux -- Python 3.14.0, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/tsljgj/private/AQA-green-agent
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.12.0
collected 3 items

tests/test_agent.py ...                                                       [100%]

================================= 3 passed in 0.39s =================================
```

## Troubleshooting

### Error: "The api_key client option must be set"

**Problem**: The OpenAI API key is not available to the agent.

**Solution**:
- If running locally: `export OPENAI_API_KEY="your-key"`
- If running in Docker: Use `-e OPENAI_API_KEY="your-key"` flag
- Make sure the key is set in the same terminal/environment where the agent is running

### Error: "peer closed connection without sending complete message body"

**Problem**: The agent crashed (usually due to missing API key) during a request.

**Solution**: Check the agent logs for the actual error. Usually this means the API key wasn't passed to the container or server process.

### Tests timeout or hang

**Problem**: The agent is not running or is running on a different port.

**Solution**:
- Verify the agent is running: `curl http://localhost:9009/.well-known/agent-card.json`
- Check if the port matches: Use `--agent-url` flag with the correct URL
- Ensure no firewall is blocking port 9009

## CI/CD Integration

For GitHub Actions or other CI/CD systems:

1. Add `OPENAI_API_KEY` as a secret in your repository settings
2. The workflow will automatically use it during test runs
3. See [.github/workflows/test-and-publish.yml](.github/workflows/test-and-publish.yml) for the CI configuration
