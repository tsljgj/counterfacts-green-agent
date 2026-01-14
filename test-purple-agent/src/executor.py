"""A2A executor for the simple QA agent."""
from a2a.server.executors import AgentExecutor
from a2a.server.tasks import EventQueue
from a2a.types import MessageSendParams

from agent import Agent


class Executor(AgentExecutor):
    """Executor that runs the simple QA agent."""

    async def execute(
        self,
        request: MessageSendParams,
        queue: EventQueue,
    ):
        """Execute the agent on an incoming request."""
        agent = Agent()
        await agent.run(request.message, queue)
