"""Simple QA agent that answers questions using OpenAI."""
import os
from openai import AsyncOpenAI
from a2a.server.tasks import EventQueue
from a2a.types import Message
from a2a.utils import get_message_text, new_agent_text_message


class Agent:
    """A simple QA agent that answers questions."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    async def run(self, message: Message, queue: EventQueue) -> None:
        """Answer a question.

        Args:
            message: The incoming question message
            queue: EventQueue for sending responses
        """
        question = get_message_text(message)

        try:
            # Use OpenAI to answer the question
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions concisely and accurately."
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.0,
                max_tokens=500
            )

            answer = response.choices[0].message.content or "I don't know."

        except Exception as e:
            answer = f"Error: {str(e)}"

        # Send the answer back
        await queue.enqueue_message(new_agent_text_message(answer))
