"""Test green agent + purple agent interaction locally."""
import asyncio
import json
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart
import httpx
from uuid import uuid4


async def run_assessment(
    purple_agent_url: str = "http://localhost:9010",
    green_agent_url: str = "http://localhost:9009",
    num_tasks: int = 3,
    difficulties: list[str] = None,
    seed: int = 42
):
    """Run a test assessment.

    Args:
        purple_agent_url: URL of the purple agent to test
        green_agent_url: URL of the green agent (AQA benchmark)
        num_tasks: Number of questions to ask
        difficulties: List of difficulty levels (easy, medium, hard)
        seed: Random seed for reproducible question sampling
    """
    if difficulties is None:
        difficulties = ["easy", "medium"]

    # Assessment request
    request = {
        "participants": {"agent": purple_agent_url},
        "config": {
            "num_tasks": num_tasks,
            "difficulty": difficulties,
            "seed": seed
        }
    }

    # Connect to green agent
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resolver = A2ACardResolver(httpx_client=client, base_url=green_agent_url)
            card = await resolver.get_agent_card()
            print(f"✅ Connected to green agent: {card.name}")
        except Exception as e:
            print(f"❌ Failed to connect to green agent at {green_agent_url}")
            print(f"   Error: {e}")
            print(f"   Make sure the green agent is running!")
            return

        try:
            # Check purple agent is reachable
            purple_resolver = A2ACardResolver(httpx_client=client, base_url=purple_agent_url)
            purple_card = await purple_resolver.get_agent_card()
            print(f"✅ Purple agent reachable: {purple_card.name}")
        except Exception as e:
            print(f"❌ Failed to connect to purple agent at {purple_agent_url}")
            print(f"   Error: {e}")
            print(f"   Make sure the purple agent is running!")
            return

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

        print("\n" + "="*70)
        print("🚀 Starting AQA Assessment")
        print("="*70)
        print(f"   Purple Agent: {purple_agent_url}")
        print(f"   Questions: {num_tasks}")
        print(f"   Difficulties: {', '.join(difficulties)}")
        print(f"   Seed: {seed}")
        print("="*70 + "\n")

        try:
            async for event in a2a_client.send_message(msg):
                # Print status updates
                if hasattr(event, 'status') and hasattr(event.status, 'message'):
                    if event.status.message:
                        from a2a.utils import get_message_text
                        status_text = get_message_text(event.status.message)
                        print(f"📊 {status_text}")

                # Print final results
                if hasattr(event, 'artifacts') and event.artifacts:
                    print("\n" + "="*70)
                    print("✅ Assessment Complete!")
                    print("="*70)

                    for artifact in event.artifacts:
                        for part in artifact.parts:
                            if hasattr(part.root, 'text'):
                                print(part.root.text)
                            elif hasattr(part.root, 'data'):
                                result = part.root.data
                                agg = result.get('aggregate', {})
                                items = result.get('items', [])

                                print(f"\n📈 Overall Results:")
                                print(f"   Pass Rate: {agg.get('pass_rate', 0):.1%} ({agg.get('correct', 0)}/{agg.get('total_tasks', 0)})")
                                print(f"   Average Score: {agg.get('avg_score', 0):.3f}")
                                print(f"   Total Time: {agg.get('total_time_ms', 0)}ms")
                                print(f"   Average Latency: {agg.get('avg_latency_ms', 0)}ms")

                                by_diff = agg.get('by_difficulty', {})
                                if by_diff:
                                    print(f"\n📊 By Difficulty:")
                                    for diff, stats in by_diff.items():
                                        print(f"   {diff.capitalize()}: {stats['correct']}/{stats['total']} ({stats['pass_rate']:.1%}), avg score: {stats['avg_score']:.3f}")

                                if items:
                                    print(f"\n📝 Individual Question Results:")
                                    for item in items:
                                        status = "✅" if item['correct'] else "❌"
                                        print(f"\n   {status} Question: {item['question'][:80]}...")
                                        print(f"      Difficulty: {item['difficulty']}")
                                        print(f"      Reference: {item['reference_answer'][:100]}")
                                        print(f"      Agent Answer: {item['agent_answer'][:100]}")
                                        print(f"      Score: {item['score']:.2f} | Latency: {item['latency_ms']}ms")
                                        print(f"      Reasoning: {item['evaluation_reasoning']}")

                    print("="*70 + "\n")

        except Exception as e:
            print(f"\n❌ Assessment failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test AQA green agent with a purple agent")
    parser.add_argument("--purple-url", default="http://localhost:9010", help="Purple agent URL")
    parser.add_argument("--green-url", default="http://localhost:9009", help="Green agent URL")
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of questions")
    parser.add_argument("--difficulty", nargs="+", default=["easy", "medium"], help="Difficulty levels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    asyncio.run(run_assessment(
        purple_agent_url=args.purple_url,
        green_agent_url=args.green_url,
        num_tasks=args.num_tasks,
        difficulties=args.difficulty,
        seed=args.seed
    ))
