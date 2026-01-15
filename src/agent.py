import time
from pathlib import Path
from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from dataset import QADataset, Question, DIFFICULTY_ORDER, SUBJECT_CATEGORIES
from evaluator import LLMEvaluator, EvaluationResult


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class QuestionResult(BaseModel):
    """Result for a single question."""
    qid: str
    difficulty: str
    category: str  # "web" or "science"
    question: str
    reference_answer: str
    agent_answer: str
    correct: bool
    score: float
    latency_ms: int
    evaluation_reasoning: str
    domain: str | None = None


# Weights for computing overall weighted score (harder questions worth more)
DIFFICULTY_WEIGHTS = {
    "easy": 1.0,
    "medium": 2.0,
    "hard": 3.0,
    "expert": 4.0
}


class AggregateResults(BaseModel):
    """Aggregate metrics across all questions."""
    total_tasks: int
    correct: int
    pass_rate: float
    avg_score: float
    weighted_score: float  # Weighted average score across difficulty levels
    total_time_ms: int
    avg_latency_ms: int
    # Individual level accuracies for leaderboard (always present, 0.0 if no questions)
    easy_accuracy: float
    medium_accuracy: float
    hard_accuracy: float
    expert_accuracy: float
    # Subject category accuracies for leaderboard
    web_accuracy: float
    science_accuracy: float
    # Cross-tabulation: category x difficulty (e.g., web_easy_accuracy)
    web_easy_accuracy: float
    web_medium_accuracy: float
    web_hard_accuracy: float
    web_expert_accuracy: float
    science_easy_accuracy: float
    science_medium_accuracy: float
    science_hard_accuracy: float
    science_expert_accuracy: float
    by_difficulty: dict[str, dict[str, Any]]
    by_category: dict[str, dict[str, Any]]  # web/science breakdown
    by_category_difficulty: dict[str, dict[str, dict[str, Any]]]  # category -> difficulty -> stats


class AssessmentResult(BaseModel):
    """Complete assessment result."""
    assessment_id: str
    config: dict[str, Any]
    items: list[QuestionResult]
    aggregate: AggregateResults
    metadata: dict[str, Any]


def generate_accuracy_graph(by_difficulty: dict[str, dict[str, Any]], bar_width: int = 30) -> str:
    """Generate an ASCII bar chart showing accuracy by difficulty level.

    Args:
        by_difficulty: Dictionary mapping difficulty -> stats (with pass_rate)
        bar_width: Width of the bar chart in characters

    Returns:
        ASCII art string representation of the graph
    """
    lines = []
    lines.append("")
    lines.append("  Accuracy by Difficulty Level")
    lines.append("  " + "=" * (bar_width + 20))
    lines.append("")

    # Sort difficulties in proper order
    ordered_diffs = [d for d in DIFFICULTY_ORDER if d in by_difficulty]

    if not ordered_diffs:
        return "  No difficulty data available"

    for diff in ordered_diffs:
        stats = by_difficulty[diff]
        pass_rate = stats.get("pass_rate", 0.0)
        correct = stats.get("correct", 0)
        total = stats.get("total", 0)

        # Create the bar
        filled_width = int(pass_rate * bar_width)
        empty_width = bar_width - filled_width

        # Use different characters for visual appeal
        bar = "\u2588" * filled_width + "\u2591" * empty_width

        # Format the label (pad to 8 chars)
        label = f"{diff.capitalize():8s}"

        # Format percentage and fraction
        pct_str = f"{pass_rate:6.1%}"
        frac_str = f"({correct}/{total})"

        lines.append(f"  {label} |{bar}| {pct_str} {frac_str}")

    lines.append("")
    lines.append("  " + "=" * (bar_width + 20))

    # Add legend
    lines.append(f"  Legend: \u2588 = correct, \u2591 = incorrect")
    lines.append("")

    return "\n".join(lines)


class Agent:
    """AQA Green Agent - QA benchmark evaluator."""

    # Required participant roles
    required_roles: list[str] = ["agent"]  # The purple agent being tested

    # Required config keys
    required_config_keys: list[str] = ["num_tasks"]

    def __init__(self):
        self.messenger = Messenger()

        # Load dataset
        dataset_path = Path(__file__).parent.parent / "data" / "data_ready"
        self.dataset = QADataset(dataset_path)

        # Initialize evaluator (will use OPENAI_API_KEY from environment)
        self.evaluator = LLMEvaluator()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the assessment request."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Validate num_tasks
        num_tasks = request.config.get("num_tasks", 0)
        if not isinstance(num_tasks, int) or num_tasks <= 0:
            return False, "num_tasks must be a positive integer"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Run the AQA benchmark assessment.

        Args:
            message: The incoming assessment request message
            updater: TaskUpdater for reporting progress and results
        """
        input_text = get_message_text(message)

        # Parse and validate request
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Extract config parameters
        config = request.config
        num_tasks = config["num_tasks"]
        difficulties = config.get("difficulty", ["easy", "medium", "hard", "expert"])
        domains = config.get("domains", None)
        seed = config.get("seed", None)
        timeout_per_question = config.get("timeout_per_question", 30)
        evaluator_model = config.get("evaluator_model", "gpt-4o-mini")

        # Get purple agent URL
        agent_url = str(request.participants["agent"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Loading {num_tasks} questions from AQA dataset...")
        )

        # Sample questions
        try:
            questions = self.dataset.sample_questions(
                num_questions=num_tasks,
                difficulties=difficulties,
                domains=domains,
                seed=seed
            )
        except ValueError as e:
            await updater.reject(new_agent_text_message(f"Dataset error: {e}"))
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Sampled {len(questions)} questions. Starting assessment...")
        )

        # Run assessment
        results: list[QuestionResult] = []
        total_time_ms = 0

        for i, question in enumerate(questions, 1):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Question {i}/{num_tasks}: {question.question[:50]}...")
            )

            # Ask purple agent
            start_time = time.time()
            try:
                agent_answer = await self.messenger.talk_to_agent(
                    message=question.question,
                    url=agent_url
                )
            except Exception as e:
                agent_answer = f"[Error: {str(e)}]"

            latency_ms = int((time.time() - start_time) * 1000)
            total_time_ms += latency_ms

            # Evaluate answer
            try:
                eval_result: EvaluationResult = await self.evaluator.evaluate(
                    question=question.question,
                    reference_answer=question.reference_answer,
                    agent_answer=agent_answer
                )
            except Exception as e:
                # Fallback evaluation
                eval_result = EvaluationResult(
                    correct=False,
                    score=0.0,
                    reasoning=f"Evaluation failed: {str(e)}"
                )

            # Record result
            results.append(QuestionResult(
                qid=question.qid,
                difficulty=question.difficulty,
                category=question.category,
                question=question.question,
                reference_answer=question.reference_answer,
                agent_answer=agent_answer,
                correct=eval_result.correct,
                score=eval_result.score,
                latency_ms=latency_ms,
                evaluation_reasoning=eval_result.reasoning,
                domain=question.domain
            ))

        # Calculate aggregate metrics
        correct_count = sum(1 for r in results if r.correct)
        pass_rate = correct_count / len(results) if results else 0.0
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        avg_latency_ms = total_time_ms // len(results) if results else 0

        # Calculate by-difficulty breakdown (ALL levels always present)
        by_difficulty: dict[str, dict[str, Any]] = {}

        # Process ALL difficulty levels (even if no questions for that level)
        for difficulty in DIFFICULTY_ORDER:
            diff_results = [r for r in results if r.difficulty == difficulty]
            diff_correct = sum(1 for r in diff_results if r.correct)
            diff_total = len(diff_results)
            by_difficulty[difficulty] = {
                "correct": diff_correct,
                "total": diff_total,
                "pass_rate": diff_correct / diff_total if diff_total > 0 else 0.0,
                "avg_score": sum(r.score for r in diff_results) / diff_total if diff_total > 0 else 0.0
            }

        # Compute weighted score across difficulty levels
        # Formula: sum(weight * pass_rate * num_questions) / sum(weight * num_questions)
        weighted_numerator = 0.0
        weighted_denominator = 0.0
        for difficulty in DIFFICULTY_ORDER:
            weight = DIFFICULTY_WEIGHTS[difficulty]
            stats = by_difficulty[difficulty]
            total = stats["total"]
            if total > 0:
                weighted_numerator += weight * stats["pass_rate"] * total
                weighted_denominator += weight * total

        weighted_score = weighted_numerator / weighted_denominator if weighted_denominator > 0 else 0.0

        # Extract individual level accuracies for easy leaderboard access
        easy_accuracy = by_difficulty["easy"]["pass_rate"]
        medium_accuracy = by_difficulty["medium"]["pass_rate"]
        hard_accuracy = by_difficulty["hard"]["pass_rate"]
        expert_accuracy = by_difficulty["expert"]["pass_rate"]

        # Calculate by-category breakdown (web vs science)
        by_category: dict[str, dict[str, Any]] = {}
        for category in SUBJECT_CATEGORIES:
            cat_results = [r for r in results if r.category == category]
            cat_correct = sum(1 for r in cat_results if r.correct)
            cat_total = len(cat_results)
            by_category[category] = {
                "correct": cat_correct,
                "total": cat_total,
                "pass_rate": cat_correct / cat_total if cat_total > 0 else 0.0,
                "avg_score": sum(r.score for r in cat_results) / cat_total if cat_total > 0 else 0.0
            }

        # Extract category accuracies for leaderboard
        web_accuracy = by_category["web"]["pass_rate"]
        science_accuracy = by_category["science"]["pass_rate"]

        # Calculate cross-tabulation: category x difficulty
        by_category_difficulty: dict[str, dict[str, dict[str, Any]]] = {}
        for category in SUBJECT_CATEGORIES:
            by_category_difficulty[category] = {}
            for difficulty in DIFFICULTY_ORDER:
                cat_diff_results = [r for r in results if r.category == category and r.difficulty == difficulty]
                cat_diff_correct = sum(1 for r in cat_diff_results if r.correct)
                cat_diff_total = len(cat_diff_results)
                by_category_difficulty[category][difficulty] = {
                    "correct": cat_diff_correct,
                    "total": cat_diff_total,
                    "pass_rate": cat_diff_correct / cat_diff_total if cat_diff_total > 0 else 0.0,
                    "avg_score": sum(r.score for r in cat_diff_results) / cat_diff_total if cat_diff_total > 0 else 0.0
                }

        # Extract cross-tabulation accuracies for leaderboard
        web_easy_accuracy = by_category_difficulty["web"]["easy"]["pass_rate"]
        web_medium_accuracy = by_category_difficulty["web"]["medium"]["pass_rate"]
        web_hard_accuracy = by_category_difficulty["web"]["hard"]["pass_rate"]
        web_expert_accuracy = by_category_difficulty["web"]["expert"]["pass_rate"]
        science_easy_accuracy = by_category_difficulty["science"]["easy"]["pass_rate"]
        science_medium_accuracy = by_category_difficulty["science"]["medium"]["pass_rate"]
        science_hard_accuracy = by_category_difficulty["science"]["hard"]["pass_rate"]
        science_expert_accuracy = by_category_difficulty["science"]["expert"]["pass_rate"]

        aggregate = AggregateResults(
            total_tasks=len(results),
            correct=correct_count,
            pass_rate=pass_rate,
            avg_score=avg_score,
            weighted_score=weighted_score,
            total_time_ms=total_time_ms,
            avg_latency_ms=avg_latency_ms,
            easy_accuracy=easy_accuracy,
            medium_accuracy=medium_accuracy,
            hard_accuracy=hard_accuracy,
            expert_accuracy=expert_accuracy,
            web_accuracy=web_accuracy,
            science_accuracy=science_accuracy,
            web_easy_accuracy=web_easy_accuracy,
            web_medium_accuracy=web_medium_accuracy,
            web_hard_accuracy=web_hard_accuracy,
            web_expert_accuracy=web_expert_accuracy,
            science_easy_accuracy=science_easy_accuracy,
            science_medium_accuracy=science_medium_accuracy,
            science_hard_accuracy=science_hard_accuracy,
            science_expert_accuracy=science_expert_accuracy,
            by_difficulty=by_difficulty,
            by_category=by_category,
            by_category_difficulty=by_category_difficulty
        )

        # Create final assessment result
        assessment_result = AssessmentResult(
            assessment_id=f"aqa-{int(time.time())}",
            config=config,
            items=results,
            aggregate=aggregate,
            metadata={
                "purple_agent_url": agent_url,
                "dataset_size": len(self.dataset),
                "evaluator_model": evaluator_model,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        )

        # Report results
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Assessment complete! Pass rate: {pass_rate:.1%} ({correct_count}/{len(results)})"
            )
        )

        # Generate fancy ASCII graph
        accuracy_graph = generate_accuracy_graph(by_difficulty)

        # Build category stats strings
        web_stats = by_category["web"]
        science_stats = by_category["science"]

        # Helper to format stats
        def fmt_stats(s: dict) -> str:
            return f"{s['correct']}/{s['total']}" if s['total'] > 0 else "--/--"

        # Build formatted results text
        results_text = (
            "=" * 70 + "\n"
            "                    AQA BENCHMARK RESULTS\n"
            "=" * 70 + "\n\n"
            f"  Overall Performance\n"
            f"  -------------------\n"
            f"  Pass Rate:       {pass_rate:6.1%} ({correct_count}/{len(results)})\n"
            f"  Weighted Score:  {weighted_score:6.3f}  (difficulty-adjusted, for ranking)\n"
            f"  Average Score:   {avg_score:6.3f}\n"
            f"  Total Time:      {total_time_ms:,}ms\n"
            f"  Average Latency: {avg_latency_ms:,}ms\n"
            f"{accuracy_graph}\n"
            f"  Accuracy by Difficulty Level\n"
            f"  ----------------------------\n"
            f"  Easy:   {easy_accuracy:6.1%}  ({fmt_stats(by_difficulty['easy'])})  weight: {DIFFICULTY_WEIGHTS['easy']:.0f}x\n"
            f"  Medium: {medium_accuracy:6.1%}  ({fmt_stats(by_difficulty['medium'])})  weight: {DIFFICULTY_WEIGHTS['medium']:.0f}x\n"
            f"  Hard:   {hard_accuracy:6.1%}  ({fmt_stats(by_difficulty['hard'])})  weight: {DIFFICULTY_WEIGHTS['hard']:.0f}x\n"
            f"  Expert: {expert_accuracy:6.1%}  ({fmt_stats(by_difficulty['expert'])})  weight: {DIFFICULTY_WEIGHTS['expert']:.0f}x\n\n"
            f"  Accuracy by Subject Category\n"
            f"  ----------------------------\n"
            f"  Web:     {web_accuracy:6.1%}  ({fmt_stats(web_stats)})\n"
            f"  Science: {science_accuracy:6.1%}  ({fmt_stats(science_stats)})\n\n"
            f"  Accuracy by Category + Difficulty (Cross-tabulation)\n"
            f"  ----------------------------------------------------\n"
            f"  Web:\n"
            f"    Easy:   {web_easy_accuracy:6.1%}  ({fmt_stats(by_category_difficulty['web']['easy'])})\n"
            f"    Medium: {web_medium_accuracy:6.1%}  ({fmt_stats(by_category_difficulty['web']['medium'])})\n"
            f"    Hard:   {web_hard_accuracy:6.1%}  ({fmt_stats(by_category_difficulty['web']['hard'])})\n"
            f"    Expert: {web_expert_accuracy:6.1%}  ({fmt_stats(by_category_difficulty['web']['expert'])})\n"
            f"  Science:\n"
            f"    Easy:   {science_easy_accuracy:6.1%}  ({fmt_stats(by_category_difficulty['science']['easy'])})\n"
            f"    Medium: {science_medium_accuracy:6.1%}  ({fmt_stats(by_category_difficulty['science']['medium'])})\n"
            f"    Hard:   {science_hard_accuracy:6.1%}  ({fmt_stats(by_category_difficulty['science']['hard'])})\n"
            f"    Expert: {science_expert_accuracy:6.1%}  ({fmt_stats(by_category_difficulty['science']['expert'])})\n\n"
            f"  Leaderboard Metrics (for ranking)\n"
            f"  ---------------------------------\n"
            f"  weighted_score:        {weighted_score:.3f}  <- DEFAULT RANKING\n"
            f"  easy_accuracy:         {easy_accuracy:.3f}\n"
            f"  medium_accuracy:       {medium_accuracy:.3f}\n"
            f"  hard_accuracy:         {hard_accuracy:.3f}\n"
            f"  expert_accuracy:       {expert_accuracy:.3f}\n"
            f"  web_accuracy:          {web_accuracy:.3f}\n"
            f"  science_accuracy:      {science_accuracy:.3f}\n"
            f"  web_easy_accuracy:     {web_easy_accuracy:.3f}\n"
            f"  web_medium_accuracy:   {web_medium_accuracy:.3f}\n"
            f"  web_hard_accuracy:     {web_hard_accuracy:.3f}\n"
            f"  web_expert_accuracy:   {web_expert_accuracy:.3f}\n"
            f"  science_easy_accuracy: {science_easy_accuracy:.3f}\n"
            f"  science_medium_accuracy:{science_medium_accuracy:.3f}\n"
            f"  science_hard_accuracy: {science_hard_accuracy:.3f}\n"
            f"  science_expert_accuracy:{science_expert_accuracy:.3f}\n\n"
            "=" * 70
        )

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=results_text)),
                Part(root=DataPart(data=assessment_result.model_dump()))
            ],
            name="AQA Assessment Result",
        )
