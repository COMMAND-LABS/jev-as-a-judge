"""Jev judge: scores an agent's answer against the dataset's reference answer."""

from typing import Any

from typesafe_sdk import Score, TypeSafeClient

JEV_JUDGE_METRIC_KEY = "jev_judge_score"

# Rubric ordered from score 0 to score 4.
JUDGE_INSTRUCTIONS = "Compared to the reference answer, how would you rate the AI agent's answer?"
JUDGE_SCORE_CRITERIA = [
    "Poor",
    "Bad, but not the worst",
    "Somewhat correct",
    "Okay",
    "Great",
]


def build_jev_judge_evaluator(*, model_name: str, api_key: str, timeout_seconds: int) -> Any:
    def jev_judge_evaluator(run: Any, example: Any) -> dict[str, Any]:
        question = (example.inputs or {}).get("question", "")
        expected = (example.outputs or {}).get("answer", "")
        predicted = (run.outputs or {}).get("answer", "")

        with TypeSafeClient(api_key=api_key, model=model_name, timeout=timeout_seconds) as client:
            response = client.system_one(
                state=(
                    f"Question: {question}\n"
                    f"Reference answer: {expected}\n"
                    f"AI agent's answer: {predicted}\n"
                ),
                questions={"judge": Score(instructions=JUDGE_INSTRUCTIONS, criteria=JUDGE_SCORE_CRITERIA)},
            )

        # Jev's score is probability-weighted, so it can fall between rubric levels (e.g. 3.6).
        answer = response.scores["judge"]
        probabilities = ", ".join(f"{level}={prob:.2f}" for level, prob in sorted(answer.probabilities.items()))
        return {
            "key": JEV_JUDGE_METRIC_KEY,
            "score": answer.score,
            "comment": f"confidence={answer.confidence:.2f}; probabilities: {probabilities}",
        }

    return jev_judge_evaluator
