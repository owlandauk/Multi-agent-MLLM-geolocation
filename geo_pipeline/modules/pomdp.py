"""
POMDP — Sequential Decision-Making for Evidence Selection

GeoBayes executes verification tasks in a fixed order. This module supports two
policies for replacing that traversal:

1. ``eig`` (default): finite-horizon approximate POMDP. The current posterior is
   the belief state, each pending verification task is an action, and repeated
   MLLM samples estimate the observation model P(o | l, a). The selected action
   maximizes expected information gain.
2. ``llm``: lightweight LLM-as-policy fallback from earlier versions.

State       b_t(l) = current posterior over geographic hypotheses
Action      a_t    = which verification task v_i to execute next
Observation o_t    = support rating in {1,2,3,4,5}
Reward      r_t    = H(b_t) - H(b_{t+1}) by default
"""

import math
import re
from models.mllm_client import MLLMClient
from config import (
    POMDP_ACTION_COST,
    POMDP_EIG_LEVELS,
    POMDP_MAX_ACTIONS,
    POMDP_MAX_NEW_TOKENS,
    POMDP_MAX_STEPS,
    POMDP_OBS_SAMPLES,
    POMDP_OBS_SMOOTHING,
    POMDP_POLICY,
    POMDP_REWARD_MODE,
    POMDP_TOP_HYPOTHESES,
)


_CHOICE_RE = re.compile(r'"?task_index"?\s*:\s*(\d+)', re.IGNORECASE)
_RATING_RE = re.compile(
    r"support[_\s]*rating[:\s]+([1-5])|rating[:\s]+([1-5])|score[:\s]+([1-5])",
    re.IGNORECASE,
)
_RATINGS = (1, 2, 3, 4, 5)
_NEUTRAL_OBS = {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0}


def _renormalize(d: dict[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in d.values())
    if total <= 0:
        n = max(len(d), 1)
        return {k: 1.0 / n for k in d}
    return {k: float(v) / total for k, v in d.items()}


def _entropy(posterior: dict[str, float]) -> float:
    vals = [float(v) for v in posterior.values() if v > 0]
    if len(vals) <= 1:
        return 0.0
    return -sum(v * math.log(max(v, 1e-12)) for v in vals)


def _top_mass(posterior: dict[str, float]) -> float:
    return max(posterior.values(), default=0.0)


def _parse_rating(text: str) -> int:
    match = _RATING_RE.search(text or "")
    if not match:
        return 3
    for group in match.groups():
        if group:
            return int(group)
    return 3


def _rating_distribution(samples: list[str]) -> dict[int, float]:
    counts = {rating: POMDP_OBS_SMOOTHING for rating in _RATINGS}
    for sample in samples:
        counts[_parse_rating(sample)] += 1.0
    total = sum(counts.values())
    return {rating: count / total for rating, count in counts.items()}


def _top_hypotheses(posterior: dict[str, float]) -> list[str]:
    items = sorted(posterior.items(), key=lambda x: -x[1])
    return [hyp for hyp, _ in items[:max(1, POMDP_TOP_HYPOTHESES)]]


class POMDPModule:
    """
    Action-selection module for one hierarchy level. By default this uses a
    model-based one-step expected information gain estimate; set
    POMDP_POLICY=llm to recover the older direct LLM policy.
    """

    def __init__(self, mllm: MLLMClient, max_steps: int = POMDP_MAX_STEPS):
        self.mllm = mllm
        self.max_steps = max_steps
        self.policy = POMDP_POLICY
        self.eig_levels = POMDP_EIG_LEVELS

    @property
    def use_expected_gain(self) -> bool:
        return self.policy in {"eig", "expected_info_gain", "pomdp", "model_based"}

    def use_expected_gain_for(self, level: str) -> bool:
        return self.use_expected_gain and level in self.eig_levels

    @property
    def policy_label(self) -> str:
        if not self.use_expected_gain:
            return self.policy
        return f"{self.policy}:{','.join(self.eig_levels)}"

    def _belief_summary(self, posterior: dict[str, float]) -> str:
        items = sorted(posterior.items(), key=lambda x: -x[1])
        return ", ".join(f"{h}: {p:.3f}" for h, p in items[:5])

    def _make_policy_prompt(
        self,
        posterior: dict[str, float],
        pending_tasks: list[dict],
        level: str,
        step: int,
    ) -> list:
        task_list = "\n".join(
            f"  [{i}] desc: {t['desc']}, region: {t.get('bbox', 'full image')}"
            for i, t in enumerate(pending_tasks)
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"You are a geolocation agent at reasoning level: {level} (step {step}).\n\n"
                        f"Current belief over locations:\n  {self._belief_summary(posterior)}\n\n"
                        f"Pending verification tasks:\n{task_list}\n\n"
                        "Select the single task that would most reduce uncertainty — i.e., "
                        "the task most likely to produce evidence that strongly supports or "
                        "contradicts the current top hypotheses.\n\n"
                        'Respond with JSON only: {"task_index": <int>, "reason": "<short reason>"}'
                    )},
                ],
            }
        ]

    def select_action(
        self,
        posterior: dict[str, float],
        pending_tasks: list[dict],
        level: str,
        step: int,
    ) -> int:
        """
        Returns the index of the selected task in pending_tasks.
        Falls back to index 0 if parsing fails.
        """
        if len(pending_tasks) == 1:
            return 0

        messages = self._make_policy_prompt(posterior, pending_tasks, level, step)
        response = self.mllm.generate(messages, max_new_tokens=POMDP_MAX_NEW_TOKENS)

        match = _CHOICE_RE.search(response)
        if match:
            idx = int(match.group(1))
            return min(idx, len(pending_tasks) - 1)
        return 0  # fallback: first task

    def _make_observation_model_prompt(
        self,
        image,
        task: dict,
        hypothesis: str,
        posterior: dict[str, float],
        level: str,
        step: int,
    ) -> list:
        bbox = task.get("bbox")
        region_note = f" Focus on region [x,y,w,h]={bbox}." if bbox else ""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "You are estimating an observation model for a finite-horizon "
                        "POMDP geolocation agent.\n"
                        f"Reasoning level: {level}; planning step: {step}.\n"
                        f"Current belief: {self._belief_summary(posterior)}\n\n"
                        f"Candidate location hypothesis: '{hypothesis}'.\n"
                        f"Potential verification action: {task.get('desc', '')}.{region_note}\n\n"
                        "If this action were executed on the image, predict how strongly "
                        "the resulting visual evidence would support this hypothesis. "
                        "Use 1 for strong contradiction, 3 for neutral/uncertain, and 5 "
                        "for strong support.\n\n"
                        "Respond exactly with:\n"
                        "Support Rating: <1-5>\n"
                        "Confidence: <0.0-1.0>"
                    )},
                ],
            }
        ]

    def _expected_information_gain(
        self,
        posterior: dict[str, float],
        observation_model: dict[str, dict[int, float]],
    ) -> float:
        prior = _renormalize(posterior)
        prior_entropy = _entropy(prior)
        prior_top = _top_mass(prior)
        expected_reward = 0.0

        for rating in _RATINGS:
            likelihoods = {}
            obs_prob = 0.0
            for hyp, prob in prior.items():
                dist = observation_model.get(hyp, _NEUTRAL_OBS)
                likelihood = max(dist.get(rating, 0.0), 1e-6)
                likelihoods[hyp] = likelihood
                obs_prob += prob * likelihood

            if obs_prob <= 0:
                continue

            next_belief = _renormalize(
                {hyp: prob * likelihoods[hyp] for hyp, prob in prior.items()}
            )
            if POMDP_REWARD_MODE == "map_gain":
                reward = max(0.0, _top_mass(next_belief) - prior_top)
            else:
                reward = max(0.0, prior_entropy - _entropy(next_belief))
            expected_reward += obs_prob * reward

        return expected_reward - POMDP_ACTION_COST

    def _action_indices(self, pending_tasks: list[dict]) -> list[int]:
        if POMDP_MAX_ACTIONS <= 0:
            return list(range(len(pending_tasks)))
        return list(range(min(len(pending_tasks), POMDP_MAX_ACTIONS)))

    def select_actions_by_expected_gain(
        self,
        images: list,
        posteriors: list[dict[str, float]],
        pending_tasks: list[list[dict]],
        level: str,
        steps: list[int],
    ) -> list[int]:
        """Return action indices using one-step expected information gain.

        For each candidate action ``a`` and top hypothesis ``l``, repeated MLLM
        samples estimate P(o | l, a), where o is a 1-5 support rating. We then
        marginalize over possible observations and choose the action with the
        largest expected uncertainty reduction.
        """
        choices = [0 for _ in images]
        prompts = []
        owners: list[tuple[int, int, str]] = []

        for item_idx, (image, posterior, tasks, step) in enumerate(
            zip(images, posteriors, pending_tasks, steps)
        ):
            if len(tasks) <= 1:
                continue
            hypotheses = _top_hypotheses(posterior)
            for action_idx in self._action_indices(tasks):
                for hypothesis in hypotheses:
                    prompts.append(
                        self._make_observation_model_prompt(
                            image, tasks[action_idx], hypothesis, posterior, level, step
                        )
                    )
                    owners.append((item_idx, action_idx, hypothesis))

        if not prompts:
            return choices

        responses = self.mllm.batch_sample_n(
            prompts,
            n=POMDP_OBS_SAMPLES,
            max_new_tokens=POMDP_MAX_NEW_TOKENS,
        )

        models: dict[tuple[int, int], dict[str, dict[int, float]]] = {}
        for (item_idx, action_idx, hypothesis), samples in zip(owners, responses):
            action_model = models.setdefault((item_idx, action_idx), {})
            action_model[hypothesis] = _rating_distribution(samples)

        for item_idx, (posterior, tasks) in enumerate(zip(posteriors, pending_tasks)):
            if len(tasks) <= 1:
                continue
            best_idx = 0
            best_score = float("-inf")
            for action_idx in self._action_indices(tasks):
                score = self._expected_information_gain(
                    posterior,
                    models.get((item_idx, action_idx), {}),
                )
                if score > best_score:
                    best_score = score
                    best_idx = action_idx
            choices[item_idx] = best_idx

        return choices

    def select_action_by_expected_gain(
        self,
        image,
        posterior: dict[str, float],
        pending_tasks: list[dict],
        level: str,
        step: int,
    ) -> int:
        return self.select_actions_by_expected_gain(
            [image], [posterior], [pending_tasks], level, [step]
        )[0]

    def should_stop(
        self,
        posterior: dict[str, float],
        step: int,
        level: str,
        all_tasks_exhausted: bool,
    ) -> bool:
        """
        POMDP stopping condition — mirrors GeoBayes Eq.11 but adds a step cap.
        At street level: stop when tasks exhausted.
        At other levels: stop when max_prob > TRANSITION_THR after at least
        one verification step, or when tasks are exhausted.
        """
        from config import TRANSITION_THR
        max_prob = max(posterior.values(), default=0.0)

        if step >= self.max_steps:
            return True
        if all_tasks_exhausted:
            return True
        if level != "street" and step > 0 and max_prob >= TRANSITION_THR:
            return True
        return False
