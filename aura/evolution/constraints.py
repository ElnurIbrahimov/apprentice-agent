"""
Constraint Validator — Guardrails for evolved skill variants.

Every mutation must pass these gates before being accepted.
"""

import logging
from typing import Dict, List, Tuple

from .types import Candidate, GEPAConfig

logger = logging.getLogger(__name__)


class ConstraintValidator:
    """Validates evolved candidates against safety and quality constraints."""

    def __init__(self, config: GEPAConfig, seed_candidate: Candidate):
        self.config = config
        self.seed_sizes = {
            comp_id: len(text)
            for comp_id, text in seed_candidate.components.items()
        }

    def validate(self, candidate: Candidate) -> Tuple[bool, List[str]]:
        """
        Run all constraint checks on a candidate.

        Returns:
            (passed, list of violation messages)
        """
        violations = []

        for comp_id, text in candidate.components.items():
            # Non-empty check
            if not text or len(text.strip()) < 10:
                violations.append(f"{comp_id}: empty or trivially short")
                continue

            # Size limit
            if len(text) > self.config.max_skill_chars:
                violations.append(
                    f"{comp_id}: {len(text)} chars exceeds "
                    f"{self.config.max_skill_chars} limit"
                )

            # Growth limit
            seed_size = self.seed_sizes.get(comp_id, len(text))
            if seed_size > 0:
                growth = len(text) / seed_size
                if growth > self.config.max_growth_ratio:
                    violations.append(
                        f"{comp_id}: grew {growth:.1%} from seed, "
                        f"exceeds {self.config.max_growth_ratio:.0%} limit"
                    )

        if violations:
            logger.warning(f"Candidate {candidate.id} failed constraints: {violations}")

        return len(violations) == 0, violations
