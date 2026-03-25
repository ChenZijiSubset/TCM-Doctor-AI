from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class SyndromeResult:
    name: str
    score: float
    matched_key_symptoms: List[str]
    matched_symptoms: List[str]
    matched_tongue: List[str]
    matched_pulse: List[str]
    matched_exclusions: List[str]
    treatment: str
    formulas: List[str]
    source: str
    explain: str


def normalize_list(items: List[str]) -> List[str]:
    return [str(x).strip() for x in items if str(x).strip()]


def score_syndrome(user_features: Dict[str, List[str]], syndrome: Dict[str, Any]) -> SyndromeResult:
    user_symptoms = set(normalize_list(user_features.get("symptoms", [])))
    user_tongue = set(normalize_list(user_features.get("tongue", [])))
    user_pulse = set(normalize_list(user_features.get("pulse", [])))

    key_symptoms = normalize_list(syndrome.get("key_symptoms", []))
    symptoms = normalize_list(syndrome.get("symptoms", []))
    tongue = normalize_list(syndrome.get("tongue", []))
    pulse = normalize_list(syndrome.get("pulse", []))
    exclusions = normalize_list(syndrome.get("exclusions", []))

    matched_key_symptoms = sorted(list(user_symptoms.intersection(set(key_symptoms))))
    matched_symptoms = sorted(list(user_symptoms.intersection(set(symptoms))))
    matched_tongue = sorted(list(user_tongue.intersection(set(tongue))))
    matched_pulse = sorted(list(user_pulse.intersection(set(pulse))))
    matched_exclusions = sorted(list(user_symptoms.intersection(set(exclusions))))

    raw_score = 0.0
    raw_score += 3.0 * len(matched_key_symptoms)
    raw_score += 1.0 * len(matched_symptoms)
    raw_score += 2.0 * len(matched_tongue)
    raw_score += 2.0 * len(matched_pulse)
    raw_score -= 3.0 * len(matched_exclusions)

    max_possible = (
        3.0 * len(key_symptoms)
        + 1.0 * len(symptoms)
        + 2.0 * len(tongue)
        + 2.0 * len(pulse)
    )

    if max_possible <= 0:
        normalized_score = 0.0
    else:
        normalized_score = max(0.0, min(100.0, (raw_score / max_possible) * 100.0))

    return SyndromeResult(
        name=syndrome.get("name", ""),
        score=round(normalized_score, 2),
        matched_key_symptoms=matched_key_symptoms,
        matched_symptoms=matched_symptoms,
        matched_tongue=matched_tongue,
        matched_pulse=matched_pulse,
        matched_exclusions=matched_exclusions,
        treatment=syndrome.get("treatment", ""),
        formulas=normalize_list(syndrome.get("formulas", [])),
        source=syndrome.get("source", ""),
        explain=syndrome.get("explain", ""),
    )


def rank_syndromes(
    user_features: Dict[str, List[str]],
    syndrome_db: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[SyndromeResult]:
    results = [score_syndrome(user_features, syndrome) for syndrome in syndrome_db]
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]