"""
hallucination_detector.py — The Core Novelty Module

This module implements the Hallucination Detection Layer that:
1. Takes a generated answer + retrieved context
2. Uses Gemini to analyze consistency
3. Returns a structured HallucinationReport
4. Drives the feedback-based self-correction loop

This is the key differentiator over standard RAG systems.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import sys

import openai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    HALLUCINATION_THRESHOLD, STRICT_MODE_THRESHOLD,
    HALLUCINATION_CHECK_PROMPT_TEMPLATE
)

console = Console()


@dataclass
class HallucinationReport:
    """Structured result of hallucination detection analysis."""
    
    consistency_score: float          # 0.0 (fully hallucinated) → 1.0 (fully grounded)
    has_hallucination: bool           # True if score < HALLUCINATION_THRESHOLD
    hallucinated_claims: List[str]    # Specific claims that are unsupported
    supported_claims: List[str]       # Claims that ARE supported
    reasoning: str                    # Detector's explanation
    needs_strict_mode: bool = False   # True if score < STRICT_MODE_THRESHOLD
    detection_attempt: int = 1        # Which detection attempt this was
    raw_response: str = ""            # Raw LLM response for debugging
    
    @property
    def verdict(self) -> str:
        if self.consistency_score >= HALLUCINATION_THRESHOLD:
            return "GROUNDED"
        elif self.consistency_score >= STRICT_MODE_THRESHOLD:
            return "PARTIAL HALLUCINATION"
        else:
            return "SEVERE HALLUCINATION"
    
    @property
    def verdict_color(self) -> str:
        if self.consistency_score >= HALLUCINATION_THRESHOLD:
            return "green"
        elif self.consistency_score >= STRICT_MODE_THRESHOLD:
            return "yellow"
        else:
            return "red"


class HallucinationDetector:
    """
    Uses Gemini to detect inconsistencies between a generated answer
    and the retrieved source context documents.

    Algorithm:
        1. Format a detection prompt with context + answer
        2. Ask Gemini to analyze claim-by-claim consistency
        3. Parse the JSON response into a HallucinationReport
        4. Return the structured report for use in self-correction
    """

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Create a .env file with your key.\n"
                "Get a key at: https://platform.openai.com/api-keys"
            )
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = OPENAI_MODEL
        self._call_count = 0

    def detect(
        self,
        answer: str,
        context: str,
        attempt: int = 1
    ) -> HallucinationReport:
        """
        Analyze an answer for hallucinations against the given context.

        Args:
            answer: The LLM-generated answer to check
            context: The retrieved context documents as a string
            attempt: Which regeneration attempt this detection is for

        Returns:
            HallucinationReport with consistency score and analysis
        """
        console.print(f"\n[bold blue][Inspect] Hallucination Detection (attempt {attempt})[/bold blue]")
        
        # Format the detection prompt with the retrieved context and generated answer
        prompt = HALLUCINATION_CHECK_PROMPT_TEMPLATE.format(
            context=context,
            answer=answer
        )

        for retry_attempt in range(3):
            try:
                self._call_count += 1
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a hallucination detection expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,   # Low temp for deterministic analysis
                    max_tokens=1024,
                    response_format={"type": "json_object"}
                )
                raw_text = response.choices[0].message.content.strip()
                
                report = self._parse_detection_response(raw_text, attempt)
                self._print_report(report)
                return report

            except openai.RateLimitError as e:
                if retry_attempt < 2:
                    console.print(f"[dim yellow]Rate limit reached. Waiting 5 seconds before retrying...[/dim yellow]")
                    time.sleep(5)
                    continue
                console.print(f"[red]Detection error: {e}[/red]")
                # Return a safe "uncertain" report on error
                return HallucinationReport(
                    consistency_score=0.5,
                    has_hallucination=False,
                    hallucinated_claims=[],
                    supported_claims=[],
                    reasoning=f"Detection failed due to rate limit error: {str(e)}",
                    needs_strict_mode=False,
                    detection_attempt=attempt,
                    raw_response=str(e)
                )
            except Exception as e:
                console.print(f"[red]Detection error: {e}[/red]")
                # Return a safe "uncertain" report on error
                return HallucinationReport(
                consistency_score=0.5,
                has_hallucination=False,
                hallucinated_claims=[],
                supported_claims=[],
                reasoning=f"Detection failed due to error: {str(e)}",
                needs_strict_mode=False,
                detection_attempt=attempt,
                raw_response=str(e)
            )

    def _parse_detection_response(self, raw_text: str, attempt: int) -> HallucinationReport:
        """Parse Gemini's JSON response into a HallucinationReport."""
        
        # Strip markdown code fences if present
        clean = re.sub(r"```json\s*|```\s*", "", raw_text).strip()
        
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', clean, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        score = float(data.get("consistency_score", 0.5))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        return HallucinationReport(
            consistency_score=score,
            has_hallucination=bool(data.get("has_hallucination", score < HALLUCINATION_THRESHOLD)),
            hallucinated_claims=list(data.get("hallucinated_claims", [])),
            supported_claims=list(data.get("supported_claims", [])),
            reasoning=str(data.get("reasoning", "No reasoning provided")),
            needs_strict_mode=score < STRICT_MODE_THRESHOLD,
            detection_attempt=attempt,
            raw_response=raw_text
        )

    def _print_report(self, report: HallucinationReport):
        """Pretty-print the detection report to console."""
        
        # Score bar
        bar_len = 30
        filled = int(report.consistency_score * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        color = report.verdict_color
        console.print(f"   [{color}]{report.verdict}[/{color}]")
        console.print(f"   Consistency: [{color}]{bar}[/{color}] {report.consistency_score:.2f}")
        
        if report.hallucinated_claims:
            console.print(f"   [red]Hallucinated claims ({len(report.hallucinated_claims)}):[/red]")
            for claim in report.hallucinated_claims[:3]:  # Show up to 3
                console.print(f"     [red]• {claim[:100]}...[/red]" if len(claim) > 100 else f"     [red]• {claim}[/red]")
        
        if report.supported_claims:
            console.print(f"   [green]Supported claims ({len(report.supported_claims)}):[/green]")
            for claim in report.supported_claims[:2]:
                console.print(f"     [green]• {claim[:100]}...[/green]" if len(claim) > 100 else f"     [green]• {claim}[/green]")
        
        console.print(f"   [dim]Reasoning: {report.reasoning[:150]}[/dim]")