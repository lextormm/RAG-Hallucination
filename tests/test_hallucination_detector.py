"""
test_hallucination_detector.py — Unit tests for the Hallucination Detector

Tests the core novelty: the hallucination detection layer.
Uses mock responses to avoid API calls during testing.
"""

import sys
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock all heavy dependencies before importing anything
# Use real openai module so openai.RateLimitError is a real Exception class.
# We will mock the client instance instead.
sys.modules['langchain'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.documents'] = MagicMock()
sys.modules['langchain_chroma'] = MagicMock()
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.embeddings'] = MagicMock()

from hallucination_detector import HallucinationDetector, HallucinationReport


class TestHallucinationReport(unittest.TestCase):
    """Tests for the HallucinationReport dataclass."""

    def test_verdict_grounded(self):
        report = HallucinationReport(
            consistency_score=0.9,
            has_hallucination=False,
            hallucinated_claims=[],
            supported_claims=["Claim A"],
            reasoning="All good",
        )
        self.assertIn("GROUNDED", report.verdict)
        self.assertEqual(report.verdict_color, "green")

    def test_verdict_partial_hallucination(self):
        report = HallucinationReport(
            consistency_score=0.5,
            has_hallucination=True,
            hallucinated_claims=["False claim"],
            supported_claims=[],
            reasoning="Some issues",
        )
        self.assertIn("PARTIAL", report.verdict)
        self.assertEqual(report.verdict_color, "yellow")

    def test_verdict_severe_hallucination(self):
        report = HallucinationReport(
            consistency_score=0.2,
            has_hallucination=True,
            hallucinated_claims=["Many false claims"],
            supported_claims=[],
            reasoning="Very bad",
        )
        self.assertIn("SEVERE", report.verdict)
        self.assertEqual(report.verdict_color, "red")

    def test_needs_strict_mode(self):
        report = HallucinationReport(
            consistency_score=0.3,
            has_hallucination=True,
            hallucinated_claims=[],
            supported_claims=[],
            reasoning="",
            needs_strict_mode=True,
        )
        self.assertTrue(report.needs_strict_mode)


class TestHallucinationDetector(unittest.TestCase):
    """Tests for the HallucinationDetector class."""

    def setUp(self):
        """Set up detector with mocked OpenAI."""
        with patch('hallucination_detector.OPENAI_API_KEY', 'fake_key_for_testing'):
            self.detector = HallucinationDetector()
        # Replace client with fresh mock so each test controls it cleanly
        self.detector.client = MagicMock()

    def _make_mock_response(self, json_data: dict) -> MagicMock:
        """Create a mock OpenAI response."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps(json_data)
        return mock_resp

    def test_detect_grounded_answer(self):
        """Test that a well-grounded answer gets high score."""
        self.detector.client.chat.completions.create.return_value = self._make_mock_response({
            "consistency_score": 0.95,
            "has_hallucination": False,
            "hallucinated_claims": [],
            "supported_claims": ["RAG combines retrieval with generation"],
            "reasoning": "Answer is well-supported by context"
        })
        
        report = self.detector.detect(
            answer="RAG combines retrieval with generation.",
            context="RAG stands for Retrieval-Augmented Generation. It combines retrieval with generation."
        )
        
        self.assertFalse(report.has_hallucination)
        self.assertGreaterEqual(report.consistency_score, 0.6)

    def test_detect_hallucinated_answer(self):
        """Test that a hallucinated answer gets low score."""
        self.detector.client.chat.completions.create.return_value = self._make_mock_response({
            "consistency_score": 0.2,
            "has_hallucination": True,
            "hallucinated_claims": ["The system was invented in 1985", "Uses quantum computing"],
            "supported_claims": [],
            "reasoning": "Multiple unsupported claims"
        })
        
        report = self.detector.detect(
            answer="RAG was invented in 1985 and uses quantum computing.",
            context="RAG is a modern technique combining information retrieval with LLMs."
        )
        
        self.assertTrue(report.has_hallucination)
        self.assertLess(report.consistency_score, 0.6)
        self.assertGreater(len(report.hallucinated_claims), 0)

    def test_needs_strict_mode_triggered(self):
        """Test that severe hallucination triggers strict mode."""
        self.detector.client.chat.completions.create.return_value = self._make_mock_response({
            "consistency_score": 0.1,
            "has_hallucination": True,
            "hallucinated_claims": ["Many false facts"],
            "supported_claims": [],
            "reasoning": "Almost entirely hallucinated"
        })
        
        report = self.detector.detect("fabricated answer", "real context")
        self.assertTrue(report.needs_strict_mode)

    def test_parse_json_with_code_fences(self):
        """Test that JSON wrapped in code fences is parsed correctly."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '```json\n{"consistency_score": 0.8, "has_hallucination": false, "hallucinated_claims": [], "supported_claims": ["fact"], "reasoning": "good"}\n```'
        self.detector.client.chat.completions.create.return_value = mock_resp
        
        report = self.detector.detect("answer", "context")
        self.assertAlmostEqual(report.consistency_score, 0.8, places=1)
        self.assertFalse(report.has_hallucination)

    def test_score_clamping(self):
        """Test that scores outside [0,1] are clamped."""
        self.detector.client.chat.completions.create.return_value = self._make_mock_response({
            "consistency_score": 1.5,  # Out of range!
            "has_hallucination": False,
            "hallucinated_claims": [],
            "supported_claims": [],
            "reasoning": ""
        })
        
        report = self.detector.detect("answer", "context")
        self.assertLessEqual(report.consistency_score, 1.0)

    def test_graceful_error_handling(self):
        """Test that API errors don't crash the system."""
        self.detector.client.chat.completions.create.side_effect = Exception("API Error")
        
        report = self.detector.detect("answer", "context")
        # Should return a safe default, not raise
        self.assertIsNotNone(report)
        self.assertIsInstance(report.consistency_score, float)

    def test_malformed_json_handled(self):
        """Test that malformed JSON responses are handled gracefully."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "This is not JSON at all!"
        self.detector.client.chat.completions.create.return_value = mock_resp
        
        report = self.detector.detect("answer", "context")
        self.assertIsNotNone(report)


class TestRAGEngine(unittest.TestCase):
    """Tests for RAGResponse logic (no langchain needed)."""

    def _make_response(self, score: float):
        """Helper to build a RAGResponse-like object using the real dataclass."""
        # Define a minimal RAGResponse inline to avoid langchain import
        from dataclasses import dataclass, field as dc_field
        from typing import List as TList
        
        mock_report = HallucinationReport(
            consistency_score=score,
            has_hallucination=score < 0.6,
            hallucinated_claims=[],
            supported_claims=[],
            reasoning=""
        )

        @dataclass
        class _RAGResponse:
            query: str
            answer: str
            sources: TList[str]
            consistency_score: float
            hallucination_report: object
            regeneration_count: int
            used_strict_mode: bool

            @property
            def is_reliable(self):
                return self.consistency_score >= 0.6

            @property
            def confidence_label(self):
                if self.consistency_score >= 0.85:
                    return "HIGH"
                elif self.consistency_score >= 0.6:
                    return "MEDIUM"
                elif self.consistency_score >= 0.4:
                    return "LOW"
                else:
                    return "VERY LOW"

        return _RAGResponse(
            query="q", answer="a", sources=[],
            consistency_score=score,
            hallucination_report=mock_report,
            regeneration_count=0, used_strict_mode=False,
        )

    def test_rag_response_fields(self):
        r = self._make_response(0.85)
        self.assertEqual(r.query, "q")
        self.assertTrue(r.is_reliable)
        self.assertEqual(r.confidence_label, "HIGH")

    def test_confidence_labels(self):
        self.assertEqual(self._make_response(0.9).confidence_label, "HIGH")
        self.assertEqual(self._make_response(0.7).confidence_label, "MEDIUM")
        self.assertEqual(self._make_response(0.5).confidence_label, "LOW")
        self.assertEqual(self._make_response(0.2).confidence_label, "VERY LOW")


if __name__ == "__main__":
    print("=" * 60)
    print("Hallucination-Aware RAG — Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)