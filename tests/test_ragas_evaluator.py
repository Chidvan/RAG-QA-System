"""Tests for RAGAS evaluator module."""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from app.core.ragas_evaluator import RAGASEvaluator


class TestRAGASEvaluator:
    """Test suite for RAGASEvaluator class."""

    @patch("app.core.ragas_evaluator.ChatOpenAI")
    @patch("app.core.ragas_evaluator.OpenAIEmbeddings")
    def test_evaluator_initialization(self, mock_embeddings, mock_llm):
        """Test that evaluator initializes correctly."""
        evaluator = RAGASEvaluator()

        assert evaluator.llm is not None
        assert evaluator.embeddings is not None
        assert len(evaluator.metrics) == 2  # faithfulness and answer_relevancy
        assert evaluator.metrics[0].name == "faithfulness"
        assert evaluator.metrics[1].name == "answer_relevancy"

    @patch("app.core.ragas_evaluator.ChatOpenAI")
    @patch("app.core.ragas_evaluator.OpenAIEmbeddings")
    @patch("app.core.ragas_evaluator.get_settings")
    def test_evaluator_uses_separate_llm_config(self, mock_settings, mock_embeddings, mock_llm):
        """Test that evaluator uses separate RAGAS LLM configuration when provided."""
        # Setup mock settings with separate RAGAS LLM
        settings = MagicMock()
        settings.llm_model = "gpt-4o-mini"
        settings.llm_temperature = 0.0
        settings.embedding_model = "text-embedding-3-small"
        settings.openai_api_key = "test-key"
        settings.ragas_llm_model = "gpt-4o"  # Different model for evaluation
        settings.ragas_llm_temperature = 0.1
        settings.ragas_embedding_model = "text-embedding-3-large"
        mock_settings.return_value = settings

        evaluator = RAGASEvaluator()

        # Verify ChatOpenAI was called with RAGAS-specific settings
        mock_llm.assert_called_once_with(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key="test-key",
        )

        # Verify OpenAIEmbeddings was called with RAGAS-specific settings
        mock_embeddings.assert_called_once_with(
            model="text-embedding-3-large",
            openai_api_key="test-key",
        )

    @patch("app.core.ragas_evaluator.ChatOpenAI")
    @patch("app.core.ragas_evaluator.OpenAIEmbeddings")
    @patch("app.core.ragas_evaluator.get_settings")
    def test_evaluator_fallback_to_default_config(self, mock_settings, mock_embeddings, mock_llm):
        """Test that evaluator falls back to default LLM when RAGAS config is None."""
        # Setup mock settings without separate RAGAS LLM
        settings = MagicMock()
        settings.llm_model = "gpt-4o-mini"
        settings.llm_temperature = 0.0
        settings.embedding_model = "text-embedding-3-small"
        settings.openai_api_key = "test-key"
        settings.ragas_llm_model = None  # Will fall back to llm_model
        settings.ragas_llm_temperature = None  # Will fall back to llm_temperature
        settings.ragas_embedding_model = None  # Will fall back to embedding_model
        mock_settings.return_value = settings

        evaluator = RAGASEvaluator()

        # Verify ChatOpenAI was called with default settings
        mock_llm.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.0,
            openai_api_key="test-key",
        )

        # Verify OpenAIEmbeddings was called with default settings
        mock_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            openai_api_key="test-key",
        )

    def test_prepare_dataset(self):
        """Test dataset preparation for RAGAS."""
        evaluator = RAGASEvaluator()

        question = "What is RAG?"
        answer = "RAG stands for Retrieval-Augmented Generation"
        contexts = ["Context 1", "Context 2"]

        dataset = evaluator._prepare_dataset(question, answer, contexts)

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 1
        assert dataset[0]["question"] == question
        assert dataset[0]["answer"] == answer
        assert dataset[0]["contexts"] == contexts

    @pytest.mark.asyncio
    @patch("app.core.ragas_evaluator.evaluate")
    async def test_aevaluate_success(self, mock_evaluate):
        """Test successful async evaluation."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.to_pandas.return_value.to_dict.return_value = [
            {"faithfulness": 0.95, "answer_relevancy": 0.87}
        ]
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()

        question = "What is RAG?"
        answer = "RAG stands for Retrieval-Augmented Generation"
        contexts = ["Context about RAG"]

        result = await evaluator.aevaluate(question, answer, contexts)

        assert result["faithfulness"] == 0.95
        assert result["answer_relevancy"] == 0.87
        assert result["error"] is None
        assert "evaluation_time_ms" in result
        assert result["evaluation_time_ms"] >= 0

    @pytest.mark.asyncio
    @patch("app.core.ragas_evaluator.evaluate")
    async def test_aevaluate_with_error(self, mock_evaluate):
        """Test evaluation error handling."""
        # Setup mock to raise exception
        mock_evaluate.side_effect = Exception("Evaluation failed")

        evaluator = RAGASEvaluator()

        question = "What is RAG?"
        answer = "RAG stands for Retrieval-Augmented Generation"
        contexts = ["Context about RAG"]

        result = await evaluator.aevaluate(question, answer, contexts)

        # Should return fallback scores
        assert result["faithfulness"] is None
        assert result["answer_relevancy"] is None
        assert result["evaluation_time_ms"] is None
        assert result["error"] == "Evaluation failed"

    def test_handle_evaluation_error(self):
        """Test error handling returns correct format."""
        evaluator = RAGASEvaluator()

        error = Exception("Test error")
        result = evaluator._handle_evaluation_error(error)

        assert result["faithfulness"] is None
        assert result["answer_relevancy"] is None
        assert result["evaluation_time_ms"] is None
        assert result["error"] == "Test error"

    @pytest.mark.asyncio
    @patch("app.core.ragas_evaluator.evaluate")
    async def test_aevaluate_with_missing_metrics(self, mock_evaluate):
        """Test evaluation when some metrics are missing."""
        # Setup mock with only one metric
        mock_result = MagicMock()
        mock_result.to_pandas.return_value.to_dict.return_value = [
            {"faithfulness": 0.95}  # answer_relevancy missing
        ]
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()

        question = "What is RAG?"
        answer = "RAG stands for Retrieval-Augmented Generation"
        contexts = ["Context about RAG"]

        result = await evaluator.aevaluate(question, answer, contexts)

        assert result["faithfulness"] == 0.95
        assert result["answer_relevancy"] is None  # Should be None when missing
        assert result["error"] is None

    @pytest.mark.asyncio
    @patch("app.core.ragas_evaluator.evaluate")
    async def test_aevaluate_with_empty_contexts(self, mock_evaluate):
        """Test evaluation with empty contexts list."""
        mock_result = MagicMock()
        mock_result.to_pandas.return_value.to_dict.return_value = [
            {"faithfulness": 0.0, "answer_relevancy": 0.5}
        ]
        mock_evaluate.return_value = mock_result

        evaluator = RAGASEvaluator()

        question = "What is RAG?"
        answer = "RAG stands for Retrieval-Augmented Generation"
        contexts = []  # Empty contexts

        result = await evaluator.aevaluate(question, answer, contexts)

        # Should still complete but with low faithfulness
        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert result["error"] is None
