"""Tests for query endpoints."""


class TestQueryEndpoints:
    """Test query API endpoints."""

    def test_query_endpoint(self, client, mock_rag_chain):
        """Test basic query endpoint."""
        request_data = {
            "question": "What is RAG?",
            "include_sources": True,
        }

        response = client.post("/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "processing_time_ms" in data
        assert data["question"] == "What is RAG?"

    def test_query_without_sources(self, client, mock_rag_chain):
        """Test query without sources."""
        request_data = {
            "question": "What is embeddings?",
            "include_sources": False,
        }

        response = client.post("/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] is None

    def test_query_with_sources(self, client, mock_rag_chain):
        """Test query with sources included."""
        request_data = {
            "question": "Explain vector databases",
            "include_sources": True,
        }

        response = client.post("/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_empty_question(self, client):
        """Test query with empty question."""
        request_data = {
            "question": "",
            "include_sources": True,
        }

        response = client.post("/query", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_query_missing_question(self, client):
        """Test query without question field."""
        request_data = {"include_sources": True}

        response = client.post("/query", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_query_with_evaluation_enabled(self, client_with_evaluation):
        """Test query with evaluation enabled."""
        request_data = {
            "question": "What is RAG?",
            "include_sources": True,
            "enable_evaluation": True,
        }

        response = client_with_evaluation.post("/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "evaluation" in data
        assert data["evaluation"] is not None
        assert "faithfulness" in data["evaluation"]
        assert "answer_relevancy" in data["evaluation"]
        assert data["evaluation"]["faithfulness"] == 0.95
        assert data["evaluation"]["answer_relevancy"] == 0.87

    def test_query_with_evaluation_disabled(self, client, mock_rag_chain):
        """Test query with evaluation disabled (default)."""
        request_data = {
            "question": "What is RAG?",
            "include_sources": True,
            "enable_evaluation": False,
        }

        response = client.post("/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "evaluation" in data
        assert data["evaluation"] is None

    def test_query_evaluation_scores_in_response(self, client_with_evaluation):
        """Test that evaluation scores are properly formatted in response."""
        request_data = {
            "question": "Explain vector databases",
            "include_sources": True,
            "enable_evaluation": True,
        }

        response = client_with_evaluation.post("/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        evaluation = data["evaluation"]

        # Check evaluation structure
        assert isinstance(evaluation["faithfulness"], float)
        assert isinstance(evaluation["answer_relevancy"], float)
        assert 0.0 <= evaluation["faithfulness"] <= 1.0
        assert 0.0 <= evaluation["answer_relevancy"] <= 1.0
        assert "evaluation_time_ms" in evaluation
        assert evaluation["error"] is None

    def test_query_with_evaluation_error(self, client_with_evaluation_error):
        """Test graceful degradation when evaluation fails."""
        request_data = {
            "question": "What is RAG?",
            "include_sources": True,
            "enable_evaluation": True,
        }

        response = client_with_evaluation_error.post("/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should still have answer even if evaluation failed
        assert "answer" in data
        assert data["answer"] is not None

        # Evaluation should have error info
        assert data["evaluation"] is not None
        assert data["evaluation"]["faithfulness"] is None
        assert data["evaluation"]["answer_relevancy"] is None
        assert data["evaluation"]["error"] is not None

    def test_search_documents(self, client, mock_vector_store):
        """Test document search endpoint."""
        from langchain_core.documents import Document

        # Configure mock to return search results
        mock_vector_store.search_with_scores.return_value = [
            (
                Document(
                    page_content="RAG content",
                    metadata={"source": "test.txt"},
                ),
                0.95,
            )
        ]

        request_data = {"question": "RAG pipeline"}
        response = client.post("/query/search", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "count" in data


class TestQueryValidation:
    """Test query validation."""

    def test_question_max_length(self, client):
        """Test question max length validation."""
        # Create a question longer than 1000 characters
        long_question = "a" * 1001

        request_data = {
            "question": long_question,
            "include_sources": True,
        }

        response = client.post("/query", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_valid_question_length(self, client, mock_rag_chain):
        """Test valid question length."""
        request_data = {
            "question": "What is machine learning?",
            "include_sources": True,
        }

        response = client.post("/query", json=request_data)

        assert response.status_code == 200
