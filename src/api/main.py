import logging
import os
from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from pydantic import BaseModel

from src.retrieval.retriever import PaperRetriever
from src.generation.generator import ResponseGenerator
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.evaluation.healer import RAGHealer
from src.evaluation.scoring import compute_composite_score
from src.classification.query_classifier import classify_query
from src.utils.db import engine, Base, get_db
from src.utils.models import QueryLog, HealingAttempt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arxivnavigator")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ArXivNavigator API")

retriever = PaperRetriever()
generator = ResponseGenerator()
evaluator = RagasEvaluator()
healer = RAGHealer()

VALID_CATEGORIES = {"cs_ai", "cs_lg", "cs_cl", "cs_cv", "cs_ir"}


class QueryRequest(BaseModel):
    query: str
    category: str = "cs_ai"


class QueryResponse(BaseModel):
    query: str
    category: str
    query_type: str
    answer: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_relevance: float | None
    composite_score: float
    healing_triggered: bool
    healing_attempts: int
    winning_strategy: str | None


@app.get("/health")
def health():
    """Checks that the API and all its real dependencies (Postgres, Pinecone, Ollama)
    are actually reachable, not just that the FastAPI process is running."""
    checks = {}

    try:
        db = next(get_db())
        db.execute(text("SELECT 1"))
        db.close()
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"unreachable: {e}"

    try:
        retriever.index.describe_index_stats()
        checks["pinecone"] = "ok"
    except Exception as e:
        checks["pinecone"] = f"unreachable: {e}"

    try:
        import requests as _requests
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = _requests.get(f"{ollama_url}/api/tags", timeout=5)
        checks["ollama"] = "ok" if resp.status_code == 200 else f"unexpected status {resp.status_code}"
    except Exception as e:
        checks["ollama"] = f"unreachable: {e}"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "dependencies": checks}


@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if request.category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{request.category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
        )

    query_type = classify_query(request.query)
    attempts_log = []

    try:
        docs = retriever.retrieve(request.query, namespace=request.category, k=3)
        if not docs:
            raise HTTPException(
                status_code=404,
                detail=f"No papers found in category '{request.category}' for this query."
            )
        context = [d.page_content for d in docs]
        answer = generator.generate(request.query, "\n".join(context))

        scores = evaluator.evaluate_response(request.query, answer, context)
        needs_healing, failed_metrics, scores_dict = healer.check_need_healing(scores)

        healing_attempts_count = 0
        final_answer = answer
        final_scores = scores_dict
        winning_strategy = None

        if needs_healing:
            result = healer.heal(
                request.query, retriever, generator, evaluator, request.category, scores
            )
            healing_attempts_count = healer.max_attempts
            if result:
                attempts_log = result.get('attempts', [])
                if result['improved']:
                    final_answer = result['answer']
                    final_scores = result['scores']
                    winning_strategy = result.get('winning_strategy')
        else:
            # No healing needed — still log attempt 0 so every query has a journey entry
            attempts_log = [{
                'attempt_number': 0,
                'strategy_name': None,
                'scores': scores_dict,
                'composite': compute_composite_score(scores_dict),
                'is_winner': True,
            }]

        composite = compute_composite_score(final_scores)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Pipeline failed for query: {request.query}")
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong processing this query: {str(e)}"
        )

    try:
        db = next(get_db())
        log_entry = QueryLog(
            query=request.query,
            category=request.category,
            query_type=query_type,
            answer=final_answer,
            faithfulness=final_scores.get('faithfulness'),
            answer_relevancy=final_scores.get('answer_relevancy'),
            context_relevance=final_scores.get('nv_context_relevance'),
            composite_score=composite,
            healing_triggered=needs_healing,
            healing_attempts=healing_attempts_count,
            winning_strategy=winning_strategy,
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

        for a in attempts_log:
            s = a['scores']
            db.add(HealingAttempt(
                query_log_id=log_entry.id,
                attempt_number=a['attempt_number'],
                strategy_name=a['strategy_name'],
                is_winner=a.get('is_winner', False),
                faithfulness=s.get('faithfulness'),
                answer_relevancy=s.get('answer_relevancy'),
                context_relevance=s.get('nv_context_relevance'),
                composite_score=a['composite'],
            ))
        db.commit()
        db.close()
    except Exception:
        logger.exception("Failed to log query to Postgres — continuing anyway.")

    return QueryResponse(
        query=request.query,
        category=request.category,
        query_type=query_type,
        answer=final_answer,
        faithfulness=final_scores.get('faithfulness'),
        answer_relevancy=final_scores.get('answer_relevancy'),
        context_relevance=final_scores.get('nv_context_relevance'),
        composite_score=composite,
        healing_triggered=needs_healing,
        healing_attempts=healing_attempts_count,
        winning_strategy=winning_strategy,
    )
