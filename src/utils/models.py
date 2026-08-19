from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from src.utils.db import Base

class QueryLog(Base):
    """Stores every query, its final answer, scores, and healing outcome."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    category = Column(String, nullable=False)
    query_type = Column(String, nullable=True)
    answer = Column(Text, nullable=False)

    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_relevance = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=True)

    healing_triggered = Column(Boolean, default=False)
    healing_attempts = Column(Integer, default=0)
    winning_strategy = Column(String, nullable=True)

    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class HealingAttempt(Base):
    """Stores each individual attempt (including attempt 0 = the original,
    pre-healing result) for a query, so we can chart the healing journey and
    before/after comparisons."""
    __tablename__ = "healing_attempts"

    id = Column(Integer, primary_key=True, index=True)
    query_log_id = Column(Integer, ForeignKey("query_logs.id"), nullable=False)

    attempt_number = Column(Integer, nullable=False)  # 0 = original, 1+ = healing attempts
    strategy_name = Column(String, nullable=True)  # null for attempt 0
    is_winner = Column(Boolean, default=False)  # True for whichever attempt was returned to the user

    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_relevance = Column(Float, nullable=True)
    composite_score = Column(Float, nullable=True)

    timestamp = Column(DateTime(timezone=True), server_default=func.now())
