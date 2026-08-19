import numpy as np
from src.evaluation.strategies import select_strategy_order
from src.evaluation.scoring import compute_composite_score

class RAGHealer:
    """Monitors RAG performance and triggers corrective actions using a set of
    named retrieval strategies, trying them in an order chosen based on the
    query and which metrics failed. Tracks every attempt (not just the winner)
    so the full healing journey can be logged and charted."""

    def __init__(self, thresholds=None, max_attempts=2):
        self.thresholds = thresholds or {
            'faithfulness': 0.4,  # lowered from 0.6 -- llama3.2:1b judge trends conservative/harsh on faithfulness, confirmed via repeated fixed-input tests
            'answer_relevancy': 0.6,
            'nv_context_relevance': 0.5,
        }
        self.max_attempts = max_attempts

    def _extract_scores(self, scores):
        if hasattr(scores, 'scores'):
            data = scores.scores
            if isinstance(data, list) and len(data) > 0:
                keys = data[0].keys()
                result = {}
                for k in keys:
                    values = [d[k] for d in data if d.get(k) is not None]
                    result[k] = np.nanmean(values) if values else float('nan')
                return result
            return data
        elif hasattr(scores, 'to_dict'):
            return scores.to_dict()
        elif isinstance(scores, dict):
            return scores
        return {}

    def check_need_healing(self, scores):
        scores_dict = self._extract_scores(scores)
        failed = [
            metric for metric, threshold in self.thresholds.items()
            if not np.isnan(scores_dict.get(metric, 1.0)) and scores_dict.get(metric, 1.0) < threshold
        ]
        return len(failed) > 0, failed, scores_dict

    def _is_fully_resolved(self, scores_dict):
        for metric, threshold in self.thresholds.items():
            value = scores_dict.get(metric, float('nan'))
            if np.isnan(value) or value < threshold:
                return False
        return True

    def _score_sum(self, scores_dict):
        values = [scores_dict.get(m) for m in self.thresholds]
        valid = [v for v in values if v is not None and not np.isnan(v)]
        return sum(valid) if valid else -1

    def heal(self, original_query, retriever, generator, evaluator, category, original_scores):
        """
        Tries named strategies in an order chosen based on the failure type, up to
        max_attempts. Returns the best result AND a full log of every attempt tried
        (including attempt 0, the original), so the healing journey can be charted.
        """
        needs_healing, failed_metrics, original_dict = self.check_need_healing(original_scores)
        if not needs_healing:
            return None

        print(f"!!! [HEALING] Triggered for query: {original_query}")
        print(f"    Failed metrics: {failed_metrics} | Original scores: {original_dict}")

        strategy_order = select_strategy_order(original_query, failed_metrics)

        # Log attempt 0 = the original, pre-healing result
        all_attempts = [{
            'attempt_number': 0,
            'strategy_name': None,
            'scores': original_dict,
            'composite': compute_composite_score(original_dict),
        }]

        best_answer = None
        best_scores = original_dict
        best_score_sum = self._score_sum(original_dict)
        best_strategy_name = None
        best_attempt_number = 0

        for attempt in range(1, self.max_attempts + 1):
            strategy = strategy_order[(attempt - 1) % len(strategy_order)]
            print(f"    Attempt {attempt} using strategy: {strategy.name}")

            docs = strategy.run(original_query, retriever, generator, category, k=5)
            if not docs:
                print(f"    Strategy '{strategy.name}' returned no documents, skipping.")
                continue

            context = [d.page_content for d in docs]
            answer = generator.generate(original_query, "\n".join(context))

            new_scores = evaluator.evaluate_response(original_query, answer, context)
            _, new_failed, new_dict = self.check_need_healing(new_scores)
            new_score_sum = self._score_sum(new_dict)

            all_attempts.append({
                'attempt_number': attempt,
                'strategy_name': strategy.name,
                'scores': new_dict,
                'composite': compute_composite_score(new_dict),
            })

            print(f"    Attempt {attempt} ({strategy.name}): {new_dict} (failed: {new_failed})")

            if new_score_sum > best_score_sum:
                best_answer = answer
                best_scores = new_dict
                best_score_sum = new_score_sum
                best_strategy_name = strategy.name
                best_attempt_number = attempt

            _, failed_metrics, _ = self.check_need_healing(best_scores)

            if self._is_fully_resolved(best_scores):
                print(f"    Fully healed after attempt {attempt} (winning strategy: {best_strategy_name}).")
                break

        for a in all_attempts:
            a['is_winner'] = (a['attempt_number'] == best_attempt_number)

        return {
            'answer': best_answer,
            'scores': best_scores,
            'improved': best_answer is not None,
            'winning_strategy': best_strategy_name,
            'attempts': all_attempts,
        }
