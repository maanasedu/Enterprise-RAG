"""Custom LLM-as-judge evaluator for this project's RAG pipeline.

IMPORTANT: despite the metric names below (faithfulness, answer_relevancy,
context_precision, context_recall) matching the terminology used by the
`ragas` package, this is NOT the ragas package. It is a hand-written
LLM-as-judge implementation that asks the same Groq model used for
generation to score each metric with a single numeric prompt. `ragas` is
listed in requirements.txt but is not imported or used anywhere in this
codebase today. Treat scores from this module as "our custom judge's
opinion", not as RAGAS's own (differently-implemented, differently
validated) metrics of the same name — the two are not directly comparable,
and a future V1 benchmark should decide explicitly whether to keep this
custom judge, switch to the real `ragas` package, or report both.
"""
import os
import time
import pandas as pd
from groq import Groq
from typing import List, Dict


class RAGASEvaluator:
    """Custom LLM-as-judge evaluator — NOT the `ragas` package.

    See the module docstring above for why this class (and its metric
    names) can be easily mistaken for the real `ragas` package.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = 'openai/gpt-oss-120b'
        self._call_count = 0

    def _llm_score(self, prompt: str, retries: int = 3):
        """Returns a float score in [0,1] on success, or None on failure.
        Never silently returns a default/fallback score — callers and the
        aggregation layer are responsible for treating None as a real
        missing measurement (NaN), not a fake data point.
        """
        import re
        for attempt in range(retries):
            try:
                # Rate limit: Groq free tier allows ~30 RPM on 120B
                if self._call_count > 0 and self._call_count % 25 == 0:
                    time.sleep(60)  # Pause every 25 calls

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    # NOTE: max_tokens on Groq bounds visible output tokens
                    # PLUS reasoning tokens combined (confirmed in Groq's
                    # official API reference). gpt-oss-120b is a reasoning
                    # model, so even at reasoning_effort='low' it typically
                    # emits some internal reasoning tokens before the final
                    # visible answer. The previous max_tokens=10 was small
                    # enough that reasoning alone could exhaust the budget,
                    # leaving message.content empty on every call — which is
                    # why every score silently fell back to 0.5. 500 leaves
                    # comfortable headroom for reasoning + a one-digit answer.
                    max_tokens=500,
                    timeout=30,
                    reasoning_effort='low',  # 'none' is rejected by Groq for gpt-oss models (400 error); low/medium/high are the valid options
                )
                self._call_count += 1
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                text = (message.content or '').strip()

                if not text:
                    reasoning_preview = getattr(message, 'reasoning', None)
                    reasoning_len = len(reasoning_preview) if reasoning_preview else 0
                    print(f'  [eval] Empty content on attempt {attempt + 1}/{retries} '
                          f'(finish_reason={finish_reason}, reasoning_chars={reasoning_len}).')
                    if attempt < retries - 1:
                        time.sleep(3 * (attempt + 1))
                        continue
                    print('  [eval] All retries exhausted with empty content — recording as FAILED, not 0.5.')
                    return None

                matches = re.findall(r'\d+\.?\d*', text)
                if matches:
                    score = float(matches[0])
                    return min(max(score, 0.0), 1.0)

                print(f'  [eval] Could not parse a numeric score from response on attempt '
                      f'{attempt + 1}/{retries}: {text!r} (finish_reason={finish_reason}).')
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                print('  [eval] All retries exhausted with unparseable content — recording as FAILED, not 0.5.')
                return None

            except Exception as e:
                print(f'  [eval] LLM call attempt {attempt + 1}/{retries} raised: {e}')
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
        print('  [eval] All retries exhausted with exceptions — recording as FAILED, not 0.5.')
        return None  # Explicit failure signal on complete failure — never a fake default

    def _score_faithfulness(self, answer: str, context: str) -> float:
        prompt = f"""Rate how faithful this answer is to the context.
Faithful means: every claim in the answer is supported by the context.
Score from 0.0 (completely unfaithful) to 1.0 (completely faithful).

Context: {context}
Answer: {answer[:300]}

Reply with only a number between 0 and 1:"""
        return self._llm_score(prompt)

    def _score_answer_relevancy(self, answer: str, question: str) -> float:
        prompt = f"""Rate how relevant this answer is to the question.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

Question: {question}
Answer: {answer[:300]}

Reply with only a number between 0 and 1:"""
        return self._llm_score(prompt)

    def _score_context_precision(self, question: str, context: str) -> float:
        prompt = f"""Rate how precisely the context contains information needed to answer the question.
Score from 0.0 (context is useless) to 1.0 (context perfectly answers the question).

Question: {question}
Context: {context}

Reply with only a number between 0 and 1:"""
        return self._llm_score(prompt)

    def _score_context_recall(self, ground_truth: str, context: str) -> float:
        prompt = f"""Rate how well the context covers the ground truth answer.
Score from 0.0 (context misses everything) to 1.0 (context covers everything).

Ground Truth: {ground_truth[:300]}
Context: {context}

Reply with only a number between 0 and 1:"""
        return self._llm_score(prompt)

    def evaluate_single(self, qa: Dict) -> Dict:
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        contexts = qa.get('contexts', [])
        ground_truth = qa.get('ground_truth', '')

        # 2000 chars is enough to hold the actual retrieved evidence for
        # most chunks (previously 600, which combined with per-prompt
        # truncation below was cutting off supporting evidence and likely
        # deflating faithfulness scores).
        context_str = '\n'.join(contexts[:3])[:2000]

        return {
            'question': question,
            'faithfulness': self._score_faithfulness(answer, context_str),
            'answer_relevancy': self._score_answer_relevancy(answer, question),
            'context_precision': self._score_context_precision(question, context_str),
            'context_recall': self._score_context_recall(ground_truth, context_str),
        }

    def evaluate(self, qa_pairs: List[Dict]) -> pd.DataFrame:
        results = []
        error_samples = 0
        metric_names = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        for i, qa in enumerate(qa_pairs):
            print(f'Evaluating {i + 1}/{len(qa_pairs)}...', end='\r')
            try:
                scores = self.evaluate_single(qa)
                # _score_* methods return None on failure (see _llm_score).
                # Convert to NaN so pandas correctly excludes these from
                # mean()/describe() instead of silently averaging in a
                # fake number.
                for m in metric_names:
                    if scores.get(m) is None:
                        scores[m] = float('nan')
                results.append(scores)
            except Exception as e:
                print(f'\n[eval] Sample {i + 1} raised an unexpected error — recording as FAILED (NaN), not 0.5: {e}')
                error_samples += 1
                results.append({
                    'question': qa.get('question', ''),
                    'faithfulness': float('nan'),
                    'answer_relevancy': float('nan'),
                    'context_precision': float('nan'),
                    'context_recall': float('nan'),
                })
        df = pd.DataFrame(results)
        if error_samples:
            print(f'\n[eval] {error_samples}/{len(qa_pairs)} samples raised unexpected errors during evaluation.')
        return df

    def print_summary(self, df: pd.DataFrame):
        print('\n=== Custom LLM-as-Judge Evaluation Summary (RAGAS-style metric names; NOT the ragas package) ===')
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            if metric not in df.columns:
                continue
            valid = df[metric].dropna()
            n_missing = int(df[metric].isna().sum())
            if len(valid) == 0:
                print(f'{metric:20s}: NO VALID SCORES — all {len(df)} samples failed to score. Do not report this metric.')
                continue
            mean = valid.mean()
            warning = ''
            if valid.nunique() == 1:
                warning = '  *** WARNING: every valid score is identical — this usually means every call hit a fallback/parsing failure. Do NOT report this number until investigated. ***'
            print(f'{metric:20s}: {mean:.3f}  (n={len(valid)}/{len(df)} scored, {n_missing} failed){warning}')