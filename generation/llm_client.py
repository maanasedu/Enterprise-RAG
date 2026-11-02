import os
import time
from groq import Groq
from typing import List, Dict
from generation.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from generation.context_budget import build_context


class RAGGenerator:
    def __init__(self, provider: str = 'groq'):
        self.provider = provider
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        # Use 70B for better quality — still free on Groq
        self.model = 'openai/gpt-oss-120b'
        self._retry_attempts = 3
        self._retry_delay = 2

    def generate(self, question: str, retrieved_chunks: List[Dict]) -> Dict:
        # Build context from whole, ranked chunks within a character budget
        # instead of joining everything and then slicing the raw string
        # (which could cut a chunk off mid-sentence regardless of rank).
        context, chunks_included, context_truncated = build_context(
            retrieved_chunks, max_chars=6000
        )
        if not context:
            context = 'No context available.'

        user_message = RAG_USER_TEMPLATE.format(
            context=context,
            question=question
        )

        for attempt in range(self._retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': RAG_SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_message}
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    timeout=30,
                    reasoning_effort='low'
                )
                answer = response.choices[0].message.content

                # Validate answer is not empty or malformed
                if not answer or len(answer.strip()) < 5:
                    raise ValueError('Empty or too-short response from LLM')

                return {
                    'answer': answer.strip(),
                    'context': context,
                    'question': question,
                    'context_chunks_included': chunks_included,
                    'context_truncated': context_truncated,
                }

            except Exception as e:
                print(f'Generation attempt {attempt + 1} failed: {e}')
                if attempt < self._retry_attempts - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    # Final fallback — never crash the API
                    return {
                        'answer': 'I was unable to generate a response at this time. Please try again.',
                        'context': context,
                        'question': question,
                        'context_chunks_included': chunks_included,
                        'context_truncated': context_truncated,
                        'error': str(e)
                    }
