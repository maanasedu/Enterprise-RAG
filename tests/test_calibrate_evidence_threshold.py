"""Integration tests for scripts/calibrate_evidence_threshold.py.

No real external calls: Cohere is mocked (same pattern as
tests/test_retrieval_eval.py), the embedder is a deterministic fake, and
sockets are blocked for the duration of every test. The pipeline under
test -- QueryRouter, HybridRetriever, CohereReranker, EvidenceEvaluator,
AdaptiveRetriever -- is the real, unmodified Dimension 2 production code;
only the Cohere HTTP call is faked.
"""
import json
import os
import socket
from types import SimpleNamespace

import pytest

import scripts.calibrate_evidence_threshold as cal
from evaluation import retrieval_eval as ev
from ingestion.chunker import Chunk
from retrieval.bm25_retriever import BM25Retriever
from retrieval.identity import chunk_identity_key
from retrieval.reranker import CohereReranker
from retrieval.vector_store import FAISSStore


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError('Network access attempted during a unit test')
    monkeypatch.setattr(socket.socket, 'connect', _blocked)


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


# A tiny corpus: one gold chunk per "sufficient" question, tagged with a
# unique marker the mock reranker looks for.
CHUNK_TEXT = {
    0: 'filler passage about generic content',
    1: 'the answer to alpha is forty two [ALPHA]',
    2: 'the answer to beta is the moon [BETA]',
    3: 'another filler passage with nothing useful',
    4: 'topically related but not the real answer [BETA_DECOY]',
}


def _make_corpus():
    chunks = [Chunk(text=t, metadata={'filename': f'doc{i}.pdf'}, strategy='recursive')
              for i, t in CHUNK_TEXT.items()]
    store = FAISSStore(dim=4)
    store.add([[1.0, 0.0, 0.0, 0.0] for _ in chunks], chunks)
    bm25 = BM25Retriever()
    bm25.index([{'text': c.text, 'metadata': c.metadata} for c in chunks])
    gold_key = {i: chunk_identity_key({'text': CHUNK_TEXT[i], 'metadata': {'filename': f'doc{i}.pdf'}})
               for i in CHUNK_TEXT}
    return store, bm25, gold_key


def _mock_reranker(score_fn):
    """score_fn(query, document_text) -> float. Every call is scored
    independently; no rate-limit sleep."""
    import cohere as cohere_mod
    reranker = CohereReranker.__new__(CohereReranker)
    reranker.client = cohere_mod.Client('fake-key-for-test')
    reranker.model = 'rerank-english-v3.0'
    reranker._last_call = 0
    reranker._min_interval = 0

    def fake_rerank(query, documents, top_n, model):
        scored = sorted(range(len(documents)), key=lambda i: -score_fn(query, documents[i]))
        results = [SimpleNamespace(index=i, relevance_score=score_fn(query, documents[i]))
                  for i in scored[:top_n]]
        return SimpleNamespace(results=results)

    reranker.client.rerank = fake_rerank
    return reranker


def _q(qid, question, status, gold_keys=(), category='factual'):
    return ev.BenchmarkQuestion(
        id=qid, category=category, question=question,
        answerable=(status == 'sufficient'), evidence_status=status,
        gold_chunk_ids=tuple(gold_keys), gold_document_ids=('d.pdf',) if gold_keys else (),
    )


# ── collect_scores: real routing/retrieval/rerank/evidence wiring ──────────
def test_collect_scores_uses_real_pipeline_and_one_round_per_question(monkeypatch):
    monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
    store, bm25, gold_key = _make_corpus()

    def score_fn(query, doc):
        if '[ALPHA]' in doc and 'alpha' in query.lower():
            return 0.95
        if '[BETA]' in doc and 'beta' in query.lower():
            return 0.9
        return 0.01

    reranker = _mock_reranker(score_fn)
    from retrieval.hybrid_retriever import HybridRetriever
    hybrid = HybridRetriever(store, bm25, FakeEmbedder(), use_reranker=False)
    hybrid.reranker = reranker

    questions = [
        _q('s_alpha', 'What is the answer to alpha?', 'sufficient', [gold_key[1]]),
        _q('s_beta', 'What is the answer to beta?', 'sufficient', [gold_key[2]]),
        _q('i_gamma', 'What is the answer to gamma?', 'insufficient'),
    ]
    collected = cal.collect_scores(questions, hybrid, evidence_top_k=3,
                                   top_k_rerank=5, top_k_retrieve=50)

    by_id = {c.record.question_id: c for c in collected}
    assert by_id['s_alpha'].record.top_score == pytest.approx(0.95)
    assert by_id['s_beta'].record.top_score == pytest.approx(0.9)
    assert by_id['i_gamma'].record.top_score == pytest.approx(0.01)
    assert all(c.record.evaluated for c in collected)
    assert all(c.record.routing_path == 'direct' for c in collected)  # none of these decompose


def test_collect_scores_routes_multi_part_questions_through_decomposition(monkeypatch):
    monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
    store, bm25, gold_key = _make_corpus()
    reranker = _mock_reranker(lambda q, d: 0.9 if '[ALPHA]' in d or '[BETA]' in d else 0.1)
    from retrieval.hybrid_retriever import HybridRetriever
    hybrid = HybridRetriever(store, bm25, FakeEmbedder(), use_reranker=False)
    hybrid.reranker = reranker

    q = _q('multi', 'What is alpha? What is beta?', 'sufficient', [gold_key[1]])
    collected = cal.collect_scores([q], hybrid, evidence_top_k=3, top_k_rerank=5, top_k_retrieve=50)
    assert collected[0].record.routing_path == 'decomposed'


def test_collect_scores_never_retries_because_threshold_never_refuses(monkeypatch):
    monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
    store, bm25, gold_key = _make_corpus()
    reranker = _mock_reranker(lambda q, d: 0.0001)  # deliberately "weak" score
    rerank_calls = {'n': 0}
    original_rerank = reranker.client.rerank

    def counted_rerank(*args, **kwargs):
        rerank_calls['n'] += 1
        return original_rerank(*args, **kwargs)

    reranker.client.rerank = counted_rerank
    from retrieval.hybrid_retriever import HybridRetriever
    hybrid = HybridRetriever(store, bm25, FakeEmbedder(), use_reranker=False)
    hybrid.reranker = reranker

    q = _q('weak', 'What is the answer to alpha?', 'sufficient', [gold_key[1]])
    collected = cal.collect_scores([q], hybrid, evidence_top_k=3, top_k_rerank=5, top_k_retrieve=50)
    assert collected[0].record.top_score == pytest.approx(0.0001)
    # exactly one rerank call -> no retry, even though the score is very low
    assert rerank_calls['n'] == 1


# ── build_report ─────────────────────────────────────────────────────────
def test_build_report_shape_and_calibration_embedding():
    from evaluation.threshold_calibration import CalibrationRecord
    collected = [
        cal.CollectedScore(CalibrationRecord('s1', 'sufficient', 0.9, True, 'factual', 'direct'), (0.9,), 3),
        cal.CollectedScore(CalibrationRecord('i1', 'insufficient', 0.1, True, 'factual', 'direct'), (0.1,), 3),
    ]
    report = cal.build_report(
        questions=[SimpleNamespace(id='s1'), SimpleNamespace(id='i1')],
        collected=collected, evidence_top_k=3, top_k_rerank=5, top_k_retrieve=50,
        benchmark_path='data/evaluation/calibration.json', processed_dir='data/processed',
    )
    assert report['split'] == 'calibration'
    assert report['n_questions'] == 2
    assert report['rerank_fallback_questions'] == 0
    assert len(report['per_question']) == 2
    assert report['calibration']['selected_threshold'] == 0.9
    assert report['protocol']['top_k_rerank'] == 5


# ── main(): CLI wiring, guards, file output ─────────────────────────────
@pytest.fixture
def calibration_fixture(tmp_path, monkeypatch):
    store, bm25, gold_key = _make_corpus()
    reranker = _mock_reranker(
        lambda q, d: 0.9 if '[ALPHA]' in d else (0.85 if '[BETA]' in d else 0.02))

    bench_dir = tmp_path / 'evaluation'
    bench_dir.mkdir()
    items = [
        {'id': 's_alpha', 'category': 'factual', 'question': 'alpha?', 'answerable': True,
         'evidence_status': 'sufficient', 'gold_chunk_ids': [gold_key[1]], 'gold_document_ids': ['doc1.pdf']},
        {'id': 's_beta', 'category': 'factual', 'question': 'beta?', 'answerable': True,
         'evidence_status': 'sufficient', 'gold_chunk_ids': [gold_key[2]], 'gold_document_ids': ['doc2.pdf']},
        {'id': 'i_gamma', 'category': 'unanswerable', 'question': 'gamma?', 'answerable': False,
         'evidence_status': 'insufficient', 'gold_chunk_ids': [], 'gold_document_ids': []},
    ]
    (bench_dir / 'calibration.json').write_text(json.dumps(items))

    monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
    monkeypatch.setattr(cal, 'load_production_components',
                        lambda processed_dir: (store, bm25, FakeEmbedder()))
    monkeypatch.setattr(cal, 'build_retrievers',
                        lambda s, b, e, configs: {'hybrid_cohere': _rig(s, b, e, reranker)})
    return SimpleNamespace(bench_dir=str(bench_dir), tmp_path=tmp_path, reranker=reranker)


def _rig(store, bm25, embedder, reranker):
    from retrieval.hybrid_retriever import HybridRetriever
    h = HybridRetriever(store, bm25, embedder, use_reranker=False)
    h.reranker = reranker
    return h


def test_main_writes_report_and_prints_summary(calibration_fixture, capsys):
    out_path = os.path.join(calibration_fixture.tmp_path, 'out.json')
    rc = cal.main([
        '--benchmark-dir', calibration_fixture.bench_dir,
        '--processed-dir', 'unused',   # load_production_components is monkeypatched
        '--output', out_path,
    ])
    assert rc == 0
    report = json.load(open(out_path))
    assert report['n_questions'] == 3
    assert report['calibration']['counts']['n_sufficient'] == 2
    assert report['calibration']['counts']['n_insufficient'] == 1
    captured = capsys.readouterr()
    assert 'Selected threshold' in captured.out


def test_main_requires_cohere_api_key(calibration_fixture, monkeypatch, capsys):
    monkeypatch.delenv('COHERE_API_KEY', raising=False)
    rc = cal.main(['--benchmark-dir', calibration_fixture.bench_dir,
                   '--output', os.path.join(calibration_fixture.tmp_path, 'out.json')])
    assert rc == 2
    assert 'COHERE_API_KEY' in capsys.readouterr().err


def test_main_refuses_on_partial_rerank_fallback(calibration_fixture, monkeypatch, capsys):
    # Second question's rerank call raises -> CohereReranker's own fallback
    # kicks in (reranked=False) for that one question only.
    calls = {'n': 0}
    original = calibration_fixture.reranker.client.rerank

    def flaky(query, documents, top_n, model):
        calls['n'] += 1
        if calls['n'] == 2:
            raise RuntimeError('simulated Cohere outage')
        return original(query, documents, top_n, model)

    calibration_fixture.reranker.client.rerank = flaky
    rc = cal.main(['--benchmark-dir', calibration_fixture.bench_dir,
                   '--output', os.path.join(calibration_fixture.tmp_path, 'out.json')])
    assert rc == 2
    assert 'fell back' in capsys.readouterr().err


def test_main_propagates_missing_index_as_documented_error(tmp_path, monkeypatch):
    # No monkeypatch of load_production_components here -> the real one runs
    # against an empty processed-dir, exercising the same IndexNotAvailableError
    # Dimension 1's own loader raises (reused, not reimplemented).
    monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
    bench_dir = tmp_path / 'evaluation'
    bench_dir.mkdir()
    (bench_dir / 'calibration.json').write_text('[]')
    rc = cal.main(['--benchmark-dir', str(bench_dir),
                   '--processed-dir', str(tmp_path / 'nope'),
                   '--output', str(tmp_path / 'out.json')])
    assert rc == 2
