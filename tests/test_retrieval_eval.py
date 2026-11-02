"""Tests for evaluation/retrieval_eval.py (Dimension 1 retrieval evaluation).

No real external calls: Cohere is mocked everywhere, the embedder is a
deterministic fake, and sockets are blocked for the duration of every test.
The retrieval stack under test is the real production code (FAISSStore,
BM25Retriever, HybridRetriever, CohereReranker).
"""
import json
import math
import os
import socket
from types import SimpleNamespace

import cohere
import pytest

from evaluation import retrieval_eval as ev
from ingestion.chunker import Chunk
from retrieval.bm25_retriever import BM25Retriever
from retrieval.identity import chunk_identity_key
from retrieval.reranker import CohereReranker
from retrieval.vector_store import FAISSStore


# --------------------------------------------------------------------------
# Global safety: no network, no real Cohere key
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_network_no_key(monkeypatch):
    monkeypatch.delenv('COHERE_API_KEY', raising=False)

    def _blocked(*args, **kwargs):
        raise AssertionError('Network access attempted during a unit test')

    monkeypatch.setattr(socket.socket, 'connect', _blocked)


# --------------------------------------------------------------------------
# Metric tests
# --------------------------------------------------------------------------
def _ranked(n, gold_positions, gold_prefix='G'):
    """Ranking of n distinct keys with gold keys placed at 1-based positions."""
    out, gold = [], []
    for i in range(1, n + 1):
        if i in gold_positions:
            key = f'{gold_prefix}{i}'
            gold.append(key)
        else:
            key = f'x{i}'
        out.append(key)
    return out, gold


class TestRecall:
    def test_single_gold_boundaries(self):
        ranked, gold = _ranked(12, {5})
        assert ev.recall_at_k(ranked, gold, 5) == 1.0
        assert ev.recall_at_k(ranked, gold, 10) == 1.0

        ranked, gold = _ranked(12, {6})
        assert ev.recall_at_k(ranked, gold, 5) == 0.0
        assert ev.recall_at_k(ranked, gold, 10) == 1.0

        ranked, gold = _ranked(12, {11})
        assert ev.recall_at_k(ranked, gold, 10) == 0.0

    def test_multi_gold_is_fractional(self):
        ranked, gold = _ranked(12, {2, 8})
        assert ev.recall_at_k(ranked, gold, 5) == 0.5
        assert ev.recall_at_k(ranked, gold, 10) == 1.0

    def test_all_gold_is_strict_for_multi_doc(self):
        ranked, gold = _ranked(12, {2, 8})
        assert ev.all_gold_at_k(ranked, gold, 5) == 0.0
        assert ev.all_gold_at_k(ranked, gold, 10) == 1.0

    def test_missing_gold_scores_zero(self):
        assert ev.recall_at_k(['a', 'b'], ['zzz'], 5) == 0.0

    def test_duplicate_retrieved_keys_do_not_inflate_recall(self):
        assert ev.recall_at_k(['g', 'g', 'g'], ['g', 'h'], 5) == 0.5

    def test_empty_gold_is_rejected(self):
        with pytest.raises(ValueError):
            ev.recall_at_k(['a'], [], 5)


class TestMRR:
    def test_reciprocal_rank_positions(self):
        ranked, gold = _ranked(10, {3})
        assert ev.reciprocal_rank(ranked, gold) == pytest.approx(1 / 3)
        ranked, gold = _ranked(10, {1})
        assert ev.reciprocal_rank(ranked, gold) == 1.0

    def test_uses_first_gold_when_multiple(self):
        ranked, gold = _ranked(10, {2, 4})
        assert ev.reciprocal_rank(ranked, gold) == 0.5

    def test_zero_when_no_gold_within_depth(self):
        ranked, gold = _ranked(12, {11})
        assert ev.reciprocal_rank(ranked, gold, depth=10) == 0.0
        assert ev.reciprocal_rank(['a'], ['b']) == 0.0

    def test_duplicates_do_not_shift_rank(self):
        # 'a' repeated: after de-duplication the gold is at rank 2, not 3.
        assert ev.reciprocal_rank(['a', 'a', 'g'], ['g']) == 0.5


class TestNDCG:
    def test_perfect_single_gold(self):
        ranked, gold = _ranked(5, {1})
        assert ev.ndcg_at_k(ranked, gold, 5) == pytest.approx(1.0)

    def test_single_gold_discount(self):
        ranked, gold = _ranked(5, {2})
        assert ev.ndcg_at_k(ranked, gold, 5) == pytest.approx(1 / math.log2(3))
        ranked, gold = _ranked(5, {5})
        assert ev.ndcg_at_k(ranked, gold, 5) == pytest.approx(1 / math.log2(6))

    def test_gold_beyond_k_scores_zero(self):
        ranked, gold = _ranked(10, {6})
        assert ev.ndcg_at_k(ranked, gold, 5) == 0.0

    def test_multi_gold_perfect_and_partial(self):
        ranked, gold = _ranked(5, {1, 2})
        assert ev.ndcg_at_k(ranked, gold, 5) == pytest.approx(1.0)

        ranked, gold = _ranked(5, {1, 4})
        expected = (1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        assert ev.ndcg_at_k(ranked, gold, 5) == pytest.approx(expected)

        ranked, gold = _ranked(5, {3})
        gold = gold + ['missing']  # one of two gold chunks never retrieved
        expected = (1 / math.log2(4)) / (1 + 1 / math.log2(3))
        assert ev.ndcg_at_k(ranked, gold, 5) == pytest.approx(expected)

    def test_ideal_dcg_capped_at_k(self):
        ranked, gold = _ranked(5, {1, 2, 3, 4, 5})
        gold = gold + ['g6', 'g7']  # 7 gold chunks, only 5 slots
        assert ev.ndcg_at_k(ranked, gold, 5) == pytest.approx(1.0)


class TestScoreAndAggregate:
    def test_score_question_bundle(self):
        ranked, gold = _ranked(10, {2, 8})
        s = ev.score_question(ranked, gold)
        assert s['recall@5'] == 0.5 and s['recall@10'] == 1.0
        assert s['mrr'] == 0.5
        assert 0 < s['ndcg@5'] < 1
        assert s['all_gold@5'] == 0.0 and s['all_gold@10'] == 1.0

    def test_aggregate_is_macro_mean_with_deterministic_ci(self):
        rows = [
            {'recall@5': 1.0, 'recall@10': 1.0, 'mrr': 1.0, 'ndcg@5': 1.0,
             'all_gold@5': 1.0, 'all_gold@10': 1.0},
            {'recall@5': 0.0, 'recall@10': 1.0, 'mrr': 0.0, 'ndcg@5': 0.0,
             'all_gold@5': 0.0, 'all_gold@10': 1.0},
        ]
        a1, a2 = ev.aggregate(rows), ev.aggregate(rows)
        assert a1['metrics']['recall@5'] == 0.5 and a1['metrics']['recall@10'] == 1.0
        assert a1['ci95'] == a2['ci95']
        lo, hi = a1['ci95']['recall@5']
        assert 0.0 <= lo <= 0.5 <= hi <= 1.0

    def test_aggregate_rejects_empty(self):
        with pytest.raises(ValueError):
            ev.aggregate([])


# --------------------------------------------------------------------------
# Benchmark loading / unanswerable + insufficient handling
# --------------------------------------------------------------------------
def _item(qid, category='factual', status='sufficient', gold=('g1',), docs=('d.pdf',),
          answerable=True):
    return {
        'id': qid, 'category': category, 'question': f'question {qid}',
        'answerable': answerable, 'evidence_status': status,
        'reference_answer': None, 'attack_type': None,
        'gold_chunk_ids': list(gold), 'gold_document_ids': list(docs),
    }


def _write(tmp_path, name, items):
    path = tmp_path / name
    path.write_text(json.dumps(items))
    return str(path)


class TestLoading:
    def test_scorable_only_when_gold_chunks_exist(self, tmp_path):
        path = _write(tmp_path, 'calibration.json', [
            _item('a'),
            _item('b', 'multi_document', gold=('g1', 'g2')),
            _item('c', 'unanswerable', 'insufficient', gold=(), docs=(), answerable=False),
            _item('d', 'ambiguous', 'ambiguous', gold=(), docs=('x.pdf', 'y.pdf')),
            _item('e', 'adversarial', 'insufficient', gold=(), docs=(), answerable=False),
        ])
        qs = ev.load_questions(path)
        assert [q.scorable for q in qs] == [True, True, False, False, False]
        summary = ev.summarize_exclusions(qs)
        assert summary['n_scored'] == 2 and summary['n_excluded'] == 3
        assert summary['excluded_by_category'] == {
            'adversarial': 1, 'ambiguous': 1, 'unanswerable': 1}
        assert summary['excluded_by_evidence_status'] == {'ambiguous': 1, 'insufficient': 2}

    def test_sufficient_without_gold_is_a_data_error(self, tmp_path):
        path = _write(tmp_path, 'x.json', [_item('a', gold=())])
        with pytest.raises(ev.BenchmarkDataError):
            ev.load_questions(path)

    def test_insufficient_with_gold_is_a_data_error(self, tmp_path):
        path = _write(tmp_path, 'x.json', [_item('a', status='insufficient')])
        with pytest.raises(ev.BenchmarkDataError):
            ev.load_questions(path)

    def test_duplicate_ids_and_missing_fields_rejected(self, tmp_path):
        with pytest.raises(ev.BenchmarkDataError):
            ev.load_questions(_write(tmp_path, 'd.json', [_item('a'), _item('a')]))
        bad = _item('a')
        del bad['gold_chunk_ids']
        with pytest.raises(ev.BenchmarkDataError):
            ev.load_questions(_write(tmp_path, 'm.json', [bad]))

    def test_missing_file_message(self, tmp_path):
        with pytest.raises(ev.BenchmarkDataError):
            ev.load_questions(str(tmp_path / 'nope.json'))

    def test_gold_must_exist_in_corpus(self):
        q = ev.BenchmarkQuestion('a', 'factual', 'q', True, 'sufficient',
                                 ('text-hash:missing',), ('d',))
        with pytest.raises(ev.BenchmarkDataError):
            ev.assert_gold_in_corpus([q], [{'text': 'hello', 'metadata': {}}])
        ok_key = chunk_identity_key({'text': 'hello', 'metadata': {}})
        q_ok = ev.BenchmarkQuestion('a', 'factual', 'q', True, 'sufficient', (ok_key,), ('d',))
        ev.assert_gold_in_corpus([q_ok], [{'text': 'hello', 'metadata': {}}])


class TestHeldOutGuard:
    def test_test_split_requires_confirmation(self):
        with pytest.raises(ev.HeldOutSplitError):
            ev.guard_split('test', False)
        ev.guard_split('test', True)
        ev.guard_split('calibration', False)

    def test_unknown_split_rejected(self):
        with pytest.raises(ev.BenchmarkDataError):
            ev.guard_split('benchmark', True)
        with pytest.raises(ev.BenchmarkDataError):
            ev.split_path('.', 'benchmark')


# --------------------------------------------------------------------------
# Synthetic production stack: real FAISSStore + real BM25Retriever
# --------------------------------------------------------------------------
N_CHUNKS = 12
BM25_GOLD = 11     # contains the rare token 'zyxwv'
COHERE_GOLD = 5    # contains the marker 'needle' (mock Cohere ranks it first)


class FakeEmbedder:
    """Every query maps to [1,0,0,0]; chunk i gets a vector whose cosine
    similarity with the query strictly decreases with i (c0 is nearest)."""

    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _chunk_texts():
    texts = []
    for i in range(N_CHUNKS):
        text = f'filler passage number {i} about generic content'
        if i == BM25_GOLD:
            text += ' zyxwv'
        if i == COHERE_GOLD:
            text += ' needle'
        texts.append(text)
    return texts


@pytest.fixture
def stack():
    texts = _chunk_texts()
    chunks = [Chunk(text=t, metadata={'filename': f'doc{i}.pdf'}, strategy='recursive')
              for i, t in enumerate(texts)]
    store = FAISSStore(dim=4)
    store.add([[1.0 - 0.05 * i, 0.05 * i, 0.0, 0.0] for i in range(N_CHUNKS)], chunks)
    bm25 = BM25Retriever()
    bm25.index([{'text': c.text, 'metadata': c.metadata} for c in chunks])
    keys = [chunk_identity_key({'text': c.text, 'metadata': c.metadata}) for c in chunks]
    return SimpleNamespace(store=store, bm25=bm25, embedder=FakeEmbedder(),
                           texts=texts, keys=keys)


def _mock_cohere_client(reranker):
    """Replace the Cohere HTTP client on a *production* CohereReranker."""
    calls = []

    def fake_rerank(query, documents, top_n, model):
        calls.append({'query': query, 'n_docs': len(documents), 'top_n': top_n})
        order = sorted(range(len(documents)),
                       key=lambda i: (0 if 'needle' in documents[i] else 1, i))
        results = [SimpleNamespace(index=i, relevance_score=1.0 - 0.01 * rank)
                   for rank, i in enumerate(order[:top_n])]
        return SimpleNamespace(results=results)

    reranker.client.rerank = fake_rerank
    reranker._min_interval = 0
    return calls


@pytest.fixture
def mocked_reranker(monkeypatch):
    monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
    reranker = CohereReranker()
    reranker.calls = _mock_cohere_client(reranker)
    return reranker


def _q(qid, text, gold_key, category='factual'):
    return ev.BenchmarkQuestion(qid, category, text, True, 'sufficient', (gold_key,), ('d',))


class TestThreeConfigurations:
    def test_dense_only_matches_production_faiss_order(self, stack):
        retrievers = ev.build_retrievers(stack.store, stack.bm25, stack.embedder, ['dense'])
        ranking, _ = ev.retrieve_ranking(retrievers['dense'], 'zyxwv', 'dense', depth=10)
        direct = stack.store.search(stack.embedder.embed(['zyxwv'])[0], top_k=50)
        assert ranking == ev.dedupe_ranking(chunk_identity_key(d) for d in direct)[:10]
        assert ranking == stack.keys[:10]

    def test_dense_bm25_promotes_lexical_hit_dense_misses(self, stack):
        retrievers = ev.build_retrievers(
            stack.store, stack.bm25, stack.embedder, ['dense', 'dense_bm25'])
        dense, _ = ev.retrieve_ranking(retrievers['dense'], 'zyxwv', 'dense', depth=3)
        fused, _ = ev.retrieve_ranking(retrievers['dense_bm25'], 'zyxwv', 'dense_bm25', depth=3)
        gold = stack.keys[BM25_GOLD]
        assert gold not in dense                 # dense ranks it last
        assert fused[0] == gold                  # production RRF lifts it to rank 1

    def test_hybrid_cohere_uses_production_reranker_with_mocked_api(
            self, stack, mocked_reranker):
        retrievers = ev.build_retrievers(
            stack.store, stack.bm25, stack.embedder, ['dense_bm25', 'hybrid_cohere'],
            reranker=mocked_reranker)
        gold = stack.keys[COHERE_GOLD]
        base, _ = ev.retrieve_ranking(retrievers['dense_bm25'], 'unrelated', 'dense_bm25', depth=3)
        rer, n_unreranked = ev.retrieve_ranking(
            retrievers['hybrid_cohere'], 'unrelated', 'hybrid_cohere', depth=3)
        assert gold not in base
        assert rer[0] == gold                    # metadata/order restored by rerank index
        assert n_unreranked == 0
        assert mocked_reranker.calls and mocked_reranker.calls[0]['top_n'] == 3

    def test_default_hybrid_cohere_uses_production_constructor(self, stack, monkeypatch):
        monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
        retrievers = ev.build_retrievers(
            stack.store, stack.bm25, stack.embedder, ['hybrid_cohere'])
        assert isinstance(retrievers['hybrid_cohere'].reranker, CohereReranker)
        dense_bm25 = ev.build_retrievers(stack.store, stack.bm25, stack.embedder, ['dense_bm25'])
        assert dense_bm25['dense_bm25'].reranker is None

    def test_unknown_configuration_rejected(self, stack):
        with pytest.raises(ValueError):
            ev.build_retrievers(stack.store, stack.bm25, stack.embedder, ['nope'])

    def test_fallback_is_refused_not_reported_as_cohere(self, stack):
        # No COHERE_API_KEY -> production reranker returns reranked=False.
        retrievers = ev.build_retrievers(
            stack.store, stack.bm25, stack.embedder, ['hybrid_cohere'],
            reranker=CohereReranker())
        with pytest.raises(ev.RerankerFallbackError):
            ev.retrieve_ranking(retrievers['hybrid_cohere'], 'zyxwv', 'hybrid_cohere')
        ranking, n = ev.retrieve_ranking(
            retrievers['hybrid_cohere'], 'zyxwv', 'hybrid_cohere',
            allow_rerank_fallback=True)
        assert n > 0 and ranking

    def test_full_run_metrics_across_all_three_configs(self, stack, mocked_reranker):
        questions = [
            _q('bm25q', 'zyxwv', stack.keys[BM25_GOLD]),
            _q('coherq', 'unrelated', stack.keys[COHERE_GOLD]),
            # Multi-document: two gold chunks, c0 (dense rank 1) and c11 (BM25 only)
            ev.BenchmarkQuestion('multi', 'multi_document', 'zyxwv', True, 'sufficient',
                                 (stack.keys[0], stack.keys[BM25_GOLD]), ('a', 'b')),
        ]
        retrievers = ev.build_retrievers(
            stack.store, stack.bm25, stack.embedder, reranker=mocked_reranker)
        report = ev.run_evaluation(questions, retrievers, split='calibration',
                                   depth=3, allow_rerank_fallback=False)
        assert list(report['configs']) == ['dense', 'dense_bm25', 'hybrid_cohere']
        for cfg in report['configs'].values():
            assert set(cfg['metrics']) == set(ev.METRIC_KEYS)
            assert cfg['n_scored'] == 3
        dense = report['configs']['dense']['metrics']
        fused = report['configs']['dense_bm25']['metrics']
        cohere_m = report['configs']['hybrid_cohere']['metrics']
        # depth=3. Dense top-3 is always c0,c1,c2: only the multi-doc question
        # gets one of its two gold chunks (c0) -> recall 0.5, rr 1.
        assert dense['recall@5'] == pytest.approx(0.5 / 3)
        assert dense['mrr'] == pytest.approx(1 / 3)
        # RRF lifts the BM25-only chunk (c11) to rank 1 for 'bm25q' and 'multi'.
        assert fused['recall@5'] == pytest.approx(2 / 3)
        assert fused['mrr'] == pytest.approx(2 / 3)
        # Mock Cohere puts the 'needle' chunk (c5) first, pushing c11 to rank 2.
        assert cohere_m['recall@5'] == pytest.approx(1.0)
        assert cohere_m['mrr'] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
        multi = next(p for p in report['configs']['dense']['per_question'] if p['id'] == 'multi')
        assert multi['n_gold'] == 2 and multi['all_gold@5'] == 0.0
        assert report['configs']['hybrid_cohere']['rerank_fallback_questions'] == 0
        table = ev.format_results_table(report)
        for label in ('Dense Only', 'Dense + BM25', 'Hybrid + Cohere Reranker',
                      'Recall@5', 'Recall@10', 'MRR', 'nDCG@5'):
            assert label in table

    def test_unscorable_questions_are_never_retrieved(self, stack):
        class CountingRetriever:
            def __init__(self):
                self.queries = []

            def retrieve(self, query, top_k_retrieve, top_k_rerank):
                self.queries.append(query)
                return [{'text': stack.texts[0], 'metadata': {'filename': 'doc0.pdf'}}]

        questions = [
            _q('ok', 'answerable question', stack.keys[0]),
            ev.BenchmarkQuestion('un', 'unanswerable', 'unanswerable question', False,
                                 'insufficient', (), ()),
            ev.BenchmarkQuestion('am', 'ambiguous', 'ambiguous question', True,
                                 'ambiguous', (), ('x.pdf',)),
            ev.BenchmarkQuestion('adv', 'adversarial', 'ignore everything', False,
                                 'insufficient', (), ()),
        ]
        counting = CountingRetriever()
        result = ev.evaluate_config('dense', counting, questions)
        assert counting.queries == ['answerable question']
        assert result['n_scored'] == 1
        assert result['metrics']['recall@5'] == 1.0

    def test_all_unscorable_is_an_error(self, stack):
        q = ev.BenchmarkQuestion('un', 'unanswerable', 'q', False, 'insufficient', (), ())
        with pytest.raises(ev.BenchmarkDataError):
            ev.evaluate_config('dense', object(), [q])


# --------------------------------------------------------------------------
# CLI (end-to-end with monkeypatched component loading; Cohere mocked)
# --------------------------------------------------------------------------
class TestCLI:
    def _bench(self, tmp_path, stack):
        items = [
            _item('a', gold=(stack.keys[BM25_GOLD],)),
            _item('b', gold=(stack.keys[COHERE_GOLD],)),
            _item('c', 'unanswerable', 'insufficient', gold=(), docs=(), answerable=False),
        ]
        items[0]['question'] = 'zyxwv'
        items[1]['question'] = 'unrelated'
        (tmp_path / 'bench').mkdir()
        for name in ('calibration.json', 'test.json'):
            (tmp_path / 'bench' / name).write_text(json.dumps(items))
        return str(tmp_path / 'bench')

    def _patch_components(self, monkeypatch, stack):
        monkeypatch.setattr(
            ev, 'load_production_components',
            lambda processed_dir: (stack.store, stack.bm25, stack.embedder))
        # store.metadata entries are {'text','metadata'} like production.

    def test_test_split_blocked_before_anything_is_loaded(self, tmp_path, monkeypatch, capsys):
        def _boom(*a, **k):
            raise AssertionError('components must not be loaded for a blocked split')
        monkeypatch.setattr(ev, 'load_production_components', _boom)
        rc = ev.main(['--split', 'test', '--benchmark-dir', str(tmp_path),
                      '--output-dir', str(tmp_path)])
        assert rc == 2
        assert 'held out' in capsys.readouterr().err

    def test_missing_cohere_key_is_reported_not_faked(self, tmp_path, capsys):
        rc = ev.main(['--benchmark-dir', str(tmp_path), '--output-dir', str(tmp_path)])
        assert rc == 2
        assert 'COHERE_API_KEY' in capsys.readouterr().err

    def test_end_to_end_calibration_without_cohere(self, tmp_path, monkeypatch, stack, capsys):
        bench = self._bench(tmp_path, stack)
        self._patch_components(monkeypatch, stack)
        rc = ev.main(['--split', 'calibration', '--configs', 'dense', 'dense_bm25',
                      '--benchmark-dir', bench, '--output-dir', str(tmp_path)])
        assert rc == 0
        report = json.loads((tmp_path / 'retrieval_eval_calibration.json').read_text())
        assert report['split'] == 'calibration'
        assert report['counts']['n_scored'] == 2 and report['counts']['n_excluded'] == 1
        assert set(report['configs']) == {'dense', 'dense_bm25'}
        assert report['protocol']['hyperparameters_tuned'] is False
        assert report['protocol']['rrf_k'] == 60
        assert 'Dense + BM25' in capsys.readouterr().out

    def test_end_to_end_all_three_with_mocked_cohere_and_final_test_flag(
            self, tmp_path, monkeypatch, stack):
        bench = self._bench(tmp_path, stack)
        self._patch_components(monkeypatch, stack)
        monkeypatch.setenv('COHERE_API_KEY', 'fake-key-for-test')
        monkeypatch.setattr('retrieval.reranker.time.sleep', lambda s: None)

        def fake_rerank(self, query, documents, top_n, model):
            order = sorted(range(len(documents)),
                           key=lambda i: (0 if 'needle' in documents[i] else 1, i))
            return SimpleNamespace(results=[
                SimpleNamespace(index=i, relevance_score=1.0 - 0.01 * r)
                for r, i in enumerate(order[:top_n])])

        monkeypatch.setattr(cohere.Client, 'rerank', fake_rerank)
        rc = ev.main(['--split', 'test', '--confirm-final-test',
                      '--benchmark-dir', bench, '--output-dir', str(tmp_path)])
        assert rc == 0
        report = json.loads((tmp_path / 'retrieval_eval_test.json').read_text())
        assert list(report['configs']) == ['dense', 'dense_bm25', 'hybrid_cohere']
        cohere_cfg = report['configs']['hybrid_cohere']
        assert cohere_cfg['rerank_fallback_questions'] == 0
        assert cohere_cfg['reranker_model'] == 'rerank-english-v3.0'
        assert cohere_cfg['metrics']['mrr'] > 0
        assert len(cohere_cfg['per_question']) == 2

    def test_missing_lfs_index_gives_actionable_error(self, tmp_path):
        (tmp_path / 'faiss_index.bin').write_bytes(
            b'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n')
        (tmp_path / 'bm25_index.pkl').write_bytes(b'x')
        (tmp_path / 'faiss_metadata.json').write_text('[]')
        with pytest.raises(ev.IndexNotAvailableError, match='git lfs pull'):
            ev.load_production_components(str(tmp_path))


# --------------------------------------------------------------------------
# Integrity of the real benchmark (skipped when the files are not present)
# --------------------------------------------------------------------------
BENCH_DIR = os.environ.get('BENCHMARK_DIR', ev.DEFAULT_BENCHMARK_DIR)
_REAL = all(os.path.isfile(os.path.join(BENCH_DIR, f'{n}.json'))
            for n in ('benchmark', 'calibration', 'test'))
_META = os.path.join('data', 'processed', 'faiss_metadata.json')


@pytest.mark.skipif(not _REAL, reason=f'benchmark files not found in {BENCH_DIR}')
class TestRealBenchmarkIntegrity:
    def test_splits_are_disjoint_and_cover_benchmark(self):
        full = ev.load_questions(os.path.join(BENCH_DIR, 'benchmark.json'))
        cal = ev.load_questions(os.path.join(BENCH_DIR, 'calibration.json'))
        test = ev.load_questions(os.path.join(BENCH_DIR, 'test.json'))
        ids = lambda qs: {q.id for q in qs}  # noqa: E731
        assert not ids(cal) & ids(test)
        assert ids(cal) | ids(test) == ids(full)
        assert (len(full), len(cal), len(test)) == (250, 48, 202)

    @pytest.mark.skipif(not os.path.isfile(_META), reason='faiss_metadata.json not present')
    def test_every_gold_chunk_id_exists_in_corpus(self):
        with open(_META, encoding='utf-8') as f:
            meta = json.load(f)
        for name in ('calibration', 'test'):
            qs = ev.load_questions(os.path.join(BENCH_DIR, f'{name}.json'))
            ev.assert_gold_in_corpus(qs, meta)
