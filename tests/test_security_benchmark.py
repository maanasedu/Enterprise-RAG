"""Dimension 3: the adversarial benchmark file, and proof that the things this
dimension promised not to touch are untouched."""
import hashlib
import json
import os

from evaluation import security_eval as se
from scripts import build_security_benchmark as builder

BENCH = os.path.join('data', 'security', 'adversarial_benchmark.json')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_benchmark_size_and_category_coverage():
    doc = se.load_cases(BENCH)
    attacks = [c for c in doc['cases'] if c['kind'] == 'attack']
    assert 50 <= len(attacks) <= 100
    for cat in builder.CATEGORIES:
        assert sum(c['category'] == cat for c in attacks) >= 10, cat
    assert doc['counts']['attacks'] == len(attacks)
    assert {'hijack', 'exfil', 'leak_prompt', 'leak_secret'} <= {c['goal'] for c in attacks}
    # every evasion family the docs discuss is actually represented
    assert {'plain', 'obfuscated', 'encoded', 'multilingual', 'paraphrase', 'delimiter_spoof',
            'roleplay', 'homoglyph', 'split', 'hybrid_topical'} <= {c['technique'] for c in attacks}


def test_indirect_and_embedded_cases_have_benign_questions_and_an_injection():
    from security.guardrails import detect
    for c in se.load_cases(BENCH)['cases']:
        if c['category'] in ('indirect_prompt_injection', 'embedded_document_injection'):
            assert c['injection'] and c['utility'] and c['expected_keywords']
            assert not detect(c['question']).flagged(2), c['id']          # the attack is only in the document
        if c['category'] in ('direct_prompt_injection', 'system_override', 'prompt_secret_extraction'):
            assert c['injection'] is None


def test_markers_are_unique_and_absent_from_the_corpus_text_used_by_cases():
    cases = se.load_cases(BENCH)['cases']
    markers = [c['marker'] for c in cases if c['marker']]
    assert len(markers) == len(set(markers))
    clean = ' '.join(c['clean_context'] for c in cases).lower()
    assert not any(m.lower() in clean for m in markers)


def test_benign_controls_include_security_flavoured_questions():
    controls = [c for c in se.load_cases(BENCH)['cases'] if c['kind'] == 'benign']
    assert len(controls) >= 15
    assert {'plain', 'security_topic', 'instruction_like_document', 'quoted_attack_in_document'} <= \
           {c['technique'] for c in controls}


def test_committed_benchmark_matches_the_generator():
    assert builder.main(['--check']) == 0


def test_builder_is_deterministic():
    assert builder.render(builder.build_document()) == builder.render(builder.build_document())


def _hashes(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            out[name] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    return out


def test_benchmark_lives_outside_data_evaluation_and_running_everything_leaves_that_folder_untouched(tmp_path):
    from scripts import evaluate_security as cli
    eval_dir = os.path.join('data', 'evaluation')
    assert os.path.normpath(os.path.dirname(BENCH)) != os.path.normpath(eval_dir)
    before = _hashes(eval_dir)
    assert builder.main(['--check']) == 0
    assert cli.main(['--mode', 'offline', '--ablation', '--output', str(tmp_path / 'r.json')]) == 0
    assert _hashes(eval_dir) == before                      # Dimension 1 / 2 data byte-for-byte unchanged


def test_calibrated_evidence_threshold_is_unchanged():
    from retrieval.adaptive import AdaptiveConfig
    from retrieval.evidence import DEFAULT_MIN_TOP_SCORE
    assert DEFAULT_MIN_TOP_SCORE == 0.019681409
    assert AdaptiveConfig.from_env({}).evidence.min_top_score == 0.019681409
