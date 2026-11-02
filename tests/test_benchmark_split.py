"""Tests for scripts/split_benchmark.py's stratified_split(). No API keys
/ network needed. Loads the script as a module via importlib since
`scripts/` is not a package."""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "split_benchmark",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "split_benchmark.py"),
)
split_benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(split_benchmark)
stratified_split = split_benchmark.stratified_split


def make_items(category, n, prefix):
    return [
        {"id": f"{prefix}_{i:03d}", "category": category, "question": f"q{i}"}
        for i in range(n)
    ]


def sample_benchmark():
    items = []
    items += make_items("factual", 40, "fact")
    items += make_items("numerical", 30, "num")
    items += make_items("multi_document", 26, "multi")
    items += make_items("comparison", 26, "cmp")
    items += make_items("conversational", 26, "conv")
    items += make_items("ambiguous", 20, "amb")
    items += make_items("unanswerable", 20, "una")
    items += make_items("adversarial", 12, "adv")
    return items


def test_no_overlap_between_calibration_and_test():
    items = sample_benchmark()
    calibration, test = stratified_split(items, calibration_fraction=0.2, seed=42)
    calibration_ids = {i["id"] for i in calibration}
    test_ids = {i["id"] for i in test}
    assert calibration_ids.isdisjoint(test_ids)


def test_split_covers_every_item_exactly_once():
    items = sample_benchmark()
    calibration, test = stratified_split(items, calibration_fraction=0.2, seed=42)
    all_ids = {i["id"] for i in items}
    split_ids = {i["id"] for i in calibration} | {i["id"] for i in test}
    assert split_ids == all_ids
    assert len(calibration) + len(test) == len(items)


def test_split_is_deterministic_for_same_seed():
    items = sample_benchmark()
    cal1, test1 = stratified_split(items, calibration_fraction=0.2, seed=42)
    cal2, test2 = stratified_split(items, calibration_fraction=0.2, seed=42)
    assert [i["id"] for i in cal1] == [i["id"] for i in cal2]
    assert [i["id"] for i in test1] == [i["id"] for i in test2]


def test_split_differs_for_different_seed():
    items = sample_benchmark()
    cal1, _ = stratified_split(items, calibration_fraction=0.2, seed=42)
    cal2, _ = stratified_split(items, calibration_fraction=0.2, seed=7)
    assert [i["id"] for i in cal1] != [i["id"] for i in cal2]


def test_every_category_represented_in_both_splits():
    items = sample_benchmark()
    calibration, test = stratified_split(items, calibration_fraction=0.2, seed=42)
    categories = {i["category"] for i in items}
    cal_categories = {i["category"] for i in calibration}
    test_categories = {i["category"] for i in test}
    assert cal_categories == categories
    assert test_categories == categories


def test_calibration_fraction_is_roughly_20_percent():
    items = sample_benchmark()
    calibration, test = stratified_split(items, calibration_fraction=0.2, seed=42)
    fraction = len(calibration) / len(items)
    assert 0.15 <= fraction <= 0.25
