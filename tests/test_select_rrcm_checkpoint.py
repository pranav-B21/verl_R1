import json

from scripts.select_rrcm_checkpoint import collect


def write_metric(path, hr5, ndcg5):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([{"HR": [hr5], "NDCG": [ndcg5]}]), encoding="utf-8")


def test_collect_uses_only_matching_validation_decodes_and_step_range(tmp_path):
    write_metric(tmp_path / "global_step_500" / "valgreedy1" / "test_metrics_top5.json", 0.01, 0.009)
    write_metric(tmp_path / "global_step_500" / "valgreedy2" / "test_metrics_top5.json", 0.02, 0.011)
    write_metric(tmp_path / "global_step_500" / "greedy1" / "test_metrics_top5.json", 0.99, 0.99)
    write_metric(tmp_path / "global_step_450" / "valgreedy1" / "test_metrics_top5.json", 0.50, 0.50)

    rows = collect(tmp_path, "valgreedy", min_step=500, max_step=1300)

    assert len(rows) == 1
    assert rows[0]["step"] == 500
    assert rows[0]["n_decodes"] == 2
    assert rows[0]["hr5_mean"] == 0.015
    assert rows[0]["ndcg5_mean"] == 0.01
