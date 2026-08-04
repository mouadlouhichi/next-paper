from __future__ import annotations

from cure_rec.data import audit_interactions, load_coat, load_movielens_1m, load_yahoo_r3


def test_load_movielens_1m_local_fixture(tmp_path):
    folder = tmp_path / "ml-1m"
    folder.mkdir()
    (folder / "ratings.dat").write_text("1::10::5::978300760\n2::11::2::978302109\n", encoding="latin-1")
    result = load_movielens_1m(tmp_path)
    assert result.dataset == "movielens_1m"
    assert len(result.interactions) == 2
    assert result.interactions["response"].tolist() == [1, 0]
    assert audit_interactions(result.interactions).permitted_claim == "descriptive_or_semisynthetic"


def test_load_coat_preserves_missing_timestamps(tmp_path):
    (tmp_path / "train.ascii").write_text("5 0\n0 3\n")
    (tmp_path / "test.ascii").write_text("0 4\n2 0\n")
    result = load_coat(tmp_path)
    assert result.dataset == "coat"
    assert len(result.interactions) == 4
    audit = audit_interactions(result.interactions)
    assert "timestamp" in audit.missing_required
    assert audit.permitted_claim == "descriptive_only"


def test_load_yahoo_r3_local_fixture(tmp_path):
    (tmp_path / "ydata-ymusic-rating-study-v1_0-train.txt").write_text("1\t10\t5\n")
    (tmp_path / "ydata-ymusic-rating-study-v1_0-test.txt").write_text("2\t11\t3\n")
    result = load_yahoo_r3(tmp_path)
    assert result.dataset == "yahoo_r3"
    assert len(result.interactions) == 2
    assert result.interactions["split"].tolist() == ["train", "test"]
