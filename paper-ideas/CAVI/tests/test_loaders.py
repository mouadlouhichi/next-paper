"""Tests for the dataset loaders (ML-1M, Amazon-Book, Yelp2018)."""
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cavi.data import (load_lightgcn_split, lightgcn_stats, load_remap_lists,
                       build_user_seq_from_split, future_from_test,
                       item_popularity, movability_from_popularity)


def write_lightgcn(tmp, prefix, train_lines, test_lines, user_list=None,
                   item_list=None):
    d = tmp
    with open(os.path.join(d, "train.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(d, "test.txt"), "w") as f:
        f.write("\n".join(test_lines) + "\n")
    if user_list is not None:
        with open(os.path.join(d, "user_list.txt"), "w") as f:
            f.write("org_id remap_id\n" + "\n".join(user_list) + "\n")
    if item_list is not None:
        with open(os.path.join(d, "item_list.txt"), "w") as f:
            f.write("org_id remap_id\n" + "\n".join(item_list) + "\n")
    return d


def test_lightgcn_split_parse():
    with tempfile.TemporaryDirectory() as tmp:
        d = write_lightgcn(tmp, "x",
                           ["0 1 2 3", "1 4 5"],   # train
                           ["0 9", "1 8"])          # test
        tr, te = load_lightgcn_split(d, "x")
        assert tr[0] == [1, 2, 3]
        assert tr[1] == [4, 5]
        assert te[0] == [9]
        assert te[1] == [8]


def test_lightgcn_stats():
    with tempfile.TemporaryDirectory() as tmp:
        d = write_lightgcn(tmp, "x",
                           ["0 1 2 3", "1 4 5"],
                           ["0 9", "1 8"])
        tr, te = load_lightgcn_split(d, "x")
        st = lightgcn_stats(tr, te)
        assert st["users"] == 2
        assert st["items"] == 10   # max item id + 1 = 9 + 1
        assert st["interactions"] == 7
        assert 0 < st["density"] < 1


def test_remap_lists():
    with tempfile.TemporaryDirectory() as tmp:
        d = write_lightgcn(tmp, "x", ["0 1"], ["0 2"],
                           user_list=["u1 0", "u2 1"],
                           item_list=["orgA 0", "orgB 1"])
        ul, il = load_remap_lists(d, "x")
        assert ul[0] == "u1"
        assert il[1] == "orgB"


def test_build_seq_and_future():
    with tempfile.TemporaryDirectory() as tmp:
        d = write_lightgcn(tmp, "x", ["0 1 2 3 4"], ["0 9 10"])
        tr, te = load_lightgcn_split(d, "x")
        seqs = build_user_seq_from_split(tr, te)
        assert [i for i, _ in seqs[0]] == [1, 2, 3, 4]
        assert future_from_test(te, 0) == [9, 10]


def test_movability_from_popularity():
    # item 1 appears 10x (most popular), item 2 appears 1x, item 3 appears 5x
    tr = {0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3]}
    pop = item_popularity(tr)
    assert pop[1] == 10 and pop[2] == 1 and pop[3] == 5
    # history window [1,2,3]; threshold 0.5*10=5 -> items 1 and 3 are anchors
    mov = movability_from_popularity(pop, [1, 2, 3], threshold_ratio=0.5)
    # mov[i] True = anchor/immovable
    assert mov == [True, False, True]
    assert item_popularity(tr)[1] == 10


def test_load_items_handles_latin1_titles():
    """MovieLens movies.dat contains Latin-1 accented chars (0xe9='e'acute);
    the loader must not crash with UnicodeDecodeError."""
    import tempfile
    from cavi.data import load_items
    raw = b"1\tToy Story (1995)\tAnimation|Children's|Comedy\n" \
          b"500\tJos\xe9 & Co (2000)\tComedy\n"   # 0xe9 = e-acute (latin-1)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        items = load_items(path)
        assert items[1] == "Animation|Children's|Comedy"
        assert "Comedy" in items[500]
    finally:
        os.remove(path)


def test_load_ratings_handles_latin1():
    import tempfile
    from cavi.data import load_ratings
    raw = b"1\t1193\t5\t978300760\n" \
          b"2\t500\t4\t978300761\n"   # ASCII numeric, but should still parse
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        rows = load_ratings(path)
        assert rows == [(1, 1193, 5.0, 978300760), (2, 500, 4.0, 978300761)]
    finally:
        os.remove(path)


def test_load_ratings_colon_separator():
    """MovieLens official files use '::' separator; must parse."""
    import tempfile
    from cavi.data import load_ratings
    raw = b"1::1193::5::978300760\n1::661::3::978302109\n"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(raw); path = f.name
    try:
        rows = load_ratings(path)
        assert rows == [(1, 1193, 5.0, 978300760), (1, 661, 3.0, 978302109)]
    finally:
        os.remove(path)


def test_load_items_colon_separator():
    """MovieLens items official format: id::title::genres."""
    import tempfile
    from cavi.data import load_items
    raw = b"1::Toy Story (1995)::Animation|Children's|Comedy\n2::Jumanji (1995)::Adventure\n"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(raw); path = f.name
    try:
        items = load_items(path)
        assert items[1] == "Animation|Children's|Comedy"
        assert items[2] == "Adventure"
    finally:
        os.remove(path)


def test_load_items_multiple_delimiters():
    """Loader is delimiter-agnostic (colon, tab, comma, semicolon)."""
    import tempfile
    from cavi.data import load_items
    for sep in ["::", "\t", ",", ";"]:
        raw = ("1" + sep + "Toy Story (1995)" + sep + "Animation|Comedy\n"
               "2" + sep + "Jumanji (1995)" + sep + "Adventure\n").encode()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(raw); path = f.name
        try:
            d = load_items(path)
            assert len(d) == 2, f"sep {sep!r} -> {len(d)}"
            assert d[1] == "Animation|Comedy"
        finally:
            os.remove(path)
