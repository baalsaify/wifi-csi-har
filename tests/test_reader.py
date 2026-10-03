import numpy as np
import pytest
from conftest import make_csv_text

from csi_har.data.reader import parse_csv_text, parse_trial_name


def test_parse_trial_name():
    info = parse_trial_name("E1_S04_C03_A07_T17.csv")
    assert (info.environment, info.subject, info.experiment, info.activity, info.trial) == (1, 4, 3, 7, 17)


def test_parse_trial_name_rejects_other_files():
    with pytest.raises(ValueError):
        parse_trial_name("readme.txt")


def test_parse_csv_text_round_trip(csv_text):
    text, expected = csv_text
    meta, csi = parse_csv_text(text)
    assert meta.shape == (64, 13)
    assert csi.shape == (64, 90)
    assert csi.dtype == np.complex64
    np.testing.assert_array_equal(csi, expected.astype(np.complex64))
    assert meta[0, 2] == 3 and meta[0, 3] == 1  # Nrx, Ntx


def test_negative_imaginary_parts_are_parsed():
    text, _ = csv_text_with_value("-20+-5i")
    _, csi = parse_csv_text(text)
    assert csi[0, 0] == complex(-20, -5)


def test_rejects_wrong_header():
    with pytest.raises(ValueError, match="header"):
        parse_csv_text("a,b,c\n1,2,3\n")


def test_rejects_truncated_row(csv_text):
    text, _ = csv_text
    truncated = text.rstrip("\n").rsplit(",", 1)[0] + "\n"
    with pytest.raises(ValueError, match="multiple"):
        parse_csv_text(truncated)


def csv_text_with_value(value: str):
    text, csi = make_csv_text(n_packets=2)
    header, first, second = text.strip().split("\n")
    cells = first.split(",")
    cells[13] = value
    return "\n".join([header, ",".join(cells), second]) + "\n", csi
