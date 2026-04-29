import pandas as pd
import numpy as np

def test_time_period_encoding():
    TIME_ORDER = [
        "VERY_EARLY_MORNING", "EARLY_AM", "AM_PEAK", "MIDDAY_BASE",
        "MIDDAY_SCHOOL", "PM_PEAK", "EVENING", "LATE_EVENING", "NIGHT", "OFF_PEAK"
    ]
    time_map = {t: i for i, t in enumerate(TIME_ORDER)}
    assert time_map["AM_PEAK"] == 2
    assert time_map["NIGHT"] == 8
    assert len(time_map) == 10

def test_no_negative_load():
    df = pd.DataFrame({"average_load": [0, 1, 5, 10, -1]})
    df = df[df["average_load"] >= 0]
    assert len(df) == 4

def test_season_year_extraction():
    seasons = pd.Series(["Fall 2023", "Fall 2024", "Fall 2022"])
    years = seasons.str.extract(r"(\d{4})")[0].astype(int)
    assert list(years) == [2023, 2024, 2022]

def test_chronological_split():
    years = [2020, 2021, 2022, 2023, 2024]
    train = [y for y in years if y <= 2022]
    val   = [y for y in years if y == 2023]
    test  = [y for y in years if y == 2024]
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1