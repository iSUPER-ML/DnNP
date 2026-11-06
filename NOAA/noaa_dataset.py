"""
NOAA temperature dataset wrapper for DNP, modeled after NPAirDataset (ISUPER).

Each "task" is a single date: stations on that date act as context+target points,
where x = (lat, lon, temp, rhum) and y = pres.

Per-dimension noise:
  - lat, lon : clean (σ=0)
  - temp     : noisy ASOS thermometer (σ=0.5°C → normalized)
  - rhum     : noisy capacitive humidity (σ=2% RH → normalized)

Stations missing on some days are padded with NaN in feature columns;
the dataset returns a 'valid' mask so downstream code can ignore them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Feature/target columns
FEATURE_COLS = ["latitude", "longitude", "temp", "rhum"]
TARGET_COL   = "pres"


class NoaaTemperatureDataset(Dataset):
    """
    Each item returns one date's worth of stations:
        x:     (n_stations, 4) — lat, lon, temp, rhum (min-max normalized to [0,1])
        y:     (n_stations, 1) — pres (min-max normalized to [0,1])
        valid: (n_stations,)   — True if (x, y) for this station/date is observed
    """

    def __init__(self, data_dir, split="train", norm_stats=None):
        self.data_dir = Path(data_dir)
        self.split    = split

        df = pd.read_csv(self.data_dir / "filtered.csv", parse_dates=["date"])
        with open(self.data_dir / "temporal_splits.json") as f:
            tsplits = json.load(f)

        wanted_dates = pd.to_datetime(tsplits[f"{split}_dates"])
        df = df[df["date"].isin(wanted_dates)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows for split={split} in {data_dir}")

        # ── Drop dates where ANY station is missing features or target ──────
        # Each station must have observed (lat, lon, temp, rhum, pres) on
        # every kept date, so __getitem__ can return consistent shapes.
        total_stations = df["station"].nunique()
        df_obs = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
        date_counts = df_obs.groupby("date")["station"].nunique()
        complete_dates = sorted(date_counts[date_counts == total_stations].index)

        n_dropped = len(wanted_dates) - len(complete_dates)
        print(f"  [{split}] kept {len(complete_dates)}/{len(wanted_dates)} dates "
              f"({n_dropped} dropped due to missing values), "
              f"{total_stations} stations")

        if len(complete_dates) == 0:
            raise ValueError(
                f"No dates with all {total_stations} stations present "
                f"in split={split}. Try fewer stations or relax filter."
            )

        df = df[df["date"].isin(complete_dates)].reset_index(drop=True)

        self.df = df
        self.dates = complete_dates
        self.station_ids = sorted(df["station"].unique())
        self.station_to_idx = {sid: i for i, sid in enumerate(self.station_ids)}

        # ── Normalization stats (computed from training, reused for val/test) ─
        if norm_stats is None:
            x_all = df[FEATURE_COLS].values
            y_all = df[[TARGET_COL]].values
            self.norm_stats = {
                "x_min": x_all.min(axis=0),
                "x_max": x_all.max(axis=0),
                "y_min": y_all.min(axis=0),
                "y_max": y_all.max(axis=0),
            }
        else:
            self.norm_stats = norm_stats

    def __len__(self):
        return len(self.dates)

    def _normalize_x(self, x):
        rng = self.norm_stats["x_max"] - self.norm_stats["x_min"]
        rng = np.where(rng == 0, 1.0, rng)
        return (x - self.norm_stats["x_min"]) / rng

    def _normalize_y(self, y):
        rng = self.norm_stats["y_max"] - self.norm_stats["y_min"]
        rng = np.where(rng == 0, 1.0, rng)
        return (y - self.norm_stats["y_min"]) / rng

    def __getitem__(self, idx):
        date = self.dates[idx]
        sub = self.df[self.df["date"] == date].sort_values("station")
        x = sub[FEATURE_COLS].values.astype(np.float32)
        y = sub[[TARGET_COL]].values.astype(np.float32)
        x = self._normalize_x(x).astype(np.float32)
        y = self._normalize_y(y).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def n_stations_per_day(self):
        """Average number of OBSERVED (non-NaN) stations per date."""
        return self.df.dropna(subset=FEATURE_COLS + [TARGET_COL]) \
                      .groupby("date").size().mean()

    def feature_sigma_normalized(self, feature_idx, sigma_raw):
        """Convert raw noise σ → normalized scale for noise_std_per_dim."""
        rng = self.norm_stats["x_max"][feature_idx] - self.norm_stats["x_min"][feature_idx]
        return float(sigma_raw / rng) if rng > 0 else 0.0