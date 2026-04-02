from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from skyfield.api import EarthSatellite, Topos, load

import lumos.calculatorLVLH as calculatorLVLH

_SPHERICAL_EARTH_RADIUS_M = 6371.0e3


@dataclass(frozen=True)
class TLERecord:
    name: str | None
    line1: str
    line2: str
    epoch_utc: datetime


@dataclass
class PreparedOrbitTrack:
    times_utc: list[datetime]
    skyfield_times: Any
    sat_alt_deg: np.ndarray
    sat_az_deg: np.ndarray
    sun_alt_deg: np.ndarray
    sun_az_deg: np.ndarray
    sat_height_m: np.ndarray
    dist_sat_obs_m: np.ndarray
    sun_vec_body: np.ndarray
    obs_vec_body: np.ndarray
    angle_past_terminator_rad: np.ndarray
    phase_angle_deg: np.ndarray
    eclipsed: np.ndarray
    r_sat_km: np.ndarray | None = None
    v_sat_km_s: np.ndarray | None = None
    source_kind: str = "unknown"
    source_index: np.ndarray | None = None
    source_name: list[str | None] | None = None
    source_epoch_utc: list[datetime | None] | None = None
    tle_records: list[TLERecord] | None = None
    tle_indices: np.ndarray | None = None
    active_tle_names: list[str | None] | None = None
    active_tle_epochs_utc: list[datetime | None] | None = None

    def to_dataframe(self, *, local_timezone: str | None = None):
        import pandas as pd

        data = {
            "UTC Time": pd.to_datetime(self.times_utc, utc=True),
            "Altitude (deg)": self.sat_alt_deg,
            "Azimuth (deg)": self.sat_az_deg,
            "Sun Altitude (deg)": self.sun_alt_deg,
            "Sun Azimuth (deg)": self.sun_az_deg,
            "Orbital Altitude (km)": self.sat_height_m / 1000.0,
            "Range to Observer (m)": self.dist_sat_obs_m,
            "Angle past terminator (rad)": self.angle_past_terminator_rad,
            "Phase angle (deg)": self.phase_angle_deg,
            "Eclipsed": self.eclipsed,
        }

        if self.tle_indices is not None:
            data["TLE index"] = self.tle_indices
        elif self.source_index is not None:
            data["Source index"] = self.source_index

        if self.active_tle_names is not None:
            data["TLE name"] = self.active_tle_names
        elif self.source_name is not None:
            data["Source name"] = self.source_name

        if self.active_tle_epochs_utc is not None:
            data["TLE epoch"] = pd.to_datetime(self.active_tle_epochs_utc, utc=True)
        elif self.source_epoch_utc is not None:
            data["Source epoch"] = pd.to_datetime(self.source_epoch_utc, utc=True)

        if self.source_kind != "unknown":
            data["Source kind"] = np.full(
                len(self.times_utc), self.source_kind, dtype=object
            )

        df = pd.DataFrame(data)
        if local_timezone is not None:
            df["Local Time"] = df["UTC Time"].dt.tz_convert(local_timezone)
        return df


PreparedTLETrack = PreparedOrbitTrack  # Alias for backward compatibility


@dataclass
class OrbitContext:
    track: PreparedTLETrack
    ts: Any
    eph: Any
    earth: Any
    sun: Any
    topos: Topos
    observer: Any
    sat_cache: dict[int, EarthSatellite] | None

    def to_dataframe(self, *, local_timezone: str | None = None):
        return self.track.to_dataframe(local_timezone=local_timezone)


@dataclass(frozen=True)
class _PreparedEnvironment:
    times_utc: list[datetime]
    ts: Any
    ts_times: Any
    eph: Any
    earth: Any
    sun: Any
    topos: Topos
    observer: Any


def _as_utc_datetime(dt: datetime) -> datetime:
    if not isinstance(dt, datetime):
        raise ValueError(f"Expected datetime, got {type(dt) !r}")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_step(step: timedelta | int | float) -> timedelta:
    if isinstance(step, timedelta):
        if step.total_seconds() <= 0:
            raise ValueError("Step must be positive")
        return step
    step = float(step)
    if step <= 0:
        raise ValueError("Step must be positive")
    return timedelta(minutes=step)


def tle_epoch_dt(line1: str) -> datetime:
    s = line1[18:32].strip()  # YYDDD.DDDDDDDD
    yy = int(s[0:2])
    ddd = float(s[2:])
    year = 1900 + yy if yy >= 57 else 2000 + yy
    day = int(np.floor(ddd))
    frac = ddd - day
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day - 1, seconds=frac * 86400.0
    )


def parse_tle_text(text: str, *, default_name: str = "SAT") -> list[TLERecord]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[TLERecord] = []

    i = 0
    while i < len(lines):
        if (
            i + 2 < len(lines)
            and not lines[i].startswith("1 ")
            and lines[i + 1].startswith("1 ")
            and lines[i + 2].startswith("2 ")
        ):
            name = lines[i]
            if name.startswith("0"):
                name = name[2:].strip()
            line1 = lines[i + 1]
            line2 = lines[i + 2]
            i += 3
        elif (
            i + 1 < len(lines)
            and lines[i].startswith("1 ")
            and lines[i + 1].startswith("2 ")
        ):
            name = default_name
            line1 = lines[i]
            line2 = lines[i + 1]
            i += 2
        else:
            i += 1
            continue

        records.append(
            TLERecord(
                name=name or default_name,
                line1=line1,
                line2=line2,
                epoch_utc=tle_epoch_dt(line1),
            )
        )
    if not records:
        raise ValueError("No valid TLE records found in text")
    records.sort(key=lambda r: r.epoch_utc)
    return records


def parse_tle_source(
    source: (
        str | Path | TLERecord | Sequence[str] | Iterable[TLERecord | Sequence[str]]
    ),
    *,
    default_name: str = "SAT",
) -> list[TLERecord]:
    if isinstance(source, TLERecord):
        return [source]

    if isinstance(source, Path):
        return parse_tle_text(
            source.expanduser().read_text(encoding="utf-8"), default_name=default_name
        )

    if isinstance(source, str):
        stripped = source.strip()
        if "\n" in stripped or stripped.startswith(("0 ", "1 ")):
            return parse_tle_text(stripped, default_name=default_name)

        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return parse_tle_text(
            path.read_text(encoding="utf-8"), default_name=default_name
        )

    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        if len(source) in (2, 3) and all(isinstance(item, str) for item in source):
            source = [source]  # type: ignore[assignment]

    records: list[TLERecord] = []
    for item in source:  # type: ignore[arg-type]
        if isinstance(item, TLERecord):
            records.append(item)
            continue

        item = list(item)
        if len(item) == 2:
            name = default_name
            line1, line2 = item
        elif len(item) == 3:
            name, line1, line2 = item
        else:
            raise ValueError(
                "Each TLE item must be (line1, line2) or (name, line1, line2)."
            )

        records.append(
            TLERecord(
                name=name,
                line1=line1,
                line2=line2,
                epoch_utc=tle_epoch_dt(line1),
            )
        )

    if not records:
        raise ValueError("No TLE records supplied.")

    records.sort(key=lambda rec: rec.epoch_utc)
    return records


def build_time_grid(
    *,
    start_utc: datetime,
    end_utc: datetime,
    step: timedelta | int | float = timedelta(minutes=1),
) -> list[datetime]:
    start_utc = _as_utc_datetime(start_utc)
    end_utc = _as_utc_datetime(end_utc)
    step = _coerce_step(step)

    if end_utc < start_utc:
        raise ValueError("end_utc must be after start_utc")

    times_utc: list[datetime] = []
    t = start_utc
    while t <= end_utc:
        times_utc.append(t)
        t += step
    return times_utc


def select_tle_indices(
    records: Sequence[TLERecord], times_utc: Sequence[datetime]
) -> np.ndarray:
    epochs = [_as_utc_datetime(record.epoch_utc) for record in records]
    indices = np.empty(len(times_utc), dtype=int)

    for i, t in enumerate(times_utc):
        indices[i] = bisect_right(epochs, _as_utc_datetime(t)) - 1

    return indices


def _prepare_environment(
    *,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    times_utc: Sequence[datetime] | None = None,
    step: timedelta | int | float = timedelta(minutes=1),
    observer: Topos | None = None,
    observer_latitude_deg: float | None = None,
    observer_longitude_deg: float | None = None,
    observer_elevation_m: float = 0.0,
    ephemeris_path: str | Path = "de421.bsp",
) -> _PreparedEnvironment:
    if times_utc is None:
        if start_utc is None or end_utc is None:
            raise ValueError("Provide either times_utc or both start_utc and end_utc.")
        times_utc = build_time_grid(start_utc=start_utc, end_utc=end_utc, step=step)
    else:
        times_utc = [_as_utc_datetime(t) for t in times_utc]

    if observer is None:
        if observer_latitude_deg is None or observer_longitude_deg is None:
            raise ValueError(
                "provide either observer=Topos(...) or observer_latitude_deg_observer_longitude_deg."
            )
        topos = Topos(
            latitude_degrees=observer_latitude_deg,
            longitude_degrees=observer_longitude_deg,
            elevation_m=observer_elevation_m,
        )
    else:
        topos = observer

    ts = load.timescale()
    ts_times = ts.from_datetimes(times_utc)

    eph = load(str(Path(ephemeris_path).expanduser()))
    earth = eph["earth"]
    sun = eph["sun"]
    observer_geo = earth + topos

    return _PreparedEnvironment(
        times_utc=list(times_utc),
        ts=ts,
        ts_times=ts_times,
        eph=eph,
        earth=earth,
        sun=sun,
        topos=topos,
        observer=observer_geo,
    )


def _observer_basis_from_position(r_obs_km):
    up = calculatorLVLH.unit(r_obs_km)
    spin_axis = np.array([0.0, 0.0, 1.0], float)

    east = np.cross(spin_axis, up)
    if np.linalg.norm(east) < 1e-12:
        east = np.cross(np.array([0.0, 1.0, 0.0], float), up)
    east = calculatorLVLH.unit(east)
    north = calculatorLVLH.unit(np.cross(up, east))
    return north, east, up


def _altaz_from_positions(r_obs_km, r_target_km):
    north, east, up = _observer_basis_from_position(r_obs_km)
    rel_hat = calculatorLVLH.unit(
        np.asarray(r_target_km, float) - np.asarray(r_obs_km, float)
    )

    alt_deg = float(np.degrees(np.arcsin(np.clip(np.dot(rel_hat, up), -1.0, 1.0))))
    az_deg = float(
        np.degrees(
            np.arctan2(
                np.dot(rel_hat, east),
                np.dot(rel_hat, north),
            )
        )
        % 360
    )
    return alt_deg, az_deg


def _prepare_tle_states(
    env: _PreparedEnvironment,
    records: Sequence[TLERecord],
    *,
    satellite_name: str = "SAT",
):
    tle_indices = select_tle_indices(records, env.times_utc)
    n = len(env.times_utc)

    r_sat_km = np.full((n, 3), np.nan, dtype=float)
    v_sat_km_s = np.full((n, 3), np.nan, dtype=float)
    active_tle_names: list[str | None] = [None] * n
    active_tle_epochs_utc: list[datetime | None] = [None] * n
    sat_cache: dict[int, EarthSatellite] = {}

    for i, (t_sf, idx) in enumerate(zip(env.ts_times, tle_indices)):
        if idx < 0:
            continue

        record = records[idx]
        active_tle_names[i] = record.name
        active_tle_epochs_utc[i] = record.epoch_utc

        satellite = sat_cache.get(idx)
        if satellite is None:
            satellite = EarthSatellite(
                record.line1,
                record.line2,
                record.name or satellite_name,
                env.ts,
            )
            sat_cache[idx] = satellite

        sat_state = satellite.at(t_sf)
        r_sat_km[i] = sat_state.position.km
        v_sat_km_s[i] = sat_state.velocity.km_per_s

    return {
        "tle_indices": tle_indices,
        "active_tle_names": active_tle_names,
        "active_tle_epochs_utc": active_tle_epochs_utc,
        "r_sat_km": r_sat_km,
        "v_sat_km_s": v_sat_km_s,
        "sat_cache": sat_cache,
    }


def _normalise_source_name(source_name, n):
    if source_name is None:
        return None
    if isinstance(source_name, str):
        return [source_name] * n

    source_name = list(source_name)
    if len(source_name) != n:
        raise ValueError("source_name sequence length must match the number of samples")
    return list(source_name)


def _coerce_state_array(name, values, n_expected=None):
    arr = np.asarray(values, float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if n_expected is not None and arr.shape[0] != n_expected:
        raise ValueError(f"{name} length must match number of samples")
    return arr


def _build_track_from_states(
    env: _PreparedEnvironment,
    *,
    r_sat_km,
    v_sat_km_s,
    source_kind: str,
    source_index=None,
    source_name=None,
    source_epoch_utc=None,
    tle_records=None,
    tle_indices=None,
    active_tle_names=None,
    active_tle_epochs_utc=None,
):
    n = len(env.times_utc)
    r_sat_km = _coerce_state_array("r_sat_km", r_sat_km, n_expected=n)
    v_sat_km_s = _coerce_state_array("v_sat_km_s", v_sat_km_s, n_expected=n)

    source_name = _normalise_source_name(source_name, n)
    if source_epoch_utc is not None:
        source_epoch_utc = list(source_epoch_utc)
        if len(source_epoch_utc) != n:
            raise ValueError("source_epoch_utc length must match number of samples")

    sat_alt_deg = np.full(n, np.nan, dtype=float)
    sat_az_deg = np.full(n, np.nan, dtype=float)
    sun_alt_deg = np.full(n, np.nan, dtype=float)
    sun_az_deg = np.full(n, np.nan, dtype=float)
    sat_height_m = np.full(n, np.nan, dtype=float)
    dist_sat_obs_m = np.full(n, np.nan, dtype=float)
    angle_past_terminator_rad = np.full(n, np.nan, dtype=float)
    phase_angle_deg = np.full(n, np.nan, dtype=float)
    eclipsed = np.zeros(n, dtype=bool)

    sun_vec_body = np.full((n, 3), np.nan, dtype=float)
    obs_vec_body = np.full((n, 3), np.nan, dtype=float)

    for i, t_sf in enumerate(env.ts_times):
        r_earth_km = env.earth.at(t_sf).position.km
        r_obs_km = env.observer.at(t_sf).position.km - r_earth_km
        r_sun_km = env.sun.at(t_sf).position.km - r_earth_km

        sun_alt_deg[i], sun_az_deg[i] = _altaz_from_positions(r_obs_km, r_sun_km)

        if not np.all(np.isfinite(r_sat_km[i])) or not np.all(
            np.isfinite(v_sat_km_s[i])
        ):
            continue

        sat_alt_deg[i], sat_az_deg[i] = _altaz_from_positions(r_obs_km, r_sat_km[i])

        sun_vec_i, obs_vec_i, ang_term_i, dist_i, sat_h_i = (
            calculatorLVLH.get_body_vectors_from_state(
                r_sat_km[i],
                v_sat_km_s[i],
                r_obs_km,
                r_sun_km,
            )
        )

        sun_vec_body[i] = sun_vec_i
        obs_vec_body[i] = obs_vec_i
        angle_past_terminator_rad[i] = float(ang_term_i)
        dist_sat_obs_m[i] = float(dist_i)
        sat_height_m[i] = float(sat_h_i)

        phase_angle_deg[i] = float(
            np.degrees(np.arccos(np.clip(np.dot(sun_vec_i, obs_vec_i), -1.0, 1.0)))
        )

        earth_ratio = _SPHERICAL_EARTH_RADIUS_M / (_SPHERICAL_EARTH_RADIUS_M + sat_h_i)
        horizon_angle = np.arcsin(np.clip(earth_ratio, -1.0, 1.0))
        angle_from_nadir = np.arccos(np.clip(sun_vec_i[2], -1.0, 1.0))
        eclipsed[i] = bool(angle_from_nadir < horizon_angle)

    return PreparedOrbitTrack(
        times_utc=list(env.times_utc),
        skyfield_times=env.ts_times,
        sat_alt_deg=sat_alt_deg,
        sat_az_deg=sat_az_deg,
        sun_alt_deg=sun_alt_deg,
        sun_az_deg=sun_az_deg,
        sat_height_m=sat_height_m,
        dist_sat_obs_m=dist_sat_obs_m,
        sun_vec_body=sun_vec_body,
        obs_vec_body=obs_vec_body,
        angle_past_terminator_rad=angle_past_terminator_rad,
        phase_angle_deg=phase_angle_deg,
        eclipsed=eclipsed,
        r_sat_km=r_sat_km,
        v_sat_km_s=v_sat_km_s,
        source_kind=source_kind,
        source_index=source_index,
        source_name=source_name,
        source_epoch_utc=source_epoch_utc,
        tle_records=tle_records,
        tle_indices=tle_indices,
        active_tle_names=active_tle_names,
        active_tle_epochs_utc=active_tle_epochs_utc,
    )


def _make_context(
    track: PreparedOrbitTrack,
    env: _PreparedEnvironment,
    *,
    sat_cache=None,
    set_calculator_earth=True,
):
    if set_calculator_earth:
        calculatorLVLH.set_earth(env.earth)

    return OrbitContext(
        track=track,
        ts=env.ts,
        eph=env.eph,
        earth=env.earth,
        sun=env.sun,
        topos=env.topos,
        observer=env.observer,
        sat_cache=sat_cache,
    )


def prepare_tle_track(
    tle_source: (
        str | Path | TLERecord | Sequence[str] | Iterable[TLERecord | Sequence[str]]
    ),
    *,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    times_utc: Sequence[datetime] | None = None,
    step: timedelta | int | float = timedelta(minutes=1),
    observer: Topos | None = None,
    observer_latitude_deg: float | None = None,
    observer_longitude_deg: float | None = None,
    observer_elevation_m: float = 0.0,
    ephemeris_path: str | Path = "de421.bsp",
    satellite_name: str = "SAT",
) -> PreparedOrbitTrack:
    records = parse_tle_source(tle_source, default_name=satellite_name)
    env = _prepare_environment(
        start_utc=start_utc,
        end_utc=end_utc,
        times_utc=times_utc,
        step=step,
        observer=observer,
        observer_latitude_deg=observer_latitude_deg,
        observer_longitude_deg=observer_longitude_deg,
        observer_elevation_m=observer_elevation_m,
        ephemeris_path=ephemeris_path,
    )
    tle_state = _prepare_tle_states(env, records, satellite_name=satellite_name)

    return _build_track_from_states(
        env,
        r_sat_km=tle_state["r_sat_km"],
        v_sat_km_s=tle_state["v_sat_km_s"],
        source_kind="tle",
        source_index=tle_state["tle_indices"],
        source_name=tle_state["active_tle_names"],
        source_epoch_utc=tle_state["active_tle_epochs_utc"],
        tle_records=records,
        tle_indices=tle_state["tle_indices"],
        active_tle_names=tle_state["active_tle_names"],
        active_tle_epochs_utc=tle_state["active_tle_epochs_utc"],
    )


def prepare_tle_context(
    tle_source,
    *,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    times_utc: Sequence[datetime] | None = None,
    step: timedelta | int | float = timedelta(minutes=1),
    observer: Topos | None = None,
    observer_latitude_deg: float | None = None,
    observer_longitude_deg: float | None = None,
    observer_elevation_m: float = 0.0,
    ephemeris_path: str | Path = "de421.bsp",
    satellite_name: str = "SAT",
    set_calculator_earth: bool = True,
) -> OrbitContext:
    records = parse_tle_source(tle_source, default_name=satellite_name)
    env = _prepare_environment(
        start_utc=start_utc,
        end_utc=end_utc,
        times_utc=times_utc,
        step=step,
        observer=observer,
        observer_latitude_deg=observer_latitude_deg,
        observer_longitude_deg=observer_longitude_deg,
        observer_elevation_m=observer_elevation_m,
        ephemeris_path=ephemeris_path,
    )
    tle_state = _prepare_tle_states(env, records, satellite_name=satellite_name)

    track = _build_track_from_states(
        env,
        r_sat_km=tle_state["r_sat_km"],
        v_sat_km_s=tle_state["v_sat_km_s"],
        source_kind="tle",
        source_index=tle_state["tle_indices"],
        source_name=tle_state["active_tle_names"],
        source_epoch_utc=tle_state["active_tle_epochs_utc"],
        tle_records=records,
        tle_indices=tle_state["tle_indices"],
        active_tle_names=tle_state["active_tle_names"],
        active_tle_epochs_utc=tle_state["active_tle_epochs_utc"],
    )

    return _make_context(
        track,
        env,
        sat_cache=tle_state["sat_cache"],
        set_calculator_earth=set_calculator_earth,
    )


def prepare_state_track(
    *,
    r_sat_km,
    v_sat_km_s,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    times_utc: Sequence[datetime] | None = None,
    step: timedelta | int | float = timedelta(minutes=1),
    observer: Topos | None = None,
    observer_latitude_deg: float | None = None,
    observer_longitude_deg: float | None = None,
    observer_elevation_m: float = 0.0,
    ephemeris_path: str | Path = "de421.bsp",
    source_name: str | Sequence[str | None] | None = None,
) -> PreparedOrbitTrack:
    env = _prepare_environment(
        start_utc=start_utc,
        end_utc=end_utc,
        times_utc=times_utc,
        step=step,
        observer=observer,
        observer_latitude_deg=observer_latitude_deg,
        observer_longitude_deg=observer_longitude_deg,
        observer_elevation_m=observer_elevation_m,
        ephemeris_path=ephemeris_path,
    )

    n = len(env.times_utc)
    r_sat_km = _coerce_state_array("r_sat_km", r_sat_km, n_expected=n)
    v_sat_km_s = _coerce_state_array("v_sat_km_s", v_sat_km_s, n_expected=n)

    return _build_track_from_states(
        env,
        r_sat_km=r_sat_km,
        v_sat_km_s=v_sat_km_s,
        source_kind="state",
        source_name=source_name,
    )


def prepare_state_context(
    *,
    r_sat_km,
    v_sat_km_s,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    times_utc: Sequence[datetime] | None = None,
    step: timedelta | int | float = timedelta(minutes=1),
    observer: Topos | None = None,
    observer_latitude_deg: float | None = None,
    observer_longitude_deg: float | None = None,
    observer_elevation_m: float = 0.0,
    ephemeris_path: str | Path = "de421.bsp",
    source_name: str | Sequence[str | None] | None = None,
    set_calculator_earth: bool = True,
) -> OrbitContext:
    env = _prepare_environment(
        start_utc=start_utc,
        end_utc=end_utc,
        times_utc=times_utc,
        step=step,
        observer=observer,
        observer_latitude_deg=observer_latitude_deg,
        observer_longitude_deg=observer_longitude_deg,
        observer_elevation_m=observer_elevation_m,
        ephemeris_path=ephemeris_path,
    )

    n = len(env.times_utc)
    r_sat_km = _coerce_state_array("r_sat_km", r_sat_km, n_expected=n)
    v_sat_km_s = _coerce_state_array("v_sat_km_s", v_sat_km_s, n_expected=n)

    track = _build_track_from_states(
        env,
        r_sat_km=r_sat_km,
        v_sat_km_s=v_sat_km_s,
        source_kind="state",
        source_name=source_name,
    )

    return _make_context(
        track, env, sat_cache=None, set_calculator_earth=set_calculator_earth
    )


__all__ = [
    "OrbitContext",
    "TLERecord",
    "PreparedOrbitTrack",
    "PreparedTLETrack",
    "tle_epoch_dt",
    "parse_tle_text",
    "parse_tle_source",
    "build_time_grid",
    "select_tle_indices",
    "prepare_tle_track",
    "prepare_tle_context",
    "prepare_state_track",
    "prepare_state_context",
]
