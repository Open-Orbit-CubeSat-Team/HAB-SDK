from model_wind import (
    LaunchSite,
    Balloon,
    Payload,
    MissionProfile,
    Model,
    ETOPO1Terrain,
    GroundNode,
    compute_node_observations_batch,
)
from gfs_wind import GFSWind
from hrrr_wind import HRRRWind
import numpy as np
import pandas as pd
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import matplotlib.pyplot as plt
import contextily as cx
from datetime import datetime, timedelta, timezone
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

pd.set_option('display.max_rows', None)

#cloc . --exclude-dir=__pycache__,.vscode,gfs_downloads,hrrr_downloads,mission_design_spaces,terrain_cache,.git

# Worker-local caches so each process only builds terrain / wind once.
_WORKER_TERRAIN = None
_WORKER_WIND = None
_WORKER_SIG = None

from dataclasses import dataclass

@dataclass(frozen=True)
class WindSpec:
    family: str          # "gfs" or "hrrr"
    product: str         # Herbie product string
    cycle_hours: int     # model cycle spacing
    preload_hours: int
    max_hours_total: int

WIND_SPECS = {
    "gfs0p25": WindSpec(
        family="gfs",
        product="pgrb2.0p25",
        cycle_hours=6,
        preload_hours=3,
        max_hours_total=24,
    ),
    "gfs0p50": WindSpec(
        family="gfs",
        product="pgrb2.0p50",
        cycle_hours=6,
        preload_hours=3,
        max_hours_total=24,
    ),
    "gfs1p00": WindSpec(
        family="gfs",
        product="pgrb2.1p00",
        cycle_hours=6,
        preload_hours=3,
        max_hours_total=24,
    ),
    "hrrr": WindSpec(
        family="hrrr",
        product="prs",
        cycle_hours=1,
        preload_hours=3,
        max_hours_total=18,
    ),
}

def get_wind_spec(wind_kind: str) -> WindSpec:
    wk = wind_kind.strip().lower()
    try:
        return WIND_SPECS[wk]
    except KeyError as e:
        raise ValueError(f"Unsupported wind_kind: {wind_kind}") from e

def extract_attributes(obj, prefix=""):
    attributes = {}
    for attr in dir(obj):
        if not callable(getattr(obj, attr)) and not attr.startswith("__"):
            value = getattr(obj, attr)
            if hasattr(value, "__dict__"):
                nested_attrs = extract_attributes(value, prefix=f"{prefix}{attr}.")
                attributes.update(nested_attrs)
            else:
                attributes[f"{prefix}{attr}"] = value
    return attributes

def profiles_to_dataframe(profiles):
    profile_data = []
    for i, profile in enumerate(profiles):
        profile_dict = {"Profile ID": i}
        attributes = extract_attributes(profile)
        profile_dict.update(attributes)
        profile_data.append(profile_dict)
    profile_df = pd.DataFrame(profile_data)
    profile_df.set_index("Profile ID", inplace=True)
    return profile_df

def mp_progress(done, total, start, bar_len=60):
    frac = done / total if total else 1.0
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = time.perf_counter() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    sys.stdout.write(
        f"\rProcessing {total} profiles. |{bar}| {done:>5}/{total}  ({100*frac:>3.0f}%)  "
        f"{rate:>5.2f} profiles/s  ETA {eta:>6.1f}s   "
    )
    sys.stdout.flush()

def parse_utc_time(t):
    if isinstance(t, str):
        return datetime.strptime(t, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return t

def format_utc_time(t):
    return t.strftime("%Y-%m-%d %H:%M")

def nominal_cycle_time_utc(wind_kind: str, launch_time_utc):
    t = parse_utc_time(launch_time_utc)
    spec = get_wind_spec(wind_kind)

    cycle_hour = (t.hour // spec.cycle_hours) * spec.cycle_hours
    return t.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

def resolve_wind_cycle_time_utc(wind_kind: str, launch_time_utc, wind_verbose: bool = True):
    last_error = None

    for cycle_time_utc in cycle_candidates_utc(wind_kind, launch_time_utc):
        try:
            wind = build_wind(wind_kind, cycle_time_utc, wind_verbose=wind_verbose)
            return cycle_time_utc, wind
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Could not build {wind_kind} wind for launch_time_utc={launch_time_utc}"
    ) from last_error

def cycle_candidates_utc(wind_kind: str, launch_time_utc, max_fallbacks=4):
    spec = get_wind_spec(wind_kind)
    nominal = nominal_cycle_time_utc(wind_kind, launch_time_utc)
    step = timedelta(hours=spec.cycle_hours)

    for k in range(max_fallbacks + 1):
        yield format_utc_time(nominal - k * step)

def build_wind(wind_kind: str, run_time_utc: str, wind_verbose: bool = True):
    spec = get_wind_spec(wind_kind)
    wk = wind_kind.strip().lower()

    if spec.family == "gfs":
        return GFSWind(
            run_utc=run_time_utc,
            save_dir="./gfs_downloads",
            product=spec.product,
            preload_hours=spec.preload_hours,
            max_hours_total=spec.max_hours_total,
            sample_time_bin_s=60.0,
            sample_alt_bin_m=100.0,
            sample_latlon_decimals=6,
            verbose=wind_verbose,
        )

    if spec.family == "hrrr":
        gfs_fallback_run = format_utc_time(
            nominal_cycle_time_utc("gfs0p25", parse_utc_time(run_time_utc))
        )

        gfs_fallback_spec = get_wind_spec("gfs0p25")
        gfs_fallback = GFSWind(
            run_utc=gfs_fallback_run,
            save_dir="./gfs_downloads",
            product=gfs_fallback_spec.product,
            preload_hours=gfs_fallback_spec.preload_hours,
            max_hours_total=gfs_fallback_spec.max_hours_total,
            sample_time_bin_s=60.0,
            sample_alt_bin_m=100.0,
            sample_latlon_decimals=6,
            verbose=wind_verbose,
        )

        return HRRRWind(
            run_utc=run_time_utc,
            save_dir="./hrrr_downloads",
            product=spec.product,
            fallback_wind=gfs_fallback,
            preload_hours=spec.preload_hours,
            max_hours_total=spec.max_hours_total,
            sample_time_bin_s=60.0,
            sample_alt_bin_m=100.0,
            sample_latlon_decimals=6,
            verbose=wind_verbose,
        )

    raise ValueError(f"Unsupported wind_kind: {wind_kind}")

def get_worker_resources(wind_kind: str, launch_time_utc, wind_verbose: bool = True):
    global _WORKER_TERRAIN, _WORKER_WIND, _WORKER_SIG

    nominal_cycle = format_utc_time(nominal_cycle_time_utc(wind_kind, launch_time_utc))

    if _WORKER_TERRAIN is None:
        _WORKER_TERRAIN = ETOPO1Terrain()

    if (
        _WORKER_WIND is None
        or _WORKER_SIG is None
        or _WORKER_SIG[0] != wind_kind.strip().lower()
        or _WORKER_SIG[1] != nominal_cycle
    ):
        resolved_cycle, wind = resolve_wind_cycle_time_utc(wind_kind, launch_time_utc,wind_verbose=wind_verbose)
        _WORKER_WIND = wind
        _WORKER_SIG = (wind_kind.strip().lower(), nominal_cycle, resolved_cycle)

    return _WORKER_TERRAIN, _WORKER_WIND

def predictor_batch(batch, dt, wind_kind, logging=False, wind_verbose=True):
    pairs = []
    for idx, profile in batch:
        terrain, wind = get_worker_resources(wind_kind, profile.launch_time_utc,wind_verbose=wind_verbose)

        out = []
        try:
            Model(
                time_step=dt,
                profiles=[profile],
                result=out,
                wind=wind,
                terrain=terrain,
                run_time_utc=profile.launch_time_utc,
                ascent_cd_scale=CD_SCALE,
            ).altitude_model(logging=logging)
            prof = out[0] if out else None
        except Exception:
            prof = None

        pairs.append((idx, prof))

    return pairs

def profile_batch_key(profile, wind_kind: str):
    nominal_cycle = format_utc_time(
        nominal_cycle_time_utc(wind_kind, profile.launch_time_utc)
    )
    return (wind_kind.strip().lower(), nominal_cycle)

def grouped_indexed_profiles(profiles, wind_kind: str):
    groups = {}
    for idx, profile in enumerate(profiles):
        key = profile_batch_key(profile, wind_kind)
        groups.setdefault(key, []).append((idx, profile))
    return groups

def chunk_group(group, chunk_size: int):
    batch = []
    for item in group:
        batch.append(item)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch

def wrap_lon_180(lon_deg):
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0

def run_profiles(
    mission_profiles,
    dt,
    wind_kind,
    use_multiprocessing=False,
    max_workers=4,
    chunk_size=10,
    wind_verbose=True,
):
    flight_profiles = [None] * len(mission_profiles)

    groups = grouped_indexed_profiles(mission_profiles, wind_kind)
    batches = []
    for group in groups.values():
        batches.extend(chunk_group(group, chunk_size))

    total = len(mission_profiles)
    start = time.perf_counter()

    if not use_multiprocessing:
        done = 0
        for batch in batches:
            pairs = predictor_batch(
                batch,
                dt=dt,
                wind_kind=wind_kind,
                logging=False,
                wind_verbose=wind_verbose,
            )
            for idx, prof in pairs:
                flight_profiles[idx] = prof
            done += len(pairs)
            mp_progress(done, total, start)

        sys.stdout.write("\n")
        sys.stdout.flush()
        end = time.perf_counter()
        print(f"Singleprocessing {total} profiles in {end - start:.2f} seconds.")
        return flight_profiles

    done = 0
    mp_progress(0, total, start)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        submit = partial(
            predictor_batch,
            dt=dt,
            wind_kind=wind_kind,
            logging=False,
            wind_verbose=wind_verbose,
        )
        futures = [executor.submit(submit, batch) for batch in batches]

        for fut in as_completed(futures):
            pairs = fut.result()
            for idx, prof in pairs:
                flight_profiles[idx] = prof
            done += len(pairs)
            mp_progress(done, total, start)

    sys.stdout.write("\n")
    sys.stdout.flush()
    end = time.perf_counter()
    print(f"Multiprocessing {total} profiles in {end - start:.2f} seconds.")
    return flight_profiles

def expand_bounds_to_aspect(xmin, xmax, ymin, ymax, width, height):
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    target_aspect = width / height
    data_aspect = xspan / yspan

    if data_aspect < target_aspect:
        # too tall/narrow -> expand x
        new_xspan = yspan * target_aspect
        xmin = cx - 0.5 * new_xspan
        xmax = cx + 0.5 * new_xspan
    else:
        # too wide/short -> expand y
        new_yspan = xspan / target_aspect
        ymin = cy - 0.5 * new_yspan
        ymax = cy + 0.5 * new_yspan

    return xmin, xmax, ymin, ymax

def time_range_utc_minutes(start, end, step_minutes):
    t0 = parse_utc_time(start)
    t1 = parse_utc_time(end)

    times = []
    t = t0
    while t <= t1:
        times.append(format_utc_time(t))
        t += timedelta(minutes=step_minutes)

    return times

def sample_series_index(times_s, t_query_s):
    if t_query_s <= times_s[0]:
        return 0
    if t_query_s >= times_s[-1]:
        return len(times_s) - 1
    return int(np.searchsorted(times_s, t_query_s, side="right") - 1)

def make_single_flight_gif(
    flight_profile,
    node_observation_profiles,
    mission_profiles,
    draw_basemap=True,
    output_path="flight_animation.gif",
    seconds_per_frame=300.0,
    fps=20,
    fig_w=10,
    fig_h=10,
):
    trace_lon = np.array([wrap_lon_180(x) for x in flight_profile.longitudes], dtype=float)
    trace_lat = np.array(flight_profile.latitudes, dtype=float)
    burst_lon = wrap_lon_180(flight_profile.burst_longitude)
    burst_lat = float(flight_profile.burst_latitude)
    land_lon = wrap_lon_180(flight_profile.longitudes[-1])
    land_lat = float(flight_profile.latitudes[-1])

    launch_lat = mission_profiles[0].launch_site.latitude
    launch_lon = wrap_lon_180(mission_profiles[0].launch_site.longitude)

    node_lons = np.array([wrap_lon_180(obs.node_longitude) for obs in node_observation_profiles], dtype=float)
    node_lats = np.array([float(obs.node_latitude) for obs in node_observation_profiles], dtype=float)

    all_lon = np.concatenate([[launch_lon, burst_lon, land_lon], trace_lon, node_lons])
    all_lat = np.concatenate([[launch_lat, burst_lat, land_lat], trace_lat, node_lats])

    xmin = all_lon.min()
    xmax = all_lon.max()
    ymin = all_lat.min()
    ymax = all_lat.max()

    lon_span = xmax - xmin
    lat_span = ymax - ymin

    lon_pad = max(0.01, 0.08 * lon_span)
    lat_pad = max(0.01, 0.08 * lat_span)

    xmin -= lon_pad
    xmax += lon_pad
    ymin -= lat_pad
    ymax += lat_pad

    xmin, xmax, ymin, ymax = expand_bounds_to_aspect(
        xmin, xmax, ymin, ymax, fig_w, fig_h
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    if draw_basemap:
        try:
            cx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=cx.providers.OpenStreetMap.Mapnik,
                attribution_size=6,
            )
        except Exception as e:
            print(f"Warning: basemap download failed: {e}")

    # static markers
    ax.scatter(
        [land_lon],
        [land_lat],
        s=30,
        alpha=0.75,
        label="Landing location",
        zorder=9,
    )

    ax.scatter(
        [burst_lon],
        [burst_lat],
        s=45,
        marker="^",
        alpha=0.9,
        label="Burst location",
        zorder=10,
    )

    ax.scatter(
        [launch_lon],
        [launch_lat],
        s=120,
        marker="*",
        label="Launch location",
        zorder=11,
    )

    ax.scatter(
        node_lons,
        node_lats,
        s=60,
        marker="s",
        alpha=0.9,
        label="Ground nodes",
        zorder=11,
    )

    # animated artists
    path_line, = ax.plot([], [], linewidth=2.5, color="black", zorder=12, label="Flight path")
    current_dot, = ax.plot([], [], marker="o", markersize=8, color="black", zorder=13)

    link_lines = []
    for _ in node_observation_profiles:
        line, = ax.plot([], [], linewidth=2.0, zorder=8)
        link_lines.append(line)

    ax.set_title("Launch, Burst, and Landing Locations")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend()

    flight_times = np.array(flight_profile.times, dtype=float)
    t_end = float(flight_times[-1])
    frame_times = np.arange(0.0, t_end + seconds_per_frame, seconds_per_frame)
    if frame_times[-1] < t_end:
        frame_times = np.append(frame_times, t_end)

    def init():
        path_line.set_data([], [])
        current_dot.set_data([], [])
        for line in link_lines:
            line.set_data([], [])
        return [path_line, current_dot, *link_lines]

    def update(frame_idx):
        t_now = float(frame_times[frame_idx])
        i = sample_series_index(flight_times, t_now)

        path_line.set_data(trace_lon[: i + 1], trace_lat[: i + 1])
        current_dot.set_data([trace_lon[i]], [trace_lat[i]])

        for obs, line in zip(node_observation_profiles, link_lines):
            j = sample_series_index(np.array(obs.times, dtype=float), t_now)

            x0 = wrap_lon_180(obs.node_longitude)
            y0 = float(obs.node_latitude)
            x1 = trace_lon[i]
            y1 = trace_lat[i]

            line.set_data([x0, x1], [y0, y1])
            line.set_color("green" if obs.terrain_clear[j] else "red")

        ax.set_title(
            f"Launch, Burst, and Landing Locations\n"
            f"T+{t_now/60.0:.1f} min"
        )

        return [path_line, current_dot, *link_lines]

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frame_times),
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

if __name__ == "__main__":
    # ---- Mission setup ----
    launch_site = LaunchSite(40.446387, -104.637853)

    payload_mass = 1.6 #2.4 #np.linspace(1.8, 2.2, 5)
    fill_volume = 75.25 #145 #np.linspace(1.8, 2.2, 5)
    drag_coeff = 0.37 #deprecated

    launch_times_utc = "2025-10-18 15:00" #"2026-04-11 15:30" #time_range_utc_minutes("2026-03-19 00:00", "2026-03-19 12:00", 30)

    mission_profiles = []
    #for launch_time_utc in launch_times_utc:
    b = Balloon(0.60, 6.02, drag_coeff, "Helium", float(fill_volume))  # Kaymont 600g
    # b = Balloon(0.80, 7.00, 0.55, "Helium", float(v_fill))    # Kaymont 800g
    # b = Balloon(1.00, 7.86, 0.55, "Helium", float(v_fill))    # Kaymont 1000g
    # b = Balloon(1.20, 8.63, 0.55, "Helium", float(v_fill))    # Kaymont 1200g
    # b = Balloon(1.50, 9.44, 0.55, "Helium", float(v_fill))    # Kaymont 1500g
    # b = Balloon(2.00, 10.54, 0.55, "Helium", float(v_fill))   # Kaymont 2000g
    # b = Balloon(3.00, 13.00, 0.55, "Helium", float(v_fill))   # Kaymont 3000g
    # b = Balloon(4.00, 15.06, 0.55, "Helium", float(v_fill))   # Kaymont 4000g
    p = Payload(payload_mass, 4 * 0.3048, 0.39)
    mission_profiles.append(
        MissionProfile(
            launch_site=launch_site,
            balloon=b,
            payload=p,
            launch_time_utc=launch_times_utc,
        )
    )

    ground_nodes = [
        GroundNode(
            latitude=40.446387,
            longitude=-104.637853,
            altitude=None,
            name="Launch Site GS",
        ),
    ]

    '''GroundNode(
        latitude=40.575846,
        longitude=-105.082805,
        altitude=None,
        name="CSU Engineering GS",
    ),
    GroundNode(
        latitude=40.265453,
        longitude=-103.640109,
        altitude=None,
        name="Brush Love's Gas Station",
    ),
    GroundNode(
        latitude=39.741689,
        longitude=-104.434049,
        altitude=None,
        name="Bennett Love's Gas Station",
    ),'''
    
    # ---- Wind selection ----
    # Choose ONE wind source for the whole batch run.
    WIND_KIND1 = "gfs0p25" 
    WIND_KIND2 = "gfs0p50" 
    WIND_KIND3 = "gfs1p00" 
    WIND_KIND4 = "hrrr"
    DRAW_GROUND_TRACES = True
    DRAW_BASEMAP = True
    WIND_VERBOSE = True
    CHECK_NODE_TERRAIN_LOS = True
    NODE_TERRAIN_STEP_M = 250.0
    NODE_TERRAIN_CLEARANCE_M = 5.0

    CD_SCALE = 1 #2.4

    MAKE_SINGLE_FLIGHT_GIF = False
    GIF_PROFILE_INDEX = 0
    GIF_SECONDS_PER_FRAME = 30.0
    GIF_FPS = 20
    GIF_OUTPUT_PATH = "260411_Prediction.gif"

    if not WIND_VERBOSE:
        os.environ["HERBIE_VERBOSE"] = "false"

    dt = 0.20
    
    def merge_flight_profile_lists(*lists, drop_none=True):
        merged = []

        for lst in lists:
            if lst is None:
                continue

            for fp in lst:
                if fp is None and drop_none:
                    continue
                merged.append(fp)

        return merged

    fp1 = run_profiles(
        mission_profiles=mission_profiles,
        dt=dt,
        wind_kind=WIND_KIND1,
        use_multiprocessing=False,
        max_workers=4,
        chunk_size=10,
        wind_verbose=WIND_VERBOSE,
    )

    fp2 = run_profiles(
        mission_profiles=mission_profiles,
        dt=dt,
        wind_kind=WIND_KIND2,
        use_multiprocessing=False,
        max_workers=4,
        chunk_size=10,
        wind_verbose=WIND_VERBOSE,
    )

    fp3 = run_profiles(
        mission_profiles=mission_profiles,
        dt=dt,
        wind_kind=WIND_KIND3,
        use_multiprocessing=False,
        max_workers=4,
        chunk_size=10,
        wind_verbose=WIND_VERBOSE,
    )

    fp4 = run_profiles(
        mission_profiles=mission_profiles,
        dt=dt,
        wind_kind=WIND_KIND4,
        use_multiprocessing=False,
        max_workers=4,
        chunk_size=10,
        wind_verbose=WIND_VERBOSE,
    )

    flight_profiles = merge_flight_profile_lists(fp1, fp2, fp3, fp4)

    '''postprocess_terrain = ETOPO1Terrain()
    node_observation_profiles = compute_node_observations_batch(
        flight_profiles=flight_profiles,
        ground_nodes=ground_nodes,
        terrain=postprocess_terrain,
        check_terrain=CHECK_NODE_TERRAIN_LOS,
        terrain_step_m=NODE_TERRAIN_STEP_M,
        terrain_clearance_m=NODE_TERRAIN_CLEARANCE_M,
    )'''

    # ---- Post-processing ----
    df = profiles_to_dataframe([p for p in flight_profiles if p is not None])

    def ascent_cd_avg(fp) -> float:
        if fp is None or not np.isfinite(fp.burst_time):
            return np.nan

        n = min(len(fp.times), len(fp.cd_values))
        if n == 0:
            return np.nan

        times = np.asarray(fp.times[:n], dtype=float)
        cds = np.asarray(fp.cd_values[:n], dtype=float)

        mask = times <= float(fp.burst_time)
        if not np.any(mask):
            return np.nan

        return float(cds[mask].mean())
    
    valid_profiles = [p for p in flight_profiles if p is not None]
    df["Cd_ascent_avg"] = [ascent_cd_avg(fp) for fp in valid_profiles]

    '''# ---- Real flight telemetry (4/11 flight) ----
    telemetry = pd.read_csv("gps_gga_stripped.csv")

    telemetry = telemetry[
        (telemetry["present"] == 1)
        & (telemetry["altitude_present"] == 1)
        & (telemetry["latitude_valid"] == 1)
        & (telemetry["longitude_valid"] == 1)
    ].copy()

    telemetry = telemetry[[
        "system_timestamp_ms",
        "altitude_m",
        "latitude_degrees",
        "longitude_degrees",
    ]].dropna()

    telemetry["t_sec"] = (
        telemetry["system_timestamp_ms"] - telemetry["system_timestamp_ms"].iloc[0]
    ) / 1000.0

    telemetry["t_min"] = telemetry["t_sec"] / 60.0
    telemetry["longitude_wrapped"] = telemetry["longitude_degrees"].apply(wrap_lon_180)

    telemetry_t_min = telemetry["t_min"].to_numpy(dtype=float)
    telemetry_alt_m = telemetry["altitude_m"].to_numpy(dtype=float)
    telemetry_lat = telemetry["latitude_degrees"].to_numpy(dtype=float)
    telemetry_lon = telemetry["longitude_wrapped"].to_numpy(dtype=float)'''

    # ---- Real flight telemetry (10/18 flight) ----
    telemetry_1018 = pd.read_csv("Flight_10-18-25_telem.csv")

    telemetry_1018 = telemetry_1018[[
        "timestamp",
        "latitude (dd)",
        "longitude (dd)",
        "altitude (m)",
    ]].dropna().copy()

    telemetry_1018["timestamp"] = pd.to_datetime(telemetry_1018["timestamp"])

    telemetry_1018["t_sec"] = (
        telemetry_1018["timestamp"] - telemetry_1018["timestamp"].iloc[0]
    ).dt.total_seconds()

    telemetry_1018["t_min"] = telemetry_1018["t_sec"] / 60.0
    telemetry_1018["longitude_wrapped"] = telemetry_1018["longitude (dd)"].apply(wrap_lon_180)

    telemetry_1018_t_min = telemetry_1018["t_min"].to_numpy(dtype=float)
    telemetry_1018_alt_m = telemetry_1018["altitude (m)"].to_numpy(dtype=float)
    telemetry_1018_lat   = telemetry_1018["latitude (dd)"].to_numpy(dtype=float)
    telemetry_1018_lon   = telemetry_1018["longitude_wrapped"].to_numpy(dtype=float)

    if not df.empty:
        df["vbar0_mps"] = (df["burst_altitude"] - df["launch_site.altitude"]) / df["burst_time"]
        df["ok_burst"] = df["burst_altitude"].notna() & df["burst_time"].notna() & (df["burst_time"] > 0)
        df["sane"] = (
            df["ok_burst"]
            & np.isfinite(df["vbar0_mps"])
            & (df["vbar0_mps"] > 0)
            & (df["vbar0_mps"] < 20)
            & (df["burst_altitude"] > 1000)
            & (df["burst_altitude"] < 60000)
        )

        # Convenience landing columns for quick checks / future plotting.
        df["landing_latitude"] = [fp.latitudes[-1] for fp in flight_profiles if fp is not None]
        df["landing_longitude"] = [wrap_lon_180(fp.longitudes[-1]) for fp in flight_profiles if fp is not None]
        df["landing_altitude"] = [fp.altitudes[-1] for fp in flight_profiles if fp is not None]
        df["landing_ground_altitude"] = [fp.ground_altitudes[-1] for fp in flight_profiles if fp is not None]
        df["burst_latitude"] = [fp.burst_latitude for fp in flight_profiles if fp is not None]
        df["burst_longitude"] = [wrap_lon_180(fp.burst_longitude) for fp in flight_profiles if fp is not None]

        df["vbar_dec_mps"] = (df["burst_altitude"] - df["landing_altitude"]) / (df["flight_time"] - df["burst_time"])

        #lat0 = np.deg2rad(40.891263)
        #lon0 = np.deg2rad(-103.835922)

        lat0 = np.deg2rad(39.418195)
        lon0 = np.deg2rad(-101.834588)

        lat = np.deg2rad(df["landing_latitude"])
        lon = np.deg2rad(df["landing_longitude"])

        dlat = lat - lat0
        dlon = lon - lon0

        a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lat)*np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        df["real_separation"] = 6371000 * c

        print(f"Total profiles: {len(df)}")
        print(f"Successful bursts: {int(df['ok_burst'].sum())}")
        print(f"Plotted (sane): {int(df['sane'].sum())}")
        print(df[[
            "payload.mass",
            "balloon.gas_volume",
            "Cd_ascent_avg",
            "launch_time_utc",
            "wind_kind",
            "wind_cycle_utc",
            "vbar0_mps",
            "burst_time",
            "burst_altitude",
            "burst_latitude",
            "burst_longitude",
            "vbar_dec_mps",
            "flight_time",
            "landing_altitude",
            "landing_latitude",
            "landing_longitude",
            "real_separation",
        ]])

    '''node_df = profiles_to_dataframe([q for q in node_observation_profiles if q is not None])

    if not node_df.empty:
        print(f"Node observation profiles: {len(node_df)}")
        print(node_df[[
            "node_name",
            "flight_profile_index",
            "launch_time_utc",
            "node_latitude",
            "node_longitude",
            "node_altitude",
        ]])'''
    
    # ---- Launch / Burst / Landing map ----
    sane_df = df[df["sane"]].copy()

    if not sane_df.empty:
        fig_w, fig_h = 10, 12
        fig, (ax_alt, ax_map) = plt.subplots(
            2, 1,
            figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [1, 2]},
            constrained_layout=True,
        )

        launch_lat = mission_profiles[0].launch_site.latitude
        launch_lon = wrap_lon_180(mission_profiles[0].launch_site.longitude)

        land_lat = sane_df["landing_latitude"].to_numpy(dtype=float)
        land_lon = sane_df["landing_longitude"].to_numpy(dtype=float)

        burst_lat = sane_df["burst_latitude"].to_numpy(dtype=float)
        burst_lon = sane_df["burst_longitude"].to_numpy(dtype=float)

        sane_profiles = [
            fp for fp, ok in zip([p for p in flight_profiles if p is not None], df["sane"].tolist())
            if ok
        ]

        # -------------------------
        # Top panel: altitude vs time
        # -------------------------
        trajectory_color = "#5f84b8"
        marker_color = "#1f3552"

        for fp in sane_profiles:
            t_min = np.array(fp.times, dtype=float) / 60.0
            alt_m = np.array(fp.altitudes, dtype=float)
            ground_m = np.array(fp.ground_altitudes, dtype=float)

            ax_alt.plot(
                t_min,
                alt_m,
                linewidth=2.0,
                alpha=0.85,
                color=trajectory_color,
            )

            # Optional terrain line
            ax_alt.plot(
                t_min,
                ground_m,
                linewidth=1.0,
                alpha=0.5,
                color=marker_color,
                linestyle="--",
            )

            # Optional burst marker
            if np.isfinite(fp.burst_time) and np.isfinite(fp.burst_altitude):
                ax_alt.scatter(
                    fp.burst_time / 60.0,
                    fp.burst_altitude,
                    s=30,
                    color=marker_color,
                    marker="^",
                    zorder=5,
                )

            # Landing marker
            ax_alt.scatter(
                fp.times[-1] / 60.0,
                fp.altitudes[-1],
                s=24,
                color=marker_color,
                marker="o",
                zorder=5,
            )

        '''# Real flight altitude
        ax_alt.plot(
            telemetry_t_min,
            telemetry_alt_m,
            color="red",
            linestyle="--",
            linewidth=2.0,
            label="Actual Flight (04/11)",
            zorder=19,
        )'''

        ax_alt.plot(
            telemetry_1018_t_min,
            telemetry_1018_alt_m,
            color="red",
            linestyle="--",
            linewidth=2.0,
            label="Actual Flight (10/18)",
            zorder=19,
        )

        ax_alt.set_title("Altitude and Trajectory Prediction Comparison for 11 April 2026")
        ax_alt.set_ylabel("Altitude (m)")
        ax_alt.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax_alt.set_axisbelow(True)

        alt_legend_handles = [
            #Line2D([0], [0], color="red", lw=2.0, linestyle="--", label="Actual Flight Track (04/11)"),
            #Line2D([0], [0], marker="X", linestyle="None", color="red", markersize=7, label="Actual Flight Landing (04/11)"),
            Line2D([0], [0], color="red", lw=2.0, linestyle="--", label="Actual Flight Track (10/18)"),
            Line2D([0], [0], marker="X", linestyle="None", color="red", markersize=7, label="Actual Flight Landing (10/18)"),
        ]
        ax_alt.legend(handles=alt_legend_handles, loc="best")

        # -------------------------
        # Bottom panel: map
        # -------------------------
        all_lon_parts = [[launch_lon], land_lon, burst_lon]
        all_lat_parts = [[launch_lat], land_lat, burst_lat]

        if DRAW_GROUND_TRACES:
            for fp in sane_profiles:
                trace_lon = np.array([wrap_lon_180(x) for x in fp.longitudes], dtype=float)
                trace_lat = np.array(fp.latitudes, dtype=float)
                all_lon_parts.append(trace_lon)
                all_lat_parts.append(trace_lat)

        all_lon = np.concatenate(all_lon_parts)
        all_lat = np.concatenate(all_lat_parts)

        xmin = all_lon.min()
        xmax = all_lon.max()
        ymin = all_lat.min()
        ymax = all_lat.max()

        lon_span = xmax - xmin
        lat_span = ymax - ymin

        lon_pad = max(0.01, 0.08 * lon_span)
        lat_pad = max(0.01, 0.08 * lat_span)

        xmin -= lon_pad
        xmax += lon_pad
        ymin -= lat_pad
        ymax += lat_pad

        xmin, xmax, ymin, ymax = expand_bounds_to_aspect(
            xmin, xmax, ymin, ymax, fig_w, fig_h * (2 / 3)
        )

        ax_map.set_xlim(xmin, xmax)
        ax_map.set_ylim(ymin, ymax)
        ax_map.set_aspect("equal", adjustable="box")

        if DRAW_BASEMAP:
            try:
                BASEMAP_STYLE = 0

                if BASEMAP_STYLE == "esri":
                    BASEMAP_PROVIDER = cx.providers.Esri.WorldImagery
                    BASEMAP_ZOOM = 10
                else:
                    BASEMAP_PROVIDER = cx.providers.OpenStreetMap.Mapnik
                    BASEMAP_ZOOM = 10

                cx.add_basemap(
                    ax_map,
                    crs="EPSG:4326",
                    source=BASEMAP_PROVIDER,
                    zoom=BASEMAP_ZOOM,
                    attribution_size=6,
                )
            except Exception as e:
                print(f"Warning: basemap download failed: {e}")

        trajectory_lw = 2.0
        trajectory_alpha = 0.85
        marker_size_small = 42
        marker_size_true = 60

        if DRAW_GROUND_TRACES:
            for fp in sane_profiles:
                trace_lon = np.array([wrap_lon_180(x) for x in fp.longitudes], dtype=float)
                trace_lat = np.array(fp.latitudes, dtype=float)
                ax_map.plot(
                    trace_lon,
                    trace_lat,
                    linewidth=trajectory_lw,
                    alpha=trajectory_alpha,
                    color=trajectory_color,
                    zorder=8,
                )

        '''# Real flight ground track
        ax_map.plot(
            telemetry_lon,
            telemetry_lat,
            color="red",
            linestyle="--",
            linewidth=2.0,
            zorder=19,
        )

        # Real flight landing point
        ax_map.scatter(
            [telemetry_lon[-1]],
            [telemetry_lat[-1]],
            s=60,
            color="red",
            marker="X",
            zorder=20,
        )'''

        ax_map.plot(
            telemetry_1018_lon,
            telemetry_1018_lat,
            color="red",
            linestyle="--",
            linewidth=2.0,
            zorder=19,
        )

        ax_map.scatter(
            [telemetry_1018_lon[-1]],
            [telemetry_1018_lat[-1]],
            s=60,
            color="red",
            marker="X",
            zorder=20,
        )

        ax_map.scatter(
            land_lon,
            land_lat,
            s=marker_size_small,
            color=marker_color,
            marker="o",
            edgecolors="none",
            zorder=11,
        )

        ax_map.scatter(
            [launch_lon],
            [launch_lat],
            s=marker_size_small,
            color=marker_color,
            marker="s",
            edgecolors="none",
            zorder=12,
        )

        '''ax_map.scatter(
            -103.835922,
            40.891263,
            s=marker_size_true,
            color=marker_color,
            marker="x",
            linewidths=2.0,
            zorder=13,
        )'''

        '''ax_map.scatter(
            -101.834588,
            39.418195,
            s=marker_size_true,
            color=marker_color,
            marker="x",
            linewidths=2.0,
            zorder=13,
        )'''

        ax_map.set_xlabel("Longitude (deg)")
        ax_map.set_ylabel("Latitude (deg)")
        ax_map.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax_map.set_axisbelow(True)

        legend_handles = [
            Line2D(
                [0], [0],
                color=trajectory_color,
                lw=trajectory_lw,
                alpha=trajectory_alpha,
                label="Predicted Trajectories",
            ),
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markersize=6,
                label="Predicted Landings",
            ),
            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markersize=6,
                label="Launch Site",
            ),
            #Line2D(
            #    [0], [0],
            #    marker="x",
            #    linestyle="None",
            #    color=marker_color,
            #    markersize=7,
            #    markeredgewidth=2.0,
            #    label="Real Landing",
            #),
            Line2D(
                [0], [0],
                color="red",
                lw=2.5,
                linestyle="--",
                label="Actual Flight Track",
            ),
            Line2D(
                [0], [0],
                marker="X",
                linestyle="None",
                color="red",
                markersize=7,
                label="Actual Flight Landing",
            ),
        ]
        ax_map.legend(handles=legend_handles, loc="best")

        plt.show()

    '''if MAKE_SINGLE_FLIGHT_GIF:
        fp = flight_profiles[GIF_PROFILE_INDEX]

        if fp is not None:
            obs_for_fp = [
                obs for obs in node_observation_profiles
                if obs.flight_profile_index == GIF_PROFILE_INDEX
            ]

            make_single_flight_gif(
                flight_profile=fp,
                node_observation_profiles=obs_for_fp,
                mission_profiles=mission_profiles,
                draw_basemap=DRAW_BASEMAP,
                output_path=GIF_OUTPUT_PATH,
                seconds_per_frame=GIF_SECONDS_PER_FRAME,
                fps=GIF_FPS,
                fig_w=10,
                fig_h=10,
            )
            print(f"Saved GIF to {GIF_OUTPUT_PATH}")'''