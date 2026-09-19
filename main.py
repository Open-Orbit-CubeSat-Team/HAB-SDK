from model import LaunchSite, Balloon, Payload, MissionProfile#, Model
import run
import numpy as np
import pandas as pd
import time
from functools import partial
from atmosphere import standardAtmosphere
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Affine2D, IdentityTransform
import sys

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
    sys.stdout.write(f"\rProcessing {total} profiles... |{bar}| {done:>5}/{total}  ({100*frac:>3.0f}%)  "
                     f"{rate:>5.2f} profiles/s  ETA {eta:>6.1f}s   ")
    sys.stdout.flush()

def chunked_indexed(profiles, chunk_size: int):
    """Yield lists of (idx, profile) of length <= chunk_size."""
    batch = []
    for i, p in enumerate(profiles):
        batch.append((i, p))
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch

if __name__ == "__main__":
    launch_site = LaunchSite(0.0)  # MSL reference for the chart

    # --- Sweep ranges (first take; adjust after you see the envelope) ---
    fill_volumes = np.linspace(0, 250, 101)            # ft^3  300: 0-250, 600,1500: 0-300; 4000: 100-400;
    suspended_masses = np.linspace(0, 3, 101)          # kg

    mission_profiles = []

    for m_payload in suspended_masses:
        for v_fill in fill_volumes:
            b = Balloon(0.30, 3.78, 0.55, "Helium", float(v_fill))  #Kaymont 300g
            #b = Balloon(0.60, 6.02, 0.55, "Helium", float(v_fill))  #Kaymont 600g
            #b = Balloon(0.80, 7.00, 0.55, "Helium", float(v_fill))  #Kaymont 800g
            #b = Balloon(1.00, 7.86, 0.55, "Helium", float(v_fill))  #Kaymont 1000g
            #b = Balloon(1.20, 8.63, 0.55, "Helium", float(v_fill))  #Kaymont 1200g
            #b = Balloon(1.50, 9.44, 0.55, "Helium", float(v_fill))  #Kaymont 1500g
            #b = Balloon(2.00, 10.54, 0.55, "Helium", float(v_fill)) #Kaymont 2000g
            #b = Balloon(3.00, 13.00, 0.55, "Helium", float(v_fill)) #Kaymont 3000g
            #b = Balloon(4.00, 15.06, 0.55, "Helium", float(v_fill)) #Kaymont 4000g
            p = Payload(m_payload, 4 * 0.3048, 0.5)
            mission_profiles.append(MissionProfile(launch_site, b, p))

    #flight_profiles = []

    flight_profiles = [None] * len(mission_profiles)

    # ---- BATCHED MULTIPROCESSING ----
    dt = 0.15
    max_workers = 16

    # start with 20–50; tune later
    CHUNK_SIZE = 25

    batches = list(chunked_indexed(mission_profiles, CHUNK_SIZE))
    total = len(mission_profiles)
    done = 0

    '''mission_profiles = [MissionProfile(
        LaunchSite(1422), 
        Balloon(0.60, 6.02, 0.55, "Helium", 125), 
        Payload(1.5, 4 * 0.3048, 0.5)
    )]

    flight_profiles = []

    start = time.perf_counter()
    Model(0.1, mission_profiles, flight_profiles).altitude_model(True, 1)
    end= time.perf_counter()
    print(f"Singleprocessing {int(len(mission_profiles))} Profiles in {end - start:.2f} seconds.")'''

    start = time.perf_counter()
    mp_progress(0, total, start)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        submit = partial(run.predictor_batch, dt=dt, logging=False, interval=100)
        futures = [executor.submit(submit, batch) for batch in batches]
        for fut in as_completed(futures):
            pairs = fut.result()  # list[(idx, FlightProfile|None)]
            for idx, prof in pairs:
                flight_profiles[idx] = prof
            done += len(pairs)
            mp_progress(done, total, start)
    end= time.perf_counter()
    sys.stdout.write("\033[?25h\n")
    sys.stdout.flush()
    print(f"Multiprocessing {total} Profiles in {end - start:.2f} seconds.")

    # Build dataframe (handles nested attrs like balloon.gas_volume, payload.mass, etc.)
    df = profiles_to_dataframe([p for p in flight_profiles if p is not None])

    # Add reference-average ascent rate (MSL, USSA76): vbar0 = z_burst / t_burst
    # (Only defined when burst_time is finite & > 0)
    df["vbar0_mps"] = df["burst_altitude"] / df["burst_time"]

    # Mark "successful burst" runs
    df["ok_burst"] = df["burst_altitude"].notna() & df["burst_time"].notna() & (df["burst_time"] > 0)

    # A sanity filter for numerical blow-ups (keeps plots stable)
    # You can tighten/loosen these once you see real envelopes.
    df["sane"] = (
        df["ok_burst"]
        & np.isfinite(df["vbar0_mps"])
        & (df["vbar0_mps"] > 0)
        & (df["vbar0_mps"] < 20)          # ascent rate should not be anywhere near 20 m/s in normal HAB ops
        & (df["burst_altitude"] > 1000)   # ignore trivial "burst" near ground
        & (df["burst_altitude"] < 60000)  # ignore pathological overshoots
    )

    df_plot = df.loc[df["sane"]].copy()

    print(f"Total profiles: {len(df)}")
    print(f"Successful bursts: {df['ok_burst'].sum()}")
    print(f"Plotted (sane): {len(df_plot)}")

    # --- Compute initial net force at launch (neutral buoyancy boundary) ---
    atm = standardAtmosphere()
    launch_alt = float(df["launch_site.altitude"].iloc[0])  # should be 0 for your chart

    p0, T0, rho0, g0 = atm._Qualities(launch_alt)

    R_u = (1.380622 * 6.022169)  # same constant used in model.py
    # Volume at launch from moles under ambient conditions (match your model's formula)
    V0_m3 = df["balloon.gas_moles"] * R_u * T0 / p0 / 1000.0

    # Helium mass (kg) — same as in model
    m_He = df["balloon.gas_moles"] * 4.002602 / 1000.0

    m_total = df["payload.mass"] + df["balloon.mass"] + m_He

    Fnet0 = rho0 * g0 * V0_m3 - m_total * g0            # N
    a0 = Fnet0 / m_total                                 # m/s^2

    # Keep only finite points for contouring
    mask0 = np.isfinite(a0) & np.isfinite(df["payload.mass"]) & np.isfinite(df["balloon.gas_volume"])
    x0 = df.loc[mask0, "payload.mass"].to_numpy()
    y0 = df.loc[mask0, "balloon.gas_volume"].to_numpy()   # ft^3 in your current convention
    a0v = a0.loc[mask0].to_numpy()

    # --- Build regular grids from the original sweep ---
    m_vals = np.sort(df["payload.mass"].unique())
    v_vals = np.sort(df["balloon.gas_volume"].unique())
    M, V = np.meshgrid(m_vals, v_vals, indexing="xy")

    df_grid = df.copy()
    df_grid["a0"] = a0
    df_grid["alt_km"] = df_grid["burst_altitude"] / 1000.0
    df_grid["tburst_min"] = df_grid["burst_time"] / 60.0

    # Background field: use sane burst velocity where available.
    # Force the non-ascending region to 0 m/s so the color field reaches
    # the neutral buoyancy boundary cleanly, then cover infeasible space
    # with a gray overlay.
    df_grid["vbar_plot"] = np.nan
    df_grid.loc[df_grid["sane"], "vbar_plot"] = df_grid.loc[df_grid["sane"], "vbar0_mps"]
    df_grid.loc[df_grid["a0"] <= 0.0, "vbar_plot"] = 0.0

    def pivot_grid(frame, value_col):
        return (
            frame.pivot(index="balloon.gas_volume", columns="payload.mass", values=value_col)
                 .reindex(index=v_vals, columns=m_vals)
                 .to_numpy()
        )

    Z_vbar = pivot_grid(df_grid, "vbar_plot")
    Z_alt = pivot_grid(df_grid, "alt_km")
    Z_tburst = pivot_grid(df_grid, "tburst_min")
    Z_a0 = pivot_grid(df_grid, "a0")

    def draw_contour_reference_axis(
        ax,
        contour_set,
        *,
        side="left",
        fmt="{:.0f}",
        title="Burst Altitude (km)",
        spine_x_axes=0.0,
        tick_len_axes=0.008,
        label_pad_axes=0.014,
        title_pad_axes=0.042,
        min_sep_axes=0.032,
        fontsize=9,
        title_fontsize=None,
        spine_lw=None,
        tick_y_offset_axes=0.0012,
    ):
        if spine_lw is None:
            spine_lw = plt.rcParams["axes.linewidth"]
        if title_fontsize is None:
            title_fontsize = plt.rcParams["axes.labelsize"]

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if side == "left":
            x_boundary = xmin
            spine_x = spine_x_axes
            tick_x0 = spine_x
            tick_x1 = spine_x - tick_len_axes
            label_x = spine_x - label_pad_axes
            title_x = spine_x - title_pad_axes
            label_ha = "right"
            title_rot = 90
        elif side == "right":
            x_boundary = xmax
            spine_x = spine_x_axes
            tick_x0 = spine_x
            tick_x1 = spine_x + tick_len_axes
            label_x = spine_x + label_pad_axes
            title_x = spine_x + title_pad_axes
            label_ha = "left"
            title_rot = 270
        else:
            raise ValueError("side must be 'left' or 'right'")

        raw_positions = []

        for level, segs in zip(contour_set.levels, contour_set.allsegs):
            y_hits = []

            for seg in segs:
                if len(seg) < 2:
                    continue

                x = seg[:, 0]
                y = seg[:, 1]

                for i in range(len(seg) - 1):
                    x0, x1 = x[i], x[i + 1]
                    y0, y1 = y[i], y[i + 1]

                    crosses = ((x0 <= x_boundary <= x1) or (x1 <= x_boundary <= x0))
                    if not crosses:
                        continue

                    if np.isclose(x1, x0):
                        y_hit = y0
                    else:
                        t = (x_boundary - x0) / (x1 - x0)
                        if 0.0 <= t <= 1.0:
                            y_hit = y0 + t * (y1 - y0)
                        else:
                            continue

                    if ymin <= y_hit <= ymax:
                        y_hits.append(y_hit)

            if not y_hits:
                continue

            y_data = float(np.median(y_hits))
            y_axes = ax.transAxes.inverted().transform(
                ax.transData.transform((x_boundary, y_data))
            )[1]
            raw_positions.append((level, y_axes))

        if not raw_positions:
            return

        raw_positions.sort(key=lambda t: t[1])

        placed = []
        for level, y_axes in raw_positions:
            y_new = y_axes
            if placed and (y_new - placed[-1][1] < min_sep_axes):
                y_new = placed[-1][1] + min_sep_axes
            placed.append((level, y_new))

        if placed:
            overflow = placed[-1][1] - 1.0
            if overflow > 0:
                placed = [(lvl, y - overflow) for lvl, y in placed]

        for i in range(len(placed) - 2, -1, -1):
            lvl_i, y_i = placed[i]
            _, y_next = placed[i + 1]
            if y_next - y_i < min_sep_axes:
                placed[i] = (lvl_i, y_next - min_sep_axes)

        placed = [(lvl, y) for (lvl, y) in placed if 0.0 <= y <= 1.0]
        if not placed:
            return

        # Full-height custom spine: top corner to bottom corner of plot area
        ax.plot(
            [spine_x, spine_x], [0.0, 1.0],
            transform=ax.transAxes,
            color="black",
            lw=spine_lw,
            clip_on=False,
            zorder=10,
        )

        for level, y_axes in placed:
            ax.plot(
                [tick_x0, tick_x1], [y_axes + tick_y_offset_axes, y_axes + tick_y_offset_axes],
                transform=ax.transAxes,
                color="black",
                lw=spine_lw,
                clip_on=False,
                zorder=10,
            )
            ax.text(
                label_x, y_axes,
                fmt.format(level),
                transform=ax.transAxes,
                ha=label_ha,
                va="center",
                fontsize=fontsize,
                clip_on=False,
                zorder=10,
            )

        ax.text(
            title_x, 0.5,
            title,
            transform=ax.transAxes,
            rotation=title_rot,
            ha="center",
            va="center",
            fontsize=title_fontsize,
            clip_on=False,
            zorder=10,
        )

    fig, ax = plt.subplots(figsize=(10, 7))

    # Match the visible axis spine thickness
    axis_lw = plt.rcParams["axes.linewidth"]
    axis_label_fs = plt.rcParams["axes.labelsize"]

    # Leave room on both sides for the custom left-side burst-altitude axis
    # and the right-side helium axis + colorbar.
    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.10)

    ax.spines["left"].set_visible(False)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", which="both",
                left=False, labelleft=False,
                right=True, labelright=True)

    ax.set_ylabel("Helium Fill Volume (ft³)")

    # Set final axis geometry before any helpers that depend on the plot
    # boundaries (for example, the custom left-side burst-altitude labels).
    ax.set_xlim(m_vals.min(), m_vals.max())
    ax.set_ylim(v_vals.min(), v_vals.max())
    ax.margins(x=0.0, y=0.0)

    # Map ascent velocity to custom red -> green -> yellow field.
    # Fill remaining NaNs above the neutral line from nearby valid cells
    # so the background reads as a continuous design-space field.
    v_target = 5.0
    v_floor = 0.0
    Z_plot = Z_vbar.copy()

    for _ in range(12):
        nan_mask = np.isnan(Z_plot) & np.isfinite(Z_a0) & (Z_a0 > 0.0)
        if not np.any(nan_mask):
            break

        neighbor_stack = np.stack([
            np.roll(Z_plot,  1, axis=0),
            np.roll(Z_plot, -1, axis=0),
            np.roll(Z_plot,  1, axis=1),
            np.roll(Z_plot, -1, axis=1),
        ])

        valid_neighbors = np.isfinite(neighbor_stack)
        counts = valid_neighbors.sum(axis=0)
        sums = np.nansum(neighbor_stack, axis=0)
        fill_vals = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        Z_plot[nan_mask & np.isfinite(fill_vals)] = fill_vals[nan_mask & np.isfinite(fill_vals)]

    Z_plot = np.clip(np.nan_to_num(Z_plot, nan=0.0), 0.0, None)
    vmax = np.nanmax(Z_plot) if np.isfinite(np.nanmax(Z_plot)) else v_target
    vmax = max(vmax, v_target)

    def map_vbar_to_c(zvals, v_target, v_floor, vmax):
        cvals = np.empty_like(zvals, dtype=float)

        below = zvals <= v_target
        above = zvals > v_target

        if v_target > v_floor:
            cvals[below] = 0.60 * np.sqrt((zvals[below] - v_floor) / (v_target - v_floor))
        else:
            cvals[below] = 0.60

        if vmax > v_target:
            cvals[above] = 1.00 - 0.40 * np.sqrt((vmax - zvals[above]) / (vmax - v_target))
        else:
            cvals[above] = 1.00

        return np.clip(cvals, 0.0, 1.0)

    # Upsample the background field for plotting so the mesh reads as a
    # smoother gradient without rerunning the full simulation sweep.
    def upsample_grid(x_old, y_old, z_old, nx=401, ny=401):
        x_new = np.linspace(x_old.min(), x_old.max(), nx)
        y_new = np.linspace(y_old.min(), y_old.max(), ny)

        # Interpolate along x for each original y row
        z_x = np.empty((len(y_old), len(x_new)), dtype=float)
        for i in range(len(y_old)):
            z_x[i, :] = np.interp(x_new, x_old, z_old[i, :])

        # Interpolate along y for each new x column
        z_new = np.empty((len(y_new), len(x_new)), dtype=float)
        for j in range(len(x_new)):
            z_new[:, j] = np.interp(y_new, y_old, z_x[:, j])

        return x_new, y_new, z_new

    def centers_to_edges(vals):
        vals = np.asarray(vals, dtype=float)
        edges = np.empty(len(vals) + 1, dtype=float)
        edges[1:-1] = 0.5 * (vals[:-1] + vals[1:])
        edges[0] = vals[0] - 0.5 * (vals[1] - vals[0])
        edges[-1] = vals[-1] + 0.5 * (vals[-1] - vals[-2])
        return edges
    
    def draw_text_with_background(
        ax,
        x,
        y,
        text,
        *,
        transform=None,
        ha="left",
        va="top",
        fontsize=8,
        family=None,
        rotation=0.0,
        rotation_mode="anchor",
        color="black",
        clip_on=False,
        zorder=20,
        background=False,
        facecolor="0.8",
        edgecolor="none",
        boxstyle="round",
        core_pad=0.16,
        feather=False,
        feather_pads=(0.28, 0.42, 0.62),
        feather_alphas=(0.28, 0.16, 0.08),
        multiline_mode="block",
        line_spacing=1.20,
        sample_field=None,
        sample_x=None,
        sample_y=None,
        sample_cmap=None,
        sample_infeasible_mask=None,
        sample_infeasible_color=(0.8, 0.8, 0.8, 1.0),
    ):
        if transform is None:
            transform = ax.transAxes

        common = dict(
            transform=transform,
            ha=ha,
            va=va,
            fontsize=fontsize,
            family=family,
            rotation=rotation,
            rotation_mode=rotation_mode,
            color=color,
            clip_on=clip_on,
        )

        def _bilinear_interp_grid(xq, yq, xg, yg, Z):
            xq = np.asarray(xq)
            yq = np.asarray(yq)

            ix = np.searchsorted(xg, xq) - 1
            iy = np.searchsorted(yg, yq) - 1

            ix = np.clip(ix, 0, len(xg) - 2)
            iy = np.clip(iy, 0, len(yg) - 2)

            x0 = xg[ix]
            x1 = xg[ix + 1]
            y0 = yg[iy]
            y1 = yg[iy + 1]

            tx = np.divide(xq - x0, x1 - x0, out=np.zeros_like(xq, dtype=float), where=(x1 != x0))
            ty = np.divide(yq - y0, y1 - y0, out=np.zeros_like(yq, dtype=float), where=(y1 != y0))

            z00 = Z[iy, ix]
            z10 = Z[iy, ix + 1]
            z01 = Z[iy + 1, ix]
            z11 = Z[iy + 1, ix + 1]

            return (
                (1 - tx) * (1 - ty) * z00 +
                tx * (1 - ty) * z10 +
                (1 - tx) * ty * z01 +
                tx * ty * z11
            )

        def _render_sampled_box(xi, yi, s, pad, alpha, zo, draw_edge=False):
            fig = ax.figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # Use the real text bbox, then expand it with the SAME pad semantics
            probe_common = common.copy()
            probe_common["rotation"] = 0.0
            probe_common["rotation_mode"] = "anchor"

            probe = ax.text(
                xi, yi, s,
                alpha=0.0,
                zorder=zo,
                **probe_common,
            )
            fig.canvas.draw()
            bbox_disp = probe.get_window_extent(renderer=renderer)
            probe.remove()

            pad_px = pad * probe.get_fontsize() * fig.dpi / 72.0

            x0d = bbox_disp.x0 - pad_px
            x1d = bbox_disp.x1 + pad_px
            y0d = bbox_disp.y0 - pad_px
            y1d = bbox_disp.y1 + pad_px

            w_disp = x1d - x0d
            h_disp = y1d - y0d
            cx_disp = 0.5 * (x0d + x1d)
            cy_disp = 0.5 * (y0d + y1d)

            # Pixel grid in the label's LOCAL display-space box
            nx = max(8, int(np.ceil(w_disp)))
            ny = max(8, int(np.ceil(h_disp)))

            x_local = np.linspace(-0.5 * w_disp, 0.5 * w_disp, nx)
            y_local = np.linspace(-0.5 * h_disp, 0.5 * h_disp, ny)
            XL, YL = np.meshgrid(x_local, y_local)

            theta = np.deg2rad(rotation)
            c = np.cos(theta)
            s_ = np.sin(theta)

            # Rotate local display coords into actual display coords
            XD = c * XL - s_ * YL + cx_disp
            YD = s_ * XL + c * YL + cy_disp

            # Map each display pixel back to data space
            pts_data = ax.transData.inverted().transform(
                np.column_stack([XD.ravel(), YD.ravel()])
            )
            XQ = pts_data[:, 0].reshape(ny, nx)
            YQ = pts_data[:, 1].reshape(ny, nx)

            # Sample the actual plotted gradient field under the box footprint
            vals = _bilinear_interp_grid(XQ, YQ, sample_x, sample_y, sample_field)
            rgba = sample_cmap(vals)

            if sample_infeasible_mask is not None:
                infeasible_vals = _bilinear_interp_grid(XQ, YQ, sample_x, sample_y, sample_infeasible_mask)
                infeasible = infeasible_vals > 0.5
                rgba[infeasible] = sample_infeasible_color

            rgba[..., 3] *= alpha

            # Draw in the label's own local display coordinates
            local_to_display = (
                Affine2D()
                .rotate_deg(rotation)
                .translate(cx_disp, cy_disp)
                + IdentityTransform()
            )

            im = ax.imshow(
                rgba,
                extent=[-0.5 * w_disp, 0.5 * w_disp, -0.5 * h_disp, 0.5 * h_disp],
                origin="lower",
                interpolation="bilinear",
                transform=local_to_display,
                zorder=zo,
                clip_on=False,
                aspect="auto",
            )

            patch = FancyBboxPatch(
                (-0.5 * w_disp, -0.5 * h_disp),
                w_disp,
                h_disp,
                boxstyle=f"{boxstyle},pad=0.0",
                facecolor="none",
                edgecolor=edgecolor if draw_edge else "none",
                linewidth=0.8 if draw_edge else 0.0,
                transform=local_to_display,
                clip_on=False,
                zorder=zo + 0.01,
            )
            ax.add_patch(patch)
            im.set_clip_path(patch)

        def _draw_one_line(xi, yi, s, zo):
            if not background:
                return ax.text(xi, yi, s, zorder=zo, **common)

            use_sampled = (
                sample_field is not None
                and sample_x is not None
                and sample_y is not None
                and sample_cmap is not None
            )

            if feather:
                for pad, alpha in zip(feather_pads, feather_alphas):
                    if use_sampled:
                        _render_sampled_box(xi, yi, s, pad, alpha, zo - 0.2, draw_edge=False)
                    else:
                        ax.text(
                            xi, yi, s,
                            zorder=zo - 0.2,
                            bbox=dict(
                                boxstyle=f"{boxstyle},pad={pad}",
                                facecolor=facecolor,
                                edgecolor="none",
                                alpha=alpha,
                            ),
                            **common,
                        )

            if use_sampled:
                _render_sampled_box(xi, yi, s, core_pad, 1.0, zo - 0.05, draw_edge=False)
                return ax.text(xi, yi, s, zorder=zo, **common)

            return ax.text(
                xi, yi, s,
                zorder=zo,
                bbox=dict(
                    boxstyle=f"{boxstyle},pad={core_pad}",
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=1.0,
                ),
                **common,
            )

        if multiline_mode == "block" or "\n" not in text:
            return _draw_one_line(x, y, text, zorder)

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        probe = ax.text(
            x, y, "Ag",
            transform=transform,
            ha=ha,
            va=va,
            fontsize=fontsize,
            family=family,
            rotation=rotation,
            rotation_mode=rotation_mode,
            color=color,
            alpha=0.0,
            clip_on=clip_on,
        )
        fig.canvas.draw()
        bbox = probe.get_window_extent(renderer=renderer)
        probe.remove()

        line_h_px = bbox.height * line_spacing
        x_disp, y_disp = transform.transform((x, y))

        artists = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            y_line_disp = y_disp - i * line_h_px
            x_line, y_line = transform.inverted().transform((x_disp, y_line_disp))
            artists.append(_draw_one_line(x_line, y_line, line, zorder))

        return artists
    
    def label_contour_offset(
        ax,
        contour_set,
        text,
        *,
        frac=0.55,
        offset_pts=8,
        side="below",
        text_kwargs=None,
        tangent_window_pts=36,
    ):
        """
        Place a label near a contour, rotated to match the local contour tangent
        in display space, then render it using draw_text_with_background().
        """
        if text_kwargs is None:
            text_kwargs = {}

        segs = contour_set.allsegs[0]
        seg = max(segs, key=len)

        if len(seg) < 3:
            raise ValueError("Contour segment too short to label")

        # Work in display space so the angle matches what is actually drawn.
        seg_disp = ax.transData.transform(seg)

        # Arc length along the contour in display pixels
        dxy = np.diff(seg_disp, axis=0)
        ds = np.hypot(dxy[:, 0], dxy[:, 1])
        s = np.concatenate([[0.0], np.cumsum(ds)])
        s_total = s[-1]

        if s_total <= 0:
            raise ValueError("Degenerate contour segment")

        # Anchor point by fraction of total contour length
        s_target = np.clip(frac, 0.0, 1.0) * s_total
        i = int(np.clip(np.searchsorted(s, s_target), 1, len(seg) - 2))

        # Interpolate the exact anchor point on the segment
        s0 = s[i - 1]
        s1 = s[i]
        if s1 > s0:
            t = (s_target - s0) / (s1 - s0)
        else:
            t = 0.0

        pA = seg[i - 1]
        pB = seg[i]
        p_anchor = (1.0 - t) * pA + t * pB

        pA_disp = seg_disp[i - 1]
        pB_disp = seg_disp[i]
        p_anchor_disp = (1.0 - t) * pA_disp + t * pB_disp

        # Compute tangent using a symmetric window in ARC LENGTH, not vertex index
        s_lo = max(0.0, s_target - tangent_window_pts)
        s_hi = min(s_total, s_target + tangent_window_pts)

        j_lo = max(0, np.searchsorted(s, s_lo) - 1)
        j_hi = min(len(seg) - 1, np.searchsorted(s, s_hi))

        if j_hi <= j_lo:
            j_lo = max(0, i - 1)
            j_hi = min(len(seg) - 1, i + 1)

        p_lo = seg_disp[j_lo]
        p_hi = seg_disp[j_hi]

        dx = p_hi[0] - p_lo[0]
        dy = p_hi[1] - p_lo[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Unit normal in display space
        n = np.array([-dy, dx], dtype=float)
        n_norm = np.hypot(n[0], n[1])
        if n_norm > 0:
            n /= n_norm
        else:
            n[:] = 0.0

        if side == "below":
            n *= -1.0
        elif side == "above":
            n *= 1.0
        else:
            raise ValueError("side must be 'below' or 'above'")

        # Offset label position in display space, then map back to data space
        px = p_anchor_disp[0] + n[0] * offset_pts * ax.figure.dpi / 72.0
        py = p_anchor_disp[1] + n[1] * offset_pts * ax.figure.dpi / 72.0
        x_lab, y_lab = ax.transData.inverted().transform((px, py))

        return draw_text_with_background(
            ax,
            x_lab,
            y_lab,
            text,
            transform=ax.transData,
            ha="center",
            va="center",
            rotation=angle,
            rotation_mode="anchor",
            **text_kwargs,
        )

    x_hi, y_hi, Z_plot_hi = upsample_grid(m_vals, v_vals, Z_plot, nx=401, ny=401)
    _, _, Z_a0_hi = upsample_grid(m_vals, v_vals, Z_a0, nx=401, ny=401)
    C_hi = map_vbar_to_c(Z_plot_hi, v_target, v_floor, vmax)
    X_edges_hi = centers_to_edges(x_hi)
    Y_edges_hi = centers_to_edges(y_hi)

    ##################################################
    # Start Color Gradient                           #
    ##################################################

    vbar_cmap = LinearSegmentedColormap.from_list(
        "vbar_design",
        [
            (0.00, "#d73027"),
            (0.40, "#f46d43"),
            (0.60, "#1a9850"),
            (0.70, "#8cbc6e"),
            (0.80, "#fee08b"),
            (1.00, "#fee08b"),
        ]
    )

    bg = ax.pcolormesh(
        X_edges_hi, Y_edges_hi, C_hi,
        shading="flat",
        cmap=vbar_cmap,
        zorder=1
    )

    '''##################################################
    # Start Black and White Gradient                 #
    ##################################################

    bw_target_cmap = LinearSegmentedColormap.from_list(
        "bw_target_soft",
        [
            (0.00, "#f4f4f4"),
            (0.40, "#f4f4f4"),
            (0.60, "#9a9a9a"),
            (0.70, "#f4f4f4"),
            (1.00, "#f4f4f4"),
        ]
    )

    bg = ax.pcolormesh(
        X_edges_hi, Y_edges_hi, C_hi,
        shading="flat",
        cmap=bw_target_cmap,
        zorder=1
    )'''

    # 2) Shade infeasible region on top using the same hi-res plotting
    # grid as the background, so no red fringe can peek through.
    infeasible_mask_hi = np.where(Z_a0_hi <= 0.0, 1.0, np.nan)
    infeasible_cmap = LinearSegmentedColormap.from_list(
        "infeasible_gray",
        ["0.8", "0.8"]
    )
    ax.pcolormesh(
        X_edges_hi, Y_edges_hi, infeasible_mask_hi,
        shading="flat",
        cmap=infeasible_cmap,
        zorder=2
    )

    # 3) Neutral buoyancy boundary
    c0 = ax.contour(
        M, V, Z_a0,
        levels=[0.0],
        linewidths=2.0,
        colors="k",
        zorder=5
    )

    label_contour_offset(
        ax,
        c0,
        "↑ Neutrally Buoyant (Lift = Weight)",
        frac=0.50,
        offset_pts=8,
        side="below",
        text_kwargs=dict(
            fontsize=9,
            background=True,
            facecolor="0.8",
            feather=True,
            core_pad=0.15,
            feather_pads=(0.17, 0.22, 0.25),
            feather_alphas=(0.80, 0.40, 0.20),
            zorder=8,
        ),
    )

    label_contour_offset(
        ax,
        c0,
        "↓ Negatively Buoyant at Launch",
        frac=0.80,
        offset_pts=8,
        side="below",
        text_kwargs=dict(
            fontsize=9,
            background=True,
            facecolor="0.8",
            feather=True,
            core_pad=0.15,
            feather_pads=(0.17, 0.22, 0.25),
            feather_alphas=(0.80, 0.40, 0.20),
            zorder=8,
        ),
    )

    label_contour_offset(
        ax,
        c0,
        "High Dispersion Risk",
        frac=0.90,
        offset_pts=7,
        side="above",
        text_kwargs=dict(
            fontsize=8,
            color="black",
            background=True,
            feather=True,
            core_pad=0.15,
            feather_pads=(0.17, 0.22, 0.25),
            feather_alphas=(0.80, 0.40, 0.20),
            sample_field=C_hi,
            sample_x=x_hi,
            sample_y=y_hi,
            sample_cmap=vbar_cmap,              #Color: vbar_cmap, BW: bw_target_cmap
            sample_infeasible_mask=(Z_a0_hi <= 0.0).astype(float),
            zorder=7,
        ),
    )

    # Target ascent-rate contour (5 m/s)
    cs_vtarget = ax.contour(
        x_hi, y_hi, Z_plot_hi,
        levels=[5.0],
        colors="#177540",           #Color: #177540, BW: #9a9a9a
        linewidths=1.5,
        linestyles="--",
        zorder=5
    )

    label_contour_offset(
        ax,
        cs_vtarget,
        "Recommended Ascent Rate (5 m/s)",
        frac=0.85,
        offset_pts=7,
        side="below",
        text_kwargs=dict(
            fontsize=8,
            color="black",
            background=True,
            feather=True,
            core_pad=0.15,
            feather_pads=(0.17, 0.22, 0.25),
            feather_alphas=(0.80, 0.40, 0.20),
            sample_field=C_hi,
            sample_x=x_hi,
            sample_y=y_hi,
            sample_cmap=vbar_cmap,              #Color: vbar_cmap, BW: bw_target_cmap
            sample_infeasible_mask=(Z_a0_hi <= 0.0).astype(float),
            zorder=7,
        ),
    )

    # 4) Performance contours on top
    alt_levels = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 28] #300g [36, 37, 38, 39, 40, 41, 42] 1500g [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] #600g [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 34]
    t_levels = [30, 35, 40, 45, 50, 60, 75, 90, 120] #300g [70, 75, 80, 85, 90, 100, 110, 120, 150, 240]#1500g [55, 60, 65, 70, 80, 90, 100, 120, 150, 240] #600g [40, 45, 50, 55, 60, 65, 75, 90, 120, 180]

    cs1 = ax.contour(
        M, V, Z_alt,
        levels=alt_levels,
        linewidths=0.8,
        colors="k",
        alpha=0.65,
        zorder=6
    )
    # Keep the altitude contour family labeled from the outside like a
    # chart axis, rather than repeating labels inside the field.
    draw_contour_reference_axis(
        ax,
        cs1,
        side="left",
        fmt="{:.0f}",
        title="Burst Altitude (km)",
        spine_x_axes=0.0,
        tick_len_axes=0.008,
        label_pad_axes=0.014,
        title_pad_axes=0.042,
        min_sep_axes=0.032,
        fontsize=9,
        title_fontsize=plt.rcParams["axes.labelsize"],
        spine_lw=plt.rcParams["axes.linewidth"],
        tick_y_offset_axes=0.0012,
    )

    cs3 = ax.contour(
        M, V, Z_tburst,
        levels=t_levels,
        linewidths=0.8,
        colors="k",
        alpha=0.55,
        zorder=6
    )
    ax.clabel(cs3, fmt="%.0f min", inline=True, fontsize=8, inline_spacing=10)

    '''#Color Legend
    legend_text = "\n".join([
            "Ascent Rate Gradient:",
            "Yellow = Excessive Cost",
            "Green = Target Ascent Rate (~5 m/s)",
            "Red = High Dispersion Risk",
            "Gray = Negatively Buoyant",
        ])

    #Monochrome Legend
    legend_text = "\n".join([
        "Ascent Rate Gradient:",
        "Above = Excessive Cost",
        "Center = Target Ascent Rate (~5 m/s)",
        "Below = High Dispersion Risk",
        "Solid = Negatively Buoyant",
    ])

    draw_text_with_background(
        ax,
        0.71, 0.15,
        legend_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        background=True,
        facecolor="0.8",
        feather=True,
        core_pad=0.15,
        feather_pads=(0.17, 0.22, 0.25),
        feather_alphas=(0.80, 0.40, 0.20),
        multiline_mode="per_line",
        line_spacing=1.20,
        zorder=20,
    )'''

    ax.set_title("Kaymont 300g Meteorological Balloon Design Space (MSL, USSA76)")
    ax.set_xlabel("Payload Mass (kg)")

    ax.xaxis.set_minor_locator(tic.AutoMinorLocator())
    ax.yaxis.set_minor_locator(tic.AutoMinorLocator())

    ax.minorticks_on()
    ax.set_axisbelow(False)
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    #ax.tick_params(axis='x', bottom=False)

    ax.grid(
        True,
        linestyle='--',
        which="major",
        color="black",
        linewidth=0.65,
        alpha=0.35
    )

    ax.grid(
        True,
        linestyle='--',
        which="minor",
        color="black",
        linewidth=0.5,
        alpha=0.35
    )
    plt.show()