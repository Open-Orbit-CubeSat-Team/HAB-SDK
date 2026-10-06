from utils import Geometry, Integrator
from atmosphere import standardAtmosphere
from dataclasses import dataclass
from functools import partial
from typing import Optional
import numpy as np
import math
import time
import requests
import gzip
import shutil
import xarray as xr
from pathlib import Path
from datetime import timedelta, datetime, timezone

R_E = 6356766.0

def _wrap_lon_180(lon_deg: float) -> float:
    return ((float(lon_deg) + 180.0) % 360.0) - 180.0

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

@dataclass(frozen=False)
class LaunchSite:
    latitude: float
    longitude: float
    altitude: Optional[float] = None

@dataclass(frozen=False)
class Balloon:
    mass: float
    burst_diameter: float
    drag_coefficient: float
    gas: str
    gas_volume: float
    gas_moles: float = 0.0

    def __post_init__(self):
        self.gas_moles = self.gas_volume * 3.048 ** 3 / 22.413636  # ft^3 to L to mol

@dataclass(frozen=False)
class Payload:
    mass: float
    parachute_diameter: float
    parachute_drag_coefficient: float
    parachute_area: float = 0.0

    def __post_init__(self):
        self.parachute_area = self.parachute_diameter ** 2 * math.pi / 4

@dataclass(frozen=False)
class MissionProfile:
    launch_site: LaunchSite
    balloon: Balloon
    payload: Payload
    launch_time_utc: datetime | str

@dataclass(frozen=False)
class FlightProfile(MissionProfile):
    times: list[float]
    latitudes: list[float]
    longitudes: list[float]
    altitudes: list[float]
    ground_altitudes: list[float]
    velocities: list[float]
    accelerations: list[float]
    forces: list[float]
    pressures: list[float]
    temperatures: list[float]
    densities: list[float]
    gravities: list[float]
    wind_u: list[float]   # east (m/s)
    wind_v: list[float]   # north (m/s)
    cd_values: list[float]
    burst_altitude: float
    burst_latitude: float
    burst_longitude: float
    max_altitude: float
    burst_time: float
    flight_time: float
    wind_kind: str
    wind_cycle_utc: str

@dataclass(frozen=False)
class GroundNode:
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    name: str = "Ground Node"

@dataclass(frozen=False)
class NodeObservationProfile:
    node_name: str
    node_latitude: float
    node_longitude: float
    node_altitude: float
    flight_profile_index: int
    launch_time_utc: datetime | str
    times: list[float]
    ranges_m: list[float]
    azimuths_deg: list[float]
    elevations_deg: list[float]
    terrain_clear: list[bool]

class ConstantTerrain:
    def __init__(self, elevation_m: float):
        self.elevation_m = float(elevation_m)

    def elevation(self, lat_deg: float, lon_deg: float) -> float:
        return self.elevation_m

class ETOPO1Terrain:
    """
    Local ETOPO1 terrain sampler.

    Automatically downloads the global ETOPO1 dataset (~500 MB) the first
    time it is used and stores it in ./terrain_cache/.

    Elevations are sampled via bilinear interpolation.
    """

    def __init__(
        self,
        cache_dir: str | Path = "./terrain_cache",
        filename: str = "ETOPO1_Ice_g_gmt4.grd",
        url: str = (
            "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/"
            "ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz"
        ),
    ):
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.cache_dir / filename
        gz_path = self.cache_dir / (filename + ".gz")

        if not self.path.exists():
            if not gz_path.exists():
                #print("Downloading ETOPO1 terrain dataset (~500 MB)...")
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(gz_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)

            print("Extracting ETOPO1 dataset...")
            with gzip.open(gz_path, "rb") as f_in:
                with open(self.path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove the compressed archive after successful extraction
            try:
                gz_path.unlink()
            except OSError:
                pass

        #print("Loading ETOPO1 terrain dataset...")
        ds = xr.open_dataset(self.path)

        # ETOPO1 grid-registered GMT file uses x=lon, y=lat, z=elevation
        self.lat = ds["y"].values
        self.lon = ds["x"].values
        self.z = ds["z"].values.astype(np.float32, copy=False)

        self.lat_min = float(self.lat.min())
        self.lat_max = float(self.lat.max())
        self.lon_min = float(self.lon.min())
        self.lon_max = float(self.lon.max())

        # Regular-grid spacing for fast direct indexing
        self.lat0 = float(self.lat[0])
        self.lon0 = float(self.lon[0])
        self.dlat = float(self.lat[1] - self.lat[0])
        self.dlon = float(self.lon[1] - self.lon[0])

    def elevation(self, lat_deg: float, lon_deg: float) -> float:
        lon = _wrap_lon_180(lon_deg)
        lat = float(lat_deg)

        if lat < self.lat_min or lat > self.lat_max:
            return 0.0
        if lon < self.lon_min or lon > self.lon_max:
            return 0.0

        # Fast direct indexing on regular grid
        i = int((lat - self.lat0) / self.dlat)
        j = int((lon - self.lon0) / self.dlon)

        i = max(0, min(i, len(self.lat) - 2))
        j = max(0, min(j, len(self.lon) - 2))

        lat0 = float(self.lat[i])
        lat1 = float(self.lat[i + 1])
        lon0 = float(self.lon[j])
        lon1 = float(self.lon[j + 1])

        z00 = float(self.z[i, j])
        z10 = float(self.z[i + 1, j])
        z01 = float(self.z[i, j + 1])
        z11 = float(self.z[i + 1, j + 1])

        # Bilinear interpolation
        tx = 0.0 if lat1 == lat0 else (lat - lat0) / (lat1 - lat0)
        ty = 0.0 if lon1 == lon0 else (lon - lon0) / (lon1 - lon0)

        z = (
            z00 * (1.0 - tx) * (1.0 - ty)
            + z10 * tx * (1.0 - ty)
            + z01 * (1.0 - tx) * ty
            + z11 * tx * ty
        )

        return float(z)

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    alt = float(alt_m)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.array([x, y, z], dtype=float)

def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    return np.array([
        [-sin_lon,              cos_lon,               0.0],
        [-sin_lat * cos_lon,   -sin_lat * sin_lon,     cos_lat],
        [ cos_lat * cos_lon,    cos_lat * sin_lon,     sin_lat],
    ], dtype=float)

def slant_range_az_el(
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_alt_m: float,
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_m: float,
):
    observer_ecef = geodetic_to_ecef(observer_lat_deg, observer_lon_deg, observer_alt_m)
    target_ecef = geodetic_to_ecef(target_lat_deg, target_lon_deg, target_alt_m)

    los_ecef = target_ecef - observer_ecef
    enu = ecef_to_enu_matrix(observer_lat_deg, observer_lon_deg) @ los_ecef

    east = float(enu[0])
    north = float(enu[1])
    up = float(enu[2])

    horizontal = math.hypot(east, north)
    slant_range = float(np.linalg.norm(enu))
    azimuth_deg = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    elevation_deg = math.degrees(math.atan2(up, horizontal))

    return slant_range, azimuth_deg, elevation_deg

def _shortest_dlon_deg(lon1_deg: float, lon0_deg: float) -> float:
    return ((float(lon1_deg) - float(lon0_deg) + 180.0) % 360.0) - 180.0

def _interp_lon_shortest(lon0_deg: float, lon1_deg: float, f: float) -> float:
    return _wrap_lon_180(float(lon0_deg) + f * _shortest_dlon_deg(lon1_deg, lon0_deg))

def _surface_distance_m(lat0_deg: float, lon0_deg: float, lat1_deg: float, lon1_deg: float) -> float:
    lat0 = math.radians(float(lat0_deg))
    lat1 = math.radians(float(lat1_deg))
    dlat = lat1 - lat0
    dlon = math.radians(_shortest_dlon_deg(lon1_deg, lon0_deg))
    latm = 0.5 * (lat0 + lat1)

    x = dlon * math.cos(latm)
    y = dlat
    return R_E * math.hypot(x, y)

def terrain_line_of_sight_clear(
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_alt_m: float,
    target_lat_deg: float,
    target_lon_deg: float,
    target_alt_m: float,
    terrain,
    sample_step_m: float = 500.0,
    clearance_m: float = 0.0,
    target_elevation_deg: float | None = None,
) -> bool:
    if target_elevation_deg is not None and target_elevation_deg <= 0.0:
        return False

    horizontal_m = _surface_distance_m(
        observer_lat_deg, observer_lon_deg,
        target_lat_deg, target_lon_deg,
    )

    if horizontal_m <= 0.0:
        return True

    n_seg = max(1, int(math.ceil(horizontal_m / float(sample_step_m))))

    for i in range(1, n_seg):
        f = i / n_seg

        lat = (1.0 - f) * float(observer_lat_deg) + f * float(target_lat_deg)
        lon = _interp_lon_shortest(observer_lon_deg, target_lon_deg, f)
        ray_alt = (1.0 - f) * float(observer_alt_m) + f * float(target_alt_m)

        terrain_alt = float(terrain.elevation(lat, lon))
        if terrain_alt + float(clearance_m) >= ray_alt:
            return False

    return True

def resolve_node_altitude(node: GroundNode, terrain) -> float:
    if node.altitude is not None:
        return float(node.altitude)
    return float(terrain.elevation(node.latitude, node.longitude))

def compute_node_observations(
    flight_profile: FlightProfile,
    node: GroundNode,
    terrain,
    flight_profile_index: int,
    check_terrain: bool = True,
    terrain_step_m: float = 500.0,
    terrain_clearance_m: float = 0.0,
) -> NodeObservationProfile:
    node_altitude = resolve_node_altitude(node, terrain)

    ranges_m = []
    azimuths_deg = []
    elevations_deg = []
    terrain_clear = []

    for lat, lon, alt in zip(
        flight_profile.latitudes,
        flight_profile.longitudes,
        flight_profile.altitudes,
    ):
        r, az, el = slant_range_az_el(
            observer_lat_deg=node.latitude,
            observer_lon_deg=node.longitude,
            observer_alt_m=node_altitude,
            target_lat_deg=lat,
            target_lon_deg=lon,
            target_alt_m=alt,
        )

        ranges_m.append(float(r))
        azimuths_deg.append(float(az))
        elevations_deg.append(float(el))

        if check_terrain:
            clear = terrain_line_of_sight_clear(
                observer_lat_deg=node.latitude,
                observer_lon_deg=node.longitude,
                observer_alt_m=node_altitude,
                target_lat_deg=lat,
                target_lon_deg=lon,
                target_alt_m=alt,
                terrain=terrain,
                sample_step_m=terrain_step_m,
                clearance_m=terrain_clearance_m,
                target_elevation_deg=el,
            )
        else:
            clear = True

        terrain_clear.append(bool(clear))

    return NodeObservationProfile(
        node_name=node.name,
        node_latitude=float(node.latitude),
        node_longitude=float(node.longitude),
        node_altitude=float(node_altitude),
        flight_profile_index=int(flight_profile_index),
        launch_time_utc=flight_profile.launch_time_utc,
        times=[float(t) for t in flight_profile.times],
        ranges_m=ranges_m,
        azimuths_deg=azimuths_deg,
        elevations_deg=elevations_deg,
        terrain_clear=terrain_clear,
    )

def compute_node_observations_batch(
    flight_profiles,
    ground_nodes,
    terrain,
    check_terrain: bool = True,
    terrain_step_m: float = 500.0,
    terrain_clearance_m: float = 0.0,
):
    results = []
    for i, fp in enumerate(flight_profiles):
        if fp is None:
            continue
        for node in ground_nodes:
            results.append(
                compute_node_observations(
                    flight_profile=fp,
                    node=node,
                    terrain=terrain,
                    flight_profile_index=i,
                    check_terrain=check_terrain,
                    terrain_step_m=terrain_step_m,
                    terrain_clearance_m=terrain_clearance_m,
                )
            )
    return results

class Model:
    helium_mm = 4.002602
    atmosphere = standardAtmosphere()

    def __init__(self, time_step, profiles, result, wind=None, terrain=None, run_time_utc=None, ascent_cd_scale: float = 1.0):
        self.time_step = time_step
        self.profiles = profiles
        self.result = result
        self.wind = wind
        self.terrain = terrain if terrain is not None else ETOPO1Terrain()
        self.ascent_cd_scale = float(ascent_cd_scale)

        if isinstance(run_time_utc, str):
            self.run_time_utc = datetime.strptime(run_time_utc, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        else:
            self.run_time_utc = run_time_utc

    def _terrain_elevation(self, lat_deg: float, lon_deg: float) -> float:
        return float(self.terrain.elevation(lat_deg, lon_deg))
    
    def _safe_balloon_volume(self, gas_moles: float, temperature: float, pressure: float) -> float:
        if not np.isfinite(temperature) or not np.isfinite(pressure) or pressure <= 0.0:
            return float("inf")
        volume = gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000.0
        return float(volume) if np.isfinite(volume) and volume >= 0.0 else float("inf")
    
    def _state_is_finite(self, r, v, a=None) -> bool:
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            return False
        if a is not None and not np.all(np.isfinite(a)):
            return False
        return True
    
    def _resolve_run_time_utc(self, profile):
        run_time = profile.launch_time_utc

        if isinstance(run_time, str):
            return datetime.strptime(run_time, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)

        return run_time

    def _mu_air_sutherland(self, T_K: float) -> float:
        mu0 = 1.716e-5   # Pa*s
        T0 = 273.15      # K
        S = 110.4        # K
        return mu0 * (T_K / T0) ** 1.5 * (T0 + S) / (T_K + S)

    def _sphere_cd_morrison(self, Re: float) -> float:
        Re = max(float(Re), 1e-12)
        '''return (
            24.0 / Re
            + 2.6 * (Re / 5.0) / (1.0 + (Re / 5.0) ** 1.52)
            + 0.411 * (Re / 2.63e5) ** (-7.94) / (1.0 + (Re / 2.63e5) ** (-8.0))
            + 0.25 * (Re / 1e6) / (1.0 + (Re / 1e6))
        )'''
        return (
            24.0 / Re
            + 2.6 * (Re / 5.0) / (1.0 + (Re / 5.0) ** 1.52)
            + 0.55 * (Re / 4.3e5) ** (-5.44) / (1.0 + (Re / 4.3e5) ** (-5.5))
            + 0.41 * (Re / 6.5e5) ** 5.0 / (1.0 + (Re / 6.5e5) ** 5.0)
        )
    
    '''def _sphere_cd_schiller_naumann(self, Re: float) -> float:
        Re = max(float(Re), 1e-12)

        # Classical Schiller–Naumann:
        #   Cd = 24/Re * (1 + 0.15 Re^0.687),   Re < ~1000
        # Often paired with a constant high-Re plateau above that.
        if Re < 1000.0:
            return 24.0 / Re * (1.0 + 0.15 * Re**0.687)

        return 0.44'''
    
    '''def _sphere_cd_clift_gauvin(self, Re: float) -> float:
        Re = max(float(Re), 1e-12)

        return (
            24.0 / Re * (1.0 + 0.15 * Re**0.687)
            + 0.42 / (1.0 + 42500.0 / Re**1.16)
        )'''
    
    '''def _sphere_cd_hoerner(self, Re: float) -> float:
        Re = max(float(Re), 1e-12)

        # Stokes regime
        if Re < 1.0:
            return 24.0 / Re

        # Transitional
        if Re < 1000.0:
            return 24.0 / Re * (1.0 + 0.15 * Re**0.687)

        # Subcritical (smooth-ish sphere)
        if Re < 2e5:
            return 0.44

        # Drag crisis / rough balloon behavior
        if Re < 1e6:
            return 0.25

        # Supercritical plateau
        return 0.20'''
    
    def _diagnostic_cd(self, r, v, profile, current_time, lat, lon, x_prev, y_prev, buoyant, Cd_fallback):
        z = float(r[2])

        pressure, temperature, density, gravity = self.atmosphere._Qualities(z)

        dx = float(r[0] - x_prev)
        dy = float(r[1] - y_prev)

        lat_new = lat + math.degrees(dy / R_E)
        cos_lat = max(1.0e-8, abs(math.cos(math.radians(lat_new))))
        lon_new = _wrap_lon_180(lon + math.degrees(dx / (R_E * cos_lat)))

        if self.wind:
            u_wind, v_wind = self.wind.uv(current_time, z, lat_new, lon_new)
        else:
            u_wind = v_wind = 0.0

        if not buoyant:
            return float(Cd_fallback)

        volume = self._safe_balloon_volume(profile.balloon.gas_moles, temperature, pressure)
        diameter = (6.0 * volume / np.pi) ** (1.0 / 3.0)

        v_rel = np.array([v[0] - u_wind, v[1] - v_wind, v[2]], dtype=float)
        v_rel_mag = float(np.linalg.norm(v_rel))

        mu = self._mu_air_sutherland(temperature)
        Re = density * v_rel_mag * diameter / max(mu, 1e-12)

        return self.ascent_cd_scale * self._sphere_cd_morrison(Re)

    # -------------------------
    # Core 3-DOF acceleration
    # -------------------------

    def _acceleration(self, r, v, profile, mass, current_time, lat, lon, x_prev, y_prev, Cd, area, buoyant: bool):
        x, y, z = r
        vx, vy, vz = v

        dx = x - x_prev
        dy = y - y_prev

        lat_new = lat + math.degrees(dy / R_E)
        cos_lat = max(1.0e-8, abs(math.cos(math.radians(lat_new))))
        lon_new = _wrap_lon_180(lon + math.degrees(dx / (R_E * cos_lat)))

        pressure, temperature, density, gravity = self.atmosphere._Qualities(z)

        # Wind (east, north)
        if self.wind:
            u_wind, v_wind = self.wind.uv(current_time, z, lat_new, lon_new)
        else:
            u_wind = v_wind = 0.0

        # Relative velocity (balloon - air)
        v_rel = np.array([vx - u_wind, vy - v_wind, vz], dtype=float)
        v_rel_mag = np.linalg.norm(v_rel)

        if buoyant:
            volume = self._safe_balloon_volume(profile.balloon.gas_moles, temperature, pressure)
            diameter = (6.0 * volume / np.pi) ** (1.0 / 3.0)
            area = np.pi * diameter**2 / 4.0

            mu = self._mu_air_sutherland(temperature)
            Re = density * v_rel_mag * diameter / max(mu, 1e-12)
            Cd = self.ascent_cd_scale * self._sphere_cd_morrison(Re)

        # Drag (vector)
        drag = -0.5 * density * Cd * area * v_rel_mag * v_rel

        # Buoyancy & gravity (vertical only)
        if buoyant:
            volume = self._safe_balloon_volume(profile.balloon.gas_moles, temperature, pressure)
            buoyancy = np.array([0.0, 0.0, density * gravity * volume], dtype=float)
        else:
            buoyancy = np.array([0.0, 0.0, 0.0], dtype=float)
        weight = np.array([0.0, 0.0, -gravity * mass], dtype=float)

        F = buoyancy + weight + drag

        '''if buoyant and 15900.0 <= z <= 16100.0:
            print(
                f"z={z:8.1f} "
                f"vz={vz:7.3f} "
                f"rho={density:9.6f} "
                f"T={temperature:7.3f} "
                f"p={pressure:8.4f} "
                f"V={volume:9.4f} "
                f"A={area:8.4f} "
                f"Cd={Cd:7.4f} "
                f"Bz={buoyancy[2]:10.4f} "
                f"Wz={weight[2]:10.4f} "
                f"Dz={drag[2]:10.4f} "
                f"Fz={F[2]:10.4f} "
                f"az={F[2]/mass:9.5f}"
            )'''

        return F / mass

    # -------------------------
    # Main simulation
    # -------------------------

    def altitude_model(self, logging=True):
        start = time.perf_counter()

        for profile in self.profiles:
            profile_run_time_utc = self._resolve_run_time_utc(profile)
            # Initial geographic state
            lat = float(profile.launch_site.latitude)
            lon = float(profile.launch_site.longitude)
            topo_z0 = self._terrain_elevation(lat, lon)
            user_z0 = float(profile.launch_site.altitude) if profile.launch_site.altitude is not None else -np.inf
            z0 = max(topo_z0, user_z0)
            profile.launch_site.altitude = z0

            wind_kind = type(self.wind).__name__.replace("Wind", "").lower()
            wind_cycle = getattr(self.wind, "run_utc_str", "unknown")

            # Local tangent-plane state
            r = np.array([0.0, 0.0, z0], dtype=float)
            v = np.array([0.0, 0.0, 0.0], dtype=float)

            times = [0.0]
            latitudes = [lat]
            longitudes = [lon]
            altitudes = [z0]
            ground_altitudes = [topo_z0]
            velocities = [v.copy()]
            accelerations = [np.zeros(3)]
            forces = [np.zeros(3)]
            pressures = []
            temperatures = []
            densities = []
            gravities = []
            wind_u = []
            wind_v = []
            cd_values = []

            ascent_mass = (
                profile.payload.mass
                + profile.balloon.mass
                + self.helium_mm * profile.balloon.gas_moles / 1000
            )
            descent_mass = profile.payload.mass
            max_steps = 250000
            step_count = 0

            viability_check_time = 300.0
            viability_min_gain = 150.0

            burst_volume = (4 / 3) * math.pi * (profile.balloon.burst_diameter / 2) ** 3
            burst_altitude = None
            burst_latitude = None
            burst_longitude = None
            burst_time = None

            t = 0.0
            x_prev = y_prev = 0.0

            current_time = profile_run_time_utc if profile_run_time_utc else None
            u_wind, v_wind = self.wind.uv(current_time, r[2], lat, lon) if self.wind else (0.0, 0.0)
            if not np.isfinite(u_wind):
                u_wind = 0.0
            if not np.isfinite(v_wind):
                v_wind = 0.0
            wind_u.append(float(u_wind))
            wind_v.append(float(v_wind))

            cd_values.append(
                self._diagnostic_cd(
                    r=r,
                    v=v,
                    profile=profile,
                    current_time=current_time,
                    lat=lat,
                    lon=lon,
                    x_prev=x_prev,
                    y_prev=y_prev,
                    buoyant=True,
                    Cd_fallback=profile.balloon.drag_coefficient,
                )
            )

            while True:
                step_count += 1
                if step_count > max_steps:
                    if burst_altitude is None:
                        burst_altitude = float("nan")
                        burst_time = float("nan")
                    break

                t_prev = t
                lat_prev = lat
                lon_prev = lon
                r_prev = r.copy()
                v_prev = v.copy()
                prev_ground = ground_altitudes[-1]
                current_time = profile_run_time_utc + timedelta(seconds=t) if profile_run_time_utc else None

                pressure, temperature, density, gravity = self.atmosphere._Qualities(r[2])
                pressures.append(pressure)
                temperatures.append(temperature)
                densities.append(density)
                gravities.append(gravity)

                # -------------------------
                # Flight phase selection
                # -------------------------
                if burst_altitude is None:
                    volume = self._safe_balloon_volume(profile.balloon.gas_moles, temperature, pressure)
                    buoyant = True
                    #if volume >= burst_volume:
                    #if pressure <= 5.8345:
                    if r[2] >= 26820.6:
                        burst_altitude = r[2]
                        burst_latitude = lat
                        burst_longitude = lon
                        burst_time = t
                        if descent_mass == 0:
                            break
                        buoyant = False
                        mass = descent_mass
                        Cd = profile.payload.parachute_drag_coefficient
                        area = profile.payload.parachute_area
                    else:
                        mass = ascent_mass
                        Cd = profile.balloon.drag_coefficient
                        area = Geometry.sphere_cross_section(volume)
                else:
                    buoyant = False
                    mass = descent_mass
                    Cd = profile.payload.parachute_drag_coefficient
                    area = profile.payload.parachute_area

                accel_fn = partial(
                    self._acceleration,
                    profile=profile,
                    mass=mass,
                    current_time=current_time,
                    lat=lat,
                    lon=lon,
                    x_prev=x_prev,
                    y_prev=y_prev,
                    Cd=Cd,
                    area=area,
                    buoyant=buoyant,
                )

                r, v, a = Integrator.rk4_second_order(r, v, accel_fn, self.time_step)

                if step_count % 50 == 0:
                    z = float(r[2])
                    #vx, vy, vz = v

                    pressure, temperature, density, gravity = self.atmosphere._Qualities(z)

                    if self.wind:
                        u_wind, v_wind = self.wind.uv(current_time, z, lat, lon)
                    else:
                        u_wind = v_wind = 0.0

                    '''volume = self._safe_balloon_volume(profile.balloon.gas_moles, temperature, pressure)
                    diameter = (6.0 * volume / np.pi) ** (1.0 / 3.0)

                    mu = self._mu_air_sutherland(temperature)

                    v_rel = np.array([vx - u_wind, vy - v_wind, vz])
                    v_rel_mag = np.linalg.norm(v_rel)

                    horiz_rel = np.hypot(vx - u_wind, vy - v_wind)

                    Re_full = density * v_rel_mag * diameter / max(mu, 1e-12)
                    Re_vert = density * abs(vz) * diameter / max(mu, 1e-12)

                    print(
                        f"z={z:8.1f}  "
                        f"vz={vz:7.3f}  "
                        f"hrel={horiz_rel:7.3f}  "
                        f"Re_full={Re_full:10.1f}  "
                        f"Re_vert={Re_vert:10.1f}"
                    )
                    print(
                        f"z={z:8.1f} "
                        f"vz={vz:7.3f} "
                        f"Re={Re_full:10.1f} "
                        f"V={volume:10.3f} "
                        f"D={diameter:8.3f}"
                    )'''

                if not self._state_is_finite(r, v, a):
                    if burst_altitude is None:
                        burst_altitude = float("nan")
                        burst_time = float("nan")
                    break

                # Failed ascent / numerical collapse before burst
                if burst_altitude is None and r[2] <= z0:
                    burst_altitude = float("nan")
                    burst_time = float("nan")

                    times.append(float(t))
                    latitudes.append(float(lat))
                    longitudes.append(float(lon))
                    altitudes.append(float(max(r[2], prev_ground)))
                    ground_altitudes.append(float(prev_ground))
                    velocities.append(v.copy())
                    accelerations.append(a.copy() if np.all(np.isfinite(a)) else np.zeros(3))
                    forces.append((a * mass).copy() if np.all(np.isfinite(a)) else np.zeros(3))
                    wind_u.append(float(u_wind) if np.isfinite(u_wind) else 0.0)
                    wind_v.append(float(v_wind) if np.isfinite(v_wind) else 0.0)
                    cd_values.append(0.0)
                    break

                # Early-ascent viability check
                if burst_altitude is None and t >= viability_check_time:
                    if (r[2] - z0) < viability_min_gain:
                        burst_altitude = float("nan")
                        burst_time = float("nan")
                        break

                # Update geographic coordinates
                dx = r[0] - x_prev
                dy = r[1] - y_prev
                lat_new = lat + math.degrees(dy / R_E)
                cos_lat = max(1.0e-8, abs(math.cos(math.radians(lat_new))))
                lon_new = _wrap_lon_180(lon + math.degrees(dx / (R_E * cos_lat)))
                lat, lon = lat_new, lon_new
                x_prev, y_prev = float(r[0]), float(r[1])
                t += self.time_step

                ground_next = self._terrain_elevation(lat, lon)
                current_time_next = profile_run_time_utc + timedelta(seconds=t) if profile_run_time_utc else None
                u_wind, v_wind = self.wind.uv(current_time_next, r[2], lat, lon) if self.wind else (0.0, 0.0)
                if not np.isfinite(u_wind):
                    u_wind = 0.0
                if not np.isfinite(v_wind):
                    v_wind = 0.0

                times.append(float(t))
                latitudes.append(float(lat))
                longitudes.append(float(lon))
                altitudes.append(float(r[2]))
                ground_altitudes.append(float(ground_next))
                velocities.append(v.copy())
                accelerations.append(a.copy())
                forces.append((a * mass).copy())
                wind_u.append(float(u_wind))
                wind_v.append(float(v_wind))

                cd_values.append(
                    self._diagnostic_cd(
                        r=r,
                        v=v,
                        profile=profile,
                        current_time=current_time_next,
                        lat=lat,
                        lon=lon,
                        x_prev=x_prev,
                        y_prev=y_prev,
                        buoyant=buoyant,
                        Cd_fallback=Cd,
                    )
                )

                # Terrain-aware landing condition after burst
                if burst_altitude is not None and r[2] <= ground_next:
                    h_prev = r_prev[2] - prev_ground
                    h_curr = r[2] - ground_next
                    denom = h_prev - h_curr
                    alpha = 1.0 if abs(denom) < 1.0e-12 else _clamp01(h_prev / denom)

                    touch_t = t_prev + alpha * (t - t_prev)
                    touch_lat = lat_prev + alpha * (lat - lat_prev)
                    touch_lon = lon_prev + alpha * (lon - lon_prev)
                    touch_ground = prev_ground + alpha * (ground_next - prev_ground)
                    touch_v = v_prev + alpha * (v - v_prev)
                    if np.all(np.isfinite(accelerations[-2])) and np.all(np.isfinite(accelerations[-1])):
                        touch_a = accelerations[-2] + alpha * (accelerations[-1] - accelerations[-2])
                    else:
                        touch_a = np.zeros(3)
                    if np.all(np.isfinite(forces[-2])) and np.all(np.isfinite(forces[-1])):
                        touch_f = forces[-2] + alpha * (forces[-1] - forces[-2])
                    else:
                        touch_f = np.zeros(3)
                    touch_u = wind_u[-2] + alpha * (wind_u[-1] - wind_u[-2])
                    touch_vwind = wind_v[-2] + alpha * (wind_v[-1] - wind_v[-2])

                    times[-1] = float(touch_t)
                    latitudes[-1] = float(touch_lat)
                    longitudes[-1] = float(touch_lon)
                    altitudes[-1] = float(touch_ground)
                    ground_altitudes[-1] = float(touch_ground)
                    velocities[-1] = touch_v.copy()
                    accelerations[-1] = touch_a.copy()
                    forces[-1] = touch_f.copy()
                    wind_u[-1] = float(touch_u) if np.isfinite(touch_u) else 0.0
                    wind_v[-1] = float(touch_vwind) if np.isfinite(touch_vwind) else 0.0
                    r[2] = touch_ground
                    break


            self.result.append(
                FlightProfile(
                    profile.launch_site,
                    profile.balloon,
                    profile.payload,
                    profile.launch_time_utc,
                    times,
                    latitudes,
                    longitudes,
                    altitudes,
                    ground_altitudes,
                    velocities,
                    accelerations,
                    forces,
                    pressures,
                    temperatures,
                    densities,
                    gravities,
                    wind_u,
                    wind_v,
                    cd_values,
                    float("nan") if burst_altitude is None else float(burst_altitude),
                    float("nan") if burst_latitude is None else float(burst_latitude),
                    float("nan") if burst_longitude is None else float(_wrap_lon_180(burst_longitude)),
                    float(np.max(altitudes)),
                    float("nan") if burst_time is None else float(burst_time),
                    float(times[-1]),
                    wind_kind=wind_kind,
                    wind_cycle_utc=wind_cycle,
                )
            )

        if logging:
            end = time.perf_counter()
            print(f"Processed {len(self.profiles)} profile(s) in {end - start:.2f} seconds")
