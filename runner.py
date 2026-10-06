from model import LaunchSite, Balloon, Payload, MissionProfile
from main import run_profiles, profiles_to_dataframe#, merge_flight_profile_lists
import pandas as pd
import os
import numpy as np

# ---- Mission setup ----
launch_site = LaunchSite(40.446387, -104.637853,1500)

payload_mass = 1 #kg
fill_volume = 100 #ft3
drag_coeff = 0.37  # deprecated

launch_time_utc = "2025-10-18 15:00"

mission_profiles = []
#for launch_time_utc in launch_times_utc:
b = Balloon(0.30, 3.78, drag_coeff, "Helium", float(fill_volume))  # Kaymont 600g
# b = Balloon(0.60, 6.02, drag_coeff, "Helium", float(fill_volume))  # Kaymont 600g
# b = Balloon(0.80, 7.00, 0.55, "Helium", float(v_fill))    # Kaymont 800g
# b = Balloon(1.00, 7.86, 0.55, "Helium", float(v_fill))    # Kaymont 1000g
# b = Balloon(1.20, 8.63, 0.55, "Helium", float(v_fill))    # Kaymont 1200g
# b = Balloon(1.50, 9.44, 0.55, "Helium", float(v_fill))    # Kaymont 1500g
# b = Balloon(2.00, 10.54, 0.55, "Helium", float(v_fill))   # Kaymont 2000g
# b = Balloon(3.00, 13.00, 0.55, "Helium", float(v_fill))   # Kaymont 3000g
# b = Balloon(4.00, 15.06, 0.55, "Helium", float(v_fill))   # Kaymont 4000g
p = Payload(payload_mass, 4 * 0.3048, 0.39)

#Create a Single Mission
mission_profiles.append(
    MissionProfile(
        launch_site=launch_site,
        balloon=b,
        payload=p,
        launch_time_utc=launch_time_utc,
    )
)

#print(mission_profiles)

WIND_KIND = "gfs1p00"
WIND_VERBOSE = True
dt = 0.20

if __name__ == "__main__":
    flp = run_profiles(
        mission_profiles=mission_profiles,
        dt=dt,
        wind_kind=WIND_KIND,
        use_multiprocessing=False,
        max_workers=4,
        chunk_size=10,
        wind_verbose=WIND_VERBOSE,
    )
    
    #fp = merge_flight_profile_lists(flight_profiles)
    

    fp = flp[0]
    #$df = profiles_to_dataframe(flp)
    print (np.size(fp))
   # print (df)

    if fp is None:
        print("Flight profile generation failed.")
    else:
        n = min(
            len(fp.times),
            len(fp.pressures),
            len(fp.temperatures),
            len(fp.densities),
            len(fp.gravities),
        )

        df = pd.DataFrame({
            "time_s": fp.times[:n],
            "latitude": fp.latitudes[:n],
            "longitude": fp.longitudes[:n],
            "altitude_m": fp.altitudes[:n],
            "ground_altitude_m": fp.ground_altitudes[:n],
            "velocity_mps": fp.velocities[:n],
            "acceleration_mps2": fp.accelerations[:n],
            "force_n": fp.forces[:n],
            "pressure_pa": fp.pressures[:n],
            "temperature_k": fp.temperatures[:n],
            "density_kgm3": fp.densities[:n],
            "gravity_mps2": fp.gravities[:n],
            "wind_u_mps": fp.wind_u[:n],
            "wind_v_mps": fp.wind_v[:n],
            "cd": fp.cd_values[:n],
        })

        output_path = "outputs/flight_profile.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved flight profile to {output_path}")
        print(f"Burst altitude: {fp.burst_altitude:.1f} m at t={fp.burst_time:.1f} s")
        print(f"Flight time: {fp.flight_time:.1f} s")
