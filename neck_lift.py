from model import LaunchSite, Balloon, Payload, MissionProfile
from atmosphere import standardAtmosphere

launch_sites = [
    LaunchSite(1426)
]

balloons = [
    Balloon(0.60, 6.02, 0.55, "Helium", 260) #140 cuft
]

payloads = [
    Payload(2.4, 4 * 0.3048, 0.5)
]

mission_profiles = [
    MissionProfile(launch_sites[0], balloons[0], payloads[0])#,
]

profile = mission_profiles[0]

atmosphere = standardAtmosphere()

altitude  = 1426

pressure, temperature, density, gravity = atmosphere._Qualities(altitude)
volume = profile.balloon.gas_moles * (1.380622 * 6.022169) * temperature / pressure / 1000
mass = profile.balloon.mass + (4.002602 * profile.balloon.gas_moles / 1000)
buoyant_force = density * gravity * volume
weight_force = gravity * mass

neck_lift1 = (buoyant_force - weight_force) / gravity
net_lift1 = (neck_lift1 - profile.payload.mass)

print("\nMolar Mass Approach (Used in Model)")
print(f"Neck lift: {neck_lift1} kg")
print(f"Net lift: {net_lift1} kg")

R_air = .287 #kJ/kg-K
R_He = 2.077

rho_air = pressure / (R_air * temperature)
rho_He = pressure / (R_He * temperature)

mass_balloon = 0.6 #kg

neck_lift2 = volume * (rho_air - rho_He) - mass_balloon
net_lift2 = (neck_lift2 - profile.payload.mass)

print("\nGas Constant/Density Approach")
print(f"Neck lift: {neck_lift2} kg")
print(f"Net lift: {net_lift2} kg\n")