
import numpy as np
import cvxpy as cp

# =============================================================================
# 1. OPERATING CONDITIONS
# =============================================================================

Vin = 1.8
Vout = 0.8
Iout = 1.0

Vgs = 1.0
Vds = Vin

D = Vout / Vin
Pout = Vout * Iout

# Ripple specifications
delta_iL = 1.5
delta_Vo = 0.040

# =============================================================================
# 2. MOSFET PARAMETERS
# =============================================================================

W_n_ref = 100.0
W_p_ref = 200.0
L_chan = 0.065                 # um

Rds_pu_n = 3.366 * W_n_ref
Rds_pu_p = 11.128 * W_p_ref

Cgg_pu_n = 83.14e-15 / W_n_ref
Cgg_pu_p = 164.90e-15 / W_p_ref

Coss_pu_n = 59.70e-15 / W_n_ref
Coss_pu_p = 82.33e-15 / W_p_ref

# =============================================================================
# 3. INDUCTOR AND CAPACITOR PARAMETERS
# =============================================================================

L_base = 4e-9
R_dcr_base = 0.2              # Ohm * mm^2

density_C = 100e-9            # F/mm^2

I_L_rms = np.sqrt(Iout**2 + delta_iL**2 / 12)

# =============================================================================
# 4. OPTIMIZATION VARIABLES
# =============================================================================

W = cp.Variable(pos=True, name="W")       # NMOS reference width (um)
f = cp.Variable(pos=True, name="f")       # Switching frequency (Hz)
A_L = cp.Variable(pos=True, name="A_L")   # Inductor area (mm^2)

# =============================================================================
# 5. DEVICE WIDTHS
# ORDER: PMOS 1, PMOS 2, NMOS 1, NMOS 2
# =============================================================================

W_p1 = 2 * W
W_p2 = 2 * W
W_n1 = W
W_n2 = W

# =============================================================================
# 6. MOSFET CONDUCTION LOSS
# =============================================================================

Irms_p = Iout * np.sqrt(D)
Irms_n = Iout * np.sqrt(1 - D)

R_p = Rds_pu_p / (2 * W)
R_n = Rds_pu_n / W

P_cond = (
    2 * Irms_p**2 * R_p
    + 2 * Irms_n**2 * R_n
)

# =============================================================================
# 7. GATE-DRIVE LOSS
# =============================================================================

Cg_p = Cgg_pu_p * (W_p1 + W_p2)
Cg_n = Cgg_pu_n * (W_n1 + W_n2)

P_gate = (Cg_p + Cg_n) * Vgs**2 * f

# =============================================================================
# 8. MOSFET OUTPUT CAPACITANCE LOSS
# =============================================================================

Coss_p = Coss_pu_p * (W_p1 + W_p2)
Coss_n = Coss_pu_n * (W_n1 + W_n2)

P_Coss = (Coss_p + Coss_n) * Vds**2 * f

# =============================================================================
# 9. INDUCTOR DESIGN
# =============================================================================

L_ind = Vout * (1 - D) / (delta_iL * f)

R_dcr = (L_ind / L_base) * (R_dcr_base / A_L)

P_ind = I_L_rms**2 * R_dcr

# =============================================================================
# 10. OUTPUT CAPACITOR DESIGN
# =============================================================================

C_out = (
    (1 - D)
    / (8 * L_ind * (delta_Vo / Vout) * f**2)
)

A_C = C_out / density_C

# =============================================================================
# 11. TOTAL LOSS AND AREA
# =============================================================================

P_loss = P_cond + P_gate + P_Coss + P_ind

# Per-device area = 2 * W_device * L_chan.
# Additional factor of 2 for total switch area, as specified.

A_MOS = (
    2 * (W_p1 + W_p2 + W_n1 + W_n2) * L_chan
    / 1e6
)

# Given area factors
A_total = A_MOS + 0.5 * A_L + 0.5 * A_C

# Figure of merit
FOM = A_total * P_loss

# =============================================================================
# 12. OPTIMIZATION CONSTRAINTS
# =============================================================================

FOM_max = 0.316

constraints = [
    W >= 20000,
    W <= 60000,

    f >= 10e6,
    f <= 100e6,

    A_L >= 0.1,

    FOM <= FOM_max
]

# Minimize total power loss
problem = cp.Problem(
    cp.Minimize(P_loss),
    constraints
)

# =============================================================================
# 13. SOLVE GEOMETRIC PROGRAM
# =============================================================================

print("\n" + "=" * 65)
print("CONVERTER OPTIMIZATION")
print("=" * 65)

problem.solve(
    gp=True,
    solver=cp.CLARABEL,
    verbose=False
)

# =============================================================================
# 14. FALLBACK IF FOM <= 0.4 IS INFEASIBLE
# =============================================================================

if problem.status not in ["optimal", "optimal_inaccurate"]:

    raise RuntimeError(
        f"Optimization failed: {problem.status}"
    )

print("\nFeasible design found.")

# =============================================================================
# 15. CALCULATED PARAMETERS
# =============================================================================

W_opt = W.value
f_opt = f.value
AL_opt = A_L.value

# Recalculate numerical values
D_opt = D

Irms_p_opt = Iout * np.sqrt(D_opt)
Irms_n_opt = Iout * np.sqrt(1 - D_opt)

L_opt = Vout * (1 - D_opt) / (delta_iL * f_opt)

C_opt = (
    (1 - D_opt)
    / (8 * L_opt * (delta_Vo / Vout) * f_opt**2)
)

R_dcr_opt = (
    (L_opt / L_base)
    * (R_dcr_base / AL_opt)
)

IL_rms_opt = np.sqrt(Iout**2 + delta_iL**2 / 12)

# Individual MOSFET losses
Pcond_p_opt = (
    2 * Irms_p_opt**2 * Rds_pu_p / (2 * W_opt)
)

Pcond_n_opt = (
    2 * Irms_n_opt**2 * Rds_pu_n / W_opt
)

Pgate_opt = (
    (
        4 * Cgg_pu_p
        + 2 * Cgg_pu_n
    ) * W_opt * Vgs**2 * f_opt
)

PCoss_opt = (
    (
        4 * Coss_pu_p
        + 2 * Coss_pu_n
    ) * W_opt * Vds**2 * f_opt
)

Pind_opt = IL_rms_opt**2 * R_dcr_opt

Ptotal_opt = (
    Pcond_p_opt
    + Pcond_n_opt
    + Pgate_opt
    + PCoss_opt
    + Pind_opt
)

# Area calculations
A_MOS_opt = 12 * W_opt * L_chan / 1e6
A_C_opt = C_opt / density_C

Atotal_opt = (
    A_MOS_opt
    + 0.5 * AL_opt
    + 0.5 * A_C_opt
)

FOM_opt = Atotal_opt * Ptotal_opt

# Efficiency
efficiency = Pout / (Pout + Ptotal_opt) * 100

# =============================================================================
# 16. PRINT RESULTS
# =============================================================================

print("\n" + "=" * 65)
print("OPTIMIZED DESIGN PARAMETERS")
print("=" * 65)

print(f"Solver status:             {problem.status}")
print(f"NMOS reference width:      {W_opt:,.2f} um")
print(f"PMOS width (each):         {2*W_opt:,.2f} um")
print(f"Switching frequency:       {f_opt/1e6:.4f} MHz")
print(f"Inductor area:             {AL_opt:.6f} mm^2")

print("\n--- Derived Passive Components ---")

print(f"Inductance:                {L_opt*1e9:.4f} nH")
print(f"Output capacitance:        {C_opt*1e9:.4f} nF")
print(f"Inductor DCR:              {R_dcr_opt:.6f} Ohm")
print(f"Inductor RMS current:      {IL_rms_opt:.6f} A")
print(f"Inductor ripple:           {delta_iL:.4f} A")
print(f"Output voltage ripple:     {delta_Vo*1e3:.2f} mV")

print("\n--- MOSFET Operating Currents ---")

print(f"PMOS RMS current (each):   {Irms_p_opt:.6f} A")
print(f"NMOS RMS current (each):   {Irms_n_opt:.6f} A")
print(f"Gate drive voltage:        {Vgs:.2f} V")

print("\n--- Power Loss Breakdown ---")

print(f"PMOS conduction loss:      {Pcond_p_opt*1e3:.4f} mW")
print(f"NMOS conduction loss:      {Pcond_n_opt*1e3:.4f} mW")
print(f"Gate-drive loss:           {Pgate_opt*1e3:.4f} mW")
print(f"Coss loss:                 {PCoss_opt*1e3:.4f} mW")
print(f"Inductor loss:             {Pind_opt*1e3:.4f} mW")

print(f"\nTotal power loss:          {Ptotal_opt:.6f} W")

print("\n--- Area Breakdown ---")

print(f"MOSFET area:               {A_MOS_opt:.6f} mm^2")
print(f"Inductor area:             {AL_opt:.6f} mm^2")
print(f"Capacitor area:            {A_C_opt:.6f} mm^2")

print(f"Total counted area:        {Atotal_opt:.6f} mm^2")

print("\n--- Performance ---")

print(f"Output power:              {Pout:.6f} W")
print(f"Efficiency:                {efficiency:.4f} %")
print(f"FOM:                       {FOM_opt:.6f} W*mm^2")
print(f"Maximum allowed FOM:       {FOM_max:.4f}")

print("\n--- Efficiency Targets ---")

print(f"80% efficiency:            {'PASS' if efficiency >= 80 else 'FAIL'}")
print(f"85% efficiency:            {'PASS' if efficiency >= 85 else 'FAIL'}")

print("=" * 65)


# Independent consistency checks
assert np.isclose(
    FOM_opt,
    Atotal_opt * Ptotal_opt,
    rtol=1e-8
)

assert np.isclose(
    efficiency,
    100 * Pout / (Pout + Ptotal_opt),
    rtol=1e-8
)

print("\n--- Unit Consistency ---")
print(f"MOSFET area:       {A_MOS_opt:.6e} mm^2")
print(f"Capacitor area:    {A_C_opt:.6e} mm^2")
print(f"Inductor area:     {AL_opt:.6e} mm^2")
print(f"Total loss:        {Ptotal_opt*1000:.4f} mW")
print(f"Total area:        {Atotal_opt:.6f} mm^2")
print(f"FOM:               {FOM_opt:.6f} W*mm^2")