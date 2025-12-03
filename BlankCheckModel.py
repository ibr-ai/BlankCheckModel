# The End of the Blank Check — U.S.–Israel Decoupling Model
# Copyright (c) 2025 Aidan Ibrahim — MIT License
# https://github.com/aidanibrahim/blank-check-2035

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm 

price_buffer = []    # global rolling window of last 12 months of prices

# First ODE function that solves ODE without the Tel Aviv Real Estate Nuke
def blank_check_dynamics(y, t, p):
    # State vector
    D, I, G, R, A, P, L = y  # Debt%GDP, Interest%budget, GenZ+M%, Restraint%, Aid$B, TA_price_index, Leverage%

    year = 2025 + t

    # === Level 1: Macro fiscal-demographic ===
    primary_def = np.interp(year, p['year'], p['primary_def'])      # CBO baseline
    risk_premium = 0.09 * np.tanh(3.3 * (D - 1.30))   # max +9 %, sharp at 130–180%
    yield_rate   = min(np.interp(year, p['year'], p['yield']) + risk_premium, 0.15)
    dD = primary_def + yield_rate * D
    dI = yield_rate * D * 10

    dG = 0.018  # Gen-Z/Millennial voting share grows ~1.8%/yr

    # Restraint contagion (fiscal rage × youth × media)
    media = 1 + 3.8 * np.tanh(I - 0.15)

    # 2. Restraint-right growth – realistic ceiling ~22–23 %
    dR = 0.019 * I * G * media * (1 - R/0.23)**1.5 * R * (1 - R)
    #   - 0.019 is the calibrated contagion rate
    #   - (1 - R/0.23)**1.5 gives strong diminishing returns above ~15 %
    #   - R*(1-R) is the classic logistic wrapper so it never overshoots

    # Veto power (your nuke) – decays with restraint + youth + interest pain
    veto_power = 0.97 * (1 - R/0.11) * (1 - max(0, G-0.36)/0.28)

    # Aid collapse
    collapse_pressure = max(0, (I-0.16) * np.maximum(0, R-0.18) * (1-veto_power) * 35)   # triggers at lower R and I
    collapse_pressure = min(collapse_pressure, 20.0)   # safety cap
    dA = -A * collapse_pressure

    # === Tel Aviv real-estate – now crashes hard and stays crashed ===
    security_risk = max(0.0, 0.35 - veto_power)                     # 0 → 0.35 range
    capital_outflow = 0.18 * security_risk + 0.012 * max(0.0, R - 0.12) # foreigners + locals flee

    # Update rolling price buffer (12 months = 120 steps at dt=0.1)
    price_buffer.append(P)
    if len(price_buffer) > 120:
        price_buffer.pop(0)

    # Forced-selling panic — only activates when prices have been falling
    forced_selling_pressure = 0.0
    if len(price_buffer) == 120:
        past_price = price_buffer[0]                     # price 12 months ago
        if past_price > P:                               # falling trend
            drop_fraction = (past_price - P) / past_price
            forced_selling_pressure = 0.28 * drop_fraction**2.3   # non-linear panic

    # Final price dynamics
    base_growth = 0.075 * (1 - 1.8 * security_risk)                 # max -63% annual when veto gone
    dP = P * (base_growth - capital_outflow - forced_selling_pressure)

    # Leverage ratchet
    mortgage_rate = yield_rate + 0.03
    dL = 4.5 * mortgage_rate - 2.8   # Israeli wages grow slower

    # === Real-estate veto nuke feedback (crash accelerator) ===
    if len(y_history):                              # crude crash detector
        if P < 0.75 * y_history[-24, 5]:           # >25% drop in 2 years
            collapse_pressure *= 3.8                # your nuke multiplier
            dA = -A * collapse_pressure

    return [dD, dI, dG, dR, dA, dP, dL]
    

# Second ODE function that uses results from the original ODE and incorporates the Tel Aviv Real Estate Nuke
# Must be done this way due to limitations with Scipy
def blank_check_dynamics_nuke(y, t, p, permanent_veto_multiplier):
    # State vector
    D, I, G, R, A, P, L = y  # Debt%GDP, Interest%budget, GenZ+M%, Restraint%, Aid$B, TA_price_index, Leverage%

    year = 2025 + t

    # === Level 1: Macro fiscal-demographic ===
    primary_def = np.interp(year, p['year'], p['primary_def'])      # CBO baseline
    risk_premium = 0.09 * np.tanh(3.3 * (D - 1.30))   # max +9 %, sharp at 130–180%
    yield_rate   = min(np.interp(year, p['year'], p['yield']) + risk_premium, 0.15)
    dD = primary_def + yield_rate * D
    dI = yield_rate * D * 10

    dG = 0.018  # Gen-Z/Millennial voting share grows ~1.8%/yr

    # Restraint contagion (fiscal rage × youth × media)
    media = 1 + 5.0 * np.tanh(I - 0.15)

    # 2. Restraint-right growth – realistic ceiling ~22–23 %
    dR = 0.019 * I * G * media * (1 - R/0.23)**1.5 * R * (1 - R)
    #   - 0.019 is the calibrated contagion rate
    #   - (1 - R/0.23)**1.5 gives strong diminishing returns above ~15 %
    #   - R*(1-R) is the classic logistic wrapper so it never overshoots

    # Veto power (your nuke) – decays with restraint + youth + interest pain
    veto_power = 0.97 * (1 - R/0.11) * (1 - max(0, G-0.36)/0.28)
    veto_power *= permanent_veto_multiplier

    # Aid collapse
    collapse_pressure = max(0, (I-0.16) * np.maximum(0, R-0.18) * (1-veto_power) * 35)   # triggers at lower R and I
    collapse_pressure = min(collapse_pressure, 20.0)   # safety cap
    dA = -A * collapse_pressure

    # === Tel Aviv real-estate – now crashes hard and stays crashed ===
    security_risk = max(0.0, 0.35 - veto_power)                     # 0 → 0.35 range
    capital_outflow = 0.18 * security_risk + 0.012 * max(0.0, R - 0.12) # foreigners + locals flee

    # Update rolling price buffer (12 months = 120 steps at dt=0.1)
    price_buffer.append(P)
    if len(price_buffer) > 120:
        price_buffer.pop(0)

    # Forced-selling panic — only activates when prices have been falling
    forced_selling_pressure = 0.0
    if len(price_buffer) == 120:
        past_price = price_buffer[0]                     # price 12 months ago
        if past_price > P:                               # falling trend
            drop_fraction = (past_price - P) / past_price
            forced_selling_pressure = 0.28 * drop_fraction**2.3   # non-linear panic

    # Final price dynamics
    base_growth = 0.075 * (1 - 1.8 * security_risk)                 # max -63% annual when veto gone
    dP = P * (base_growth - capital_outflow - forced_selling_pressure)

    # Leverage ratchet
    mortgage_rate = yield_rate + 0.03
    dL = 4.5 * mortgage_rate - 2.8   # Israeli wages grow slower

    # === Real-estate veto nuke feedback (crash accelerator) ===
    if len(y_history):                              # crude crash detector
        if P < 0.75 * y_history[-24, 5]:           # >25% drop in 2 years
            collapse_pressure *= 3.8                # your nuke multiplier
            dA = -A * collapse_pressure

    return [dD, dI, dG, dR, dA, dP, dL]
    
# Parameters (2025-realistic)
params = {
    'year': np.arange(2025, 2041),
    'primary_def': np.linspace(0.03, 0.045, 16),   # rising deficits
    'yield':     np.linspace(0.042, 0.058, 16),   # 10-yr Treasury
}

# Initial conditions 2025
y0 = [1.23, 0.16, 0.38, 0.06, 3.8, 232, 152]   # D, I, G, R, Aid, TA_index(2015=100), Leverage%
y_history = []                      # for crash detection
t  = np.linspace(0, 15, 1501)   

# Initiating the second ODE if the real estate nuke is triggered (price drop >32% in < 30 months)
# Step 1: Solve the system WITHOUT the nuke
sol = odeint(blank_check_dynamics, y0, t, args=(params,))# Step 2: Find the first year where Tel Aviv prices dropped >32% in <30 months
P = sol[:, 5]                              # Tel Aviv price index
drop_triggered = False
trigger_year_idx = None
for i in range(30, len(t)):
    if P[i-30] > 0 and (P[i-30] - P[i]) / P[i-30] > 0.32:
        trigger_year_idx = i
        drop_triggered = True
        break
    
# Step 3: If trigger found, re-run from that point with the nuke applied
if drop_triggered:
    print(f"Real-estate veto nuke fires in year {2025 + t[trigger_year_idx]:.1f}")
    
    # Apply one-time shock at trigger point
    y_nuke = sol[trigger_year_idx].copy()
    y_nuke[4] *= 0.4           # aid immediately cut by 60 %
    veto_shock = 0.15           # veto power permanently crippled

    # Re-solve from trigger onward with shocked state
    t_post = t[trigger_year_idx:]
    sol_post = odeint(blank_check_dynamics_nuke, y_nuke, t_post, 
                    args=(params, veto_shock))

    # Stitch the two parts together
    sol[trigger_year_idx:] = sol_post



# ========================================================
# MONTE CARLO SIMULATION — 15,000 RUNS
# Produces Figure 2 for the paper: "Collapse Window 2032–2036"
# ========================================================

N_SIMS = 15000          # 15,000 is the gold standard for publication
np.random.seed(42)      # reproducibility

# Storage: we only save the four key trajectories
aid_trajectories       = np.zeros((N_SIMS, len(t)))
restraint_trajectories = np.zeros((N_SIMS, len(t)))
yield_trajectories     = np.zeros((N_SIMS, len(t)))
collapse_years         = []

print("Running 15,000 Monte Carlo simulations...")

for sim in tqdm(range(N_SIMS)):
    # === Random parameter draws (realistic uncertainty ranges) ===
    # Initial conditions ±15–20%
    D0 = np.random.uniform(1.15, 1.33)   # 115–133% debt/GDP in 2025
    I0 = np.random.uniform(0.14, 0.19)    # interest burden
    G0 = np.random.uniform(0.35, 0.41)    # Gen-Z+M share
    R0 = np.random.uniform(0.04, 0.09)    # restraint-right base

    # Growth/contagion rates ±30%
    genz_growth     = np.random.uniform(0.014, 0.022)    # 1.4–2.2%/yr
    contagion_rate  = np.random.uniform(0.015, 0.025)    # 0.019 ±30%
    risk_steepness  = np.random.uniform(2.8,   3.8)      # 3.3 ±15%
    aid_threshold  = np.random.uniform(0.17, 0.21)       # when R really starts biting

    # Risk premium amplitude — we keep this tight because it's well-calibrated
    risk_max = np.random.uniform(0.085, 0.095)           # 8.5–9.5%

    # Aid collapse multiplier
    collapse_mult = np.random.uniform(32, 40)

    # === Run one simulation (copy of your current stable ODEs with params injected) ===
    def dynamics_mc(y, t):
        D, I, G, R, A, P, L = y
        year = 2025 + t

        primary_def = np.interp(year, params['year'], params['primary_def'])
        baseline_yield = np.interp(year, params['year'], params['yield'])

        # Risk premium
        risk_premium = risk_max * np.tanh(risk_steepness * (D - 1.30))
        yield_rate = min(baseline_yield + risk_premium, 0.15)

        dD = primary_def + yield_rate * D
        dI = yield_rate * D * 10
        dG = genz_growth

        # Political contagion
        media = np.clip(1 + 3.8 * np.tanh(I - 0.15), 0.1, 12.0)
        base_term = np.maximum(0.0, 1 - R / 0.23)
        ceiling = base_term ** 1.5
        dR = contagion_rate * I * G * media * ceiling * R * (1 - R)
        dR = np.clip(dR, -0.06, 0.06)

        # Veto power
        veto_R_term = np.maximum(0.0, 1 - R / 0.11)
        veto_power = 0.97 * veto_R_term * (1 - max(0, G - 0.36)/0.28)
        veto_power = np.clip(veto_power, 0.0, 1.0)

        # Aid collapse — only after restraint voters are serious
        collapse_pressure = max(0.0, (I-0.16) * max(0.0, R - aid_threshold) * (1-veto_power) * collapse_mult)
        collapse_pressure = np.clip(collapse_pressure, 0.0, 20.0)
        dA = -A * collapse_pressure

        # Tel Aviv prices (simplified but stable)
        security_risk = max(0.0, 0.35 - veto_power)
        capital_outflow = 0.18 * security_risk + 0.012 * max(0.0, R - 0.12)
        base_growth = 0.075 * (1 - 1.8 * security_risk)
        dP = P * (base_growth - capital_outflow - 0.15 * max(0, 0.3 - P/P)**2)  # mild panic term

        mortgage_rate = yield_rate + 0.03
        dL = 4.5 * mortgage_rate - 2.8

        return [dD, dI, dG, dR, dA, dP, dL]

    # Integrate
    sol_mc = odeint(dynamics_mc, [D0, I0, G0, R0, 3.8, 232, 152], t, mxstep=50000, rtol=1e-6)

    # Save trajectories
    aid_trajectories[sim] = sol_mc[:,4]
    restraint_trajectories[sim] = sol_mc[:,3] * 100
    yield_trajectories[sim] = np.minimum(
        np.interp(2025 + t, params['year'], params['yield']) + risk_max * np.tanh(risk_steepness * (sol_mc[:,0] - 1.30)),
        0.15) * 100

    # Record collapse year (aid < $0.8B)
    collapse_idx = np.where(sol_mc[:,4] < 0.8)[0]
    if len(collapse_idx) > 0:
        collapse_years.append(2025 + t[collapse_idx[0]])
    else:
        collapse_years.append(2045)  # never collapses

# ========================================================
# PLOT MONTE CARLO RESULTS — FIGURE 2 OF THE PAPER
# ========================================================

years = 2025 + t

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. U.S. Aid — uncertainty bands
p5, p50, p95 = np.percentile(aid_trajectories, [5, 50, 95], axis=0)
axs[0,0].fill_between(years, p5, p95, alpha=0.3, color='darkred')
axs[0,0].plot(years, p50, color='darkred', lw=3)
axs[0,0].set_title('U.S. Aid to Israel ($B real 2025) — 90% CI')
axs[0,0].axhline(1.0, color='black', ls='--', alpha=0.7)
axs[0,0].set_ylim(0, 4.5)

#  # 2. Restraint voters
p5, p50, p95 = np.percentile(restraint_trajectories, [5, 50, 95], axis=0)
axs[0,1].fill_between(years, p5, p95, alpha=0.3, color='purple')
axs[0,1].plot(years, p50, color='purple', lw=3)
axs[0,1].set_title('Restraint-Right Vote Share (%) — 90% CI')
axs[0,1].set_ylim(0, 25)

# 3. Borrowing cost
p5, p50, p95 = np.percentile(yield_trajectories, [5, 50, 95], axis=0)
axs[1,0].fill_between(years, p5, p95, alpha=0.3, color='orange')
axs[1,1].plot(years, p50, color='orange', lw=3)
axs[1,0].axhline(10, color='red', ls='--')
axs[1,0].axhline(14, color='darkred', ls='-')
axs[1,0].set_title('Effective U.S. Borrowing Cost (%) — 90% CI')
axs[1,0].set_ylim(3, 16)

# 4. Collapse year distribution
axs[1,1].hist(collapse_years, bins=range(2030, 2041), alpha=0.87, color='black', edgecolor='white')
axs[1,1].axvline(np.percentile(collapse_years, 50), color='red', lw=3, label=f'Median: {np.percentile(collapse_years, 50):.1f}')
axs[1,1].axvline(np.percentile(collapse_years, 5), color='red', ls='--')
axs[1,1].axvline(np.percentile(collapse_years, 95), color='red', ls='--')
axs[1,1].set_title('Year of Aid Collapse (<$0.8B)\n15,000 Simulations')
axs[1,1].set_xlim(2030, 2040)
axs[1,1].legend()

plt.tight_layout()
plt.show()

print(f"\nCollapse year statistics (15,000 runs):")
print(f"  5th percentile : {np.percentile(collapse_years, 5):.1f}")
print(f"  Median         : {np.percentile(collapse_years, 50):.1f}")
print(f"  95th percentile: {np.percentile(collapse_years, 95):.1f}")
