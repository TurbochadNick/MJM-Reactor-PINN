#!/usr/bin/env python3
"""MJM v1 — Publication-quality result plots."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json, sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PLOT_DIR = REPO_ROOT / 'artifacts' / 'legacy_v1_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from mjm_solver_v1 import (evaluate_design, find_critical_radius,
    compute_temperature_coefficient, FLiBeProperties, effective_delayed_fraction)

# ── Style ──
plt.rcParams.update({
    'figure.facecolor':'#0C0F14','axes.facecolor':'#12161E',
    'axes.edgecolor':'#2A2F3A','axes.labelcolor':'#C9D1D9',
    'text.color':'#C9D1D9','xtick.color':'#7C8493','ytick.color':'#7C8493',
    'grid.color':'#1C2028','grid.alpha':0.5,'font.family':'monospace','font.size':9,
})
CR='#DC143C'; CY='#00D4FF'; GD='#FFB800'; GR='#58A6FF'; WH='#E6EDF3'; DM='#7C8493'

# ── Run baseline at critical dimensions ──
print("Computing critical design...")
crit = evaluate_design(enrichment=0.1975, uf4_mol_frac=0.04,
    radius=23.6, height=47.2, temperature=900.0, water_vol_frac=0.0,
    nr=40, nz=60, verbose=False)

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: 6-PANEL NEUTRONICS DASHBOARD
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
fig.suptitle('MJM Molten-Salt SMR — 10 MW(th) Neutronic Design\n'
    f'FLiBe + 4% UF4 | 19.75% HALEU | R={23.6}cm H={47.2}cm | 900K | '
    f'k_eff={crit["k_eff"]:.5f}',
    fontsize=13, fontweight='bold', color=WH, y=0.98)

r=crit['r']; z=crit['z']; R,Z=np.meshgrid(r,z,indexing='ij')

# P1: Fast flux
ax=fig.add_subplot(gs[0,0])
p1=crit['phi1']/np.max(crit['phi1'])
im=ax.pcolormesh(R,Z,p1,cmap='inferno',shading='gouraud')
ax.set_xlabel('r (cm)'); ax.set_ylabel('z (cm)')
ax.set_title('Fast Flux φ₁(r,z)',color=CY,fontsize=11)
plt.colorbar(im,ax=ax,label='φ₁/φ₁_max',shrink=0.8)

# P2: Thermal flux
ax=fig.add_subplot(gs[0,1])
p2=crit['phi2']/np.max(crit['phi2'])
im=ax.pcolormesh(R,Z,p2,cmap='plasma',shading='gouraud')
ax.set_xlabel('r (cm)'); ax.set_ylabel('z (cm)')
ax.set_title('Thermal Flux φ₂(r,z)',color=GD,fontsize=11)
plt.colorbar(im,ax=ax,label='φ₂/φ₂_max',shrink=0.8)

# P3: Radial profiles at midplane
ax=fig.add_subplot(gs[0,2])
mz=crit['phi1'].shape[1]//2
ax.plot(r,crit['phi1'][:,mz]/np.max(crit['phi1'][:,mz]),color=CY,lw=2,label='Fast')
ax.plot(r,crit['phi2'][:,mz]/np.max(crit['phi2'][:,mz]),color=GD,lw=2,label='Thermal')
from scipy.special import j0
ax.plot(r,j0(2.405*r/23.6)/j0(0),'--',color=DM,lw=1,label='J₀ ref')
ax.set_xlabel('r (cm)'); ax.set_ylabel('Normalised Flux')
ax.set_title('Radial Profiles (midplane)',color=WH,fontsize=11)
ax.legend(fontsize=8); ax.grid(True)

# P4: Temperature coefficient
ax=fig.add_subplot(gs[1,0])
temps=np.linspace(773,1050,15)
k_t=[]
for T in temps:
    res=evaluate_design(radius=23.6,height=47.2,temperature=T,nr=22,nz=33)
    k_t.append(res.get('k_eff',np.nan))
k_t=np.array(k_t)
ax.plot(temps,k_t,'o-',color=CY,ms=3,lw=2)
ax.axhline(1.0,color=CR,ls='--',alpha=0.7,label='k=1')
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('k_eff')
ax.set_title('Temperature Reactivity',color=CY,fontsize=11)
ax.legend(fontsize=8); ax.grid(True)

# Compute alpha on twin axis
ax2=ax.twinx()
dk=np.gradient(k_t,temps)
alpha=dk/k_t*1e5
ax2.plot(temps,alpha,'s-',color=CR,ms=2,lw=1.5)
ax2.set_ylabel('α_T (pcm/K)',color=CR)

# P5: Enrichment sweep
ax=fig.add_subplot(gs[1,1])
enrs=np.linspace(0.03,0.25,12)
k_e=[]
for e in enrs:
    res=evaluate_design(enrichment=e,radius=23.6,height=47.2,nr=22,nz=33)
    k_e.append(res.get('k_eff',np.nan))
ax.plot(enrs*100,k_e,'o-',color=GR,ms=4,lw=2)
ax.axhline(1.0,color=CR,ls='--',alpha=0.7)
ax.axvline(19.75,color=GD,ls=':',alpha=0.7,label='HALEU limit')
ax.axvline(5.0,color=DM,ls=':',alpha=0.5,label='LEU limit')
ax.set_xlabel('Enrichment (%)'); ax.set_ylabel('k_eff')
ax.set_title('Enrichment Sensitivity',color=GR,fontsize=11)
ax.legend(fontsize=8); ax.grid(True)

# P6: Convergence
ax=fig.add_subplot(gs[1,2])
hist=crit['history']
iters=range(len(hist['k']))
ax.semilogy(list(iters),hist['res'],'o-',color=CR,ms=2,lw=1.5)
ax.axhline(1e-6,color=DM,ls='--',label='tol=1e-6')
ax.set_xlabel('Iteration'); ax.set_ylabel('|Δk/k|')
ax.set_title('Convergence',color=CR,fontsize=11)
ax.legend(fontsize=8); ax.grid(True)

dashboard_path = PLOT_DIR / 'mjm_v1_dashboard.png'
plt.savefig(dashboard_path,dpi=150,bbox_inches='tight')
print(f"Saved: {dashboard_path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: DESIGN SPACE MAP
# ═══════════════════════════════════════════════════════════════
print("Computing design space heatmap...")
fig2,ax=plt.subplots(figsize=(10,7))
enr_r=np.linspace(0.03,0.25,14)
rad_r=np.linspace(15,60,14)
kmap=np.zeros((len(enr_r),len(rad_r)))
for i,e in enumerate(enr_r):
    for j,rad in enumerate(rad_r):
        res=evaluate_design(enrichment=e,radius=rad,height=2*rad,nr=18,nz=27)
        kmap[i,j]=res.get('k_eff',np.nan)

RAD,ENR=np.meshgrid(rad_r,enr_r*100)
im=ax.pcolormesh(RAD,ENR,kmap,cmap='RdYlGn',shading='gouraud',vmin=0.4,vmax=2.0)
try:
    cs=ax.contour(RAD,ENR,kmap,levels=[1.0,1.03],colors=[WH,'#FFD700'],linewidths=[2,1.5])
    ax.clabel(cs,fmt={1.0:'k=1.0',1.03:'k=1.03'},fontsize=9,colors=WH)
except: pass
ax.plot(23.6,19.75,'*',color=CR,ms=15,zorder=10,label='MJM Design Point')
ax.set_xlabel('Core Radius (cm)',fontsize=12)
ax.set_ylabel('Enrichment (%)',fontsize=12)
ax.set_title('MJM Design Space: k_eff(Enrichment, Radius)\n'
    '10 MW(th) | FLiBe + 4% UF4 | H=2R | 900K',fontsize=12,fontweight='bold',color=WH)
plt.colorbar(im,ax=ax,label='k_eff')
ax.axhline(19.75,color=GD,ls=':',alpha=0.5,label='HALEU limit')
ax.axhline(5.0,color=DM,ls=':',alpha=0.5,label='LEU limit')
ax.legend(fontsize=9,loc='lower right')
ax.grid(True,alpha=0.3)
plt.tight_layout()
design_space_path = PLOT_DIR / 'mjm_v1_design_space.png'
plt.savefig(design_space_path,dpi=150,bbox_inches='tight')
print(f"Saved: {design_space_path}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: FLiBe PROPERTY CURVES (Beneš & Konings)
# ═══════════════════════════════════════════════════════════════
fig3,axes=plt.subplots(1,3,figsize=(15,4.5))
fig3.suptitle('FLiBe (66-34 mol%) Thermophysical Properties — Beneš & Konings (2009)',
    fontsize=11,fontweight='bold',color=WH)

T=np.linspace(750,1100,100)
axes[0].plot(T,[FLiBeProperties.density(t) for t in T],color=CY,lw=2)
axes[0].set_xlabel('T (K)'); axes[0].set_ylabel('ρ (kg/m³)')
axes[0].set_title('Density',color=CY); axes[0].grid(True)

axes[1].plot(T,[FLiBeProperties.viscosity(t) for t in T],color=GD,lw=2)
axes[1].set_xlabel('T (K)'); axes[1].set_ylabel('η (mPa·s)')
axes[1].set_title('Viscosity',color=GD); axes[1].grid(True)

axes[2].semilogy(T,[FLiBeProperties.vapor_pressure(t) for t in T],color=CR,lw=2)
axes[2].set_xlabel('T (K)'); axes[2].set_ylabel('p (Pa)')
axes[2].set_title('Vapor Pressure',color=CR); axes[2].grid(True)

plt.tight_layout(rect=[0,0,1,0.92])
flibe_path = PLOT_DIR / 'mjm_v1_flibe_props.png'
plt.savefig(flibe_path,dpi=150,bbox_inches='tight')
print(f"Saved: {flibe_path}")

print("\nAll plots complete.")
