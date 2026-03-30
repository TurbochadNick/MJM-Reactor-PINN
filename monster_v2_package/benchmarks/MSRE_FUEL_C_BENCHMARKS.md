# MSRE Fuel C Benchmarks

This file consolidates the `ORNL-TM-730` values that matter most for the v2 finite-cylinder diffusion PINN.

## Primary Use

Use these values for:

- target validation
- sanity checks on data generation
- loss calibration near the reference operating point
- acceptance criteria for the v2 package

## Reference Case

- Source: `ORNL-TM-730`, Part III, Nuclear Analysis
- Fuel: `Fuel C`
- Salt composition: `65 LiF - 29.2 BeF2 - 5 ZrF4 - 0.8 UF4` mol%
- Uranium isotopics: `35% U-235`, `64.4% U-238`, with trace `U-234` and `U-236`
- Density at `1200 degF`: `142.7 lb/ft^3`

## Geometry And Core Model

From `ORNL-TM-730`, delayed-neutron and diffusion calculations used:

- Physical radius used in delayed-neutron nonleakage model: `R = 27.75 in`
- Physical height used in delayed-neutron nonleakage model: `H = 68.9 in`
- Fuel volume within those boundaries: `25.0 ft^3`
- One-region lattice reference for XS generation: `22.5 vol% fuel`, `77.5 vol% graphite`

Useful derived interpretation:

- Bare-cylinder buckling form:
  - `Bg^2 = Br^2 + Bz^2`
  - `Br^2 = (2.405 / Re)^2`
  - `Bz^2 = (pi / He)^2`

## Reactivity Coefficients

Source: `Sec. 3.7`, `Table 3.4`, `Table 3.5`

Fuel C temperature coefficients:

| Quantity | ORNL units | SI-style conversion |
| --- | ---: | ---: |
| Fuel/salt coefficient | `-3.28e-5 /degF` | `-5.904 pcm/K` |
| Graphite coefficient | `-3.68e-5 /degF` | `-6.624 pcm/K` |
| Total coefficient | `-6.96e-5 /degF` | `-12.528 pcm/K` |

Important interpretation:

- The v1 solver result around `-53 pcm/K` is far too negative relative to this benchmark.
- Fuel C is the main validation target for `alpha_T`.

## Density Reactivity

Source: `Sec. 3.8`, `Table 3.5`

Fuel C density-related coefficients:

| Quantity | Value |
| --- | ---: |
| Fuel salt density coefficient | `0.182` |
| Graphite density coefficient | `0.767` |

`Sec. 3.8` also gives the shrinkage relation:

- `delta k / k = beta_s * (delta N_s / N_s) + beta_g * (delta N_g / N_g)`
- For shrinkage-induced salt displacement:
  - `delta k / k = beta_s * (v_g / v_s) * f1`
  - `v_s / v_g = 0.225 / 0.775`

ORNL text takeaway:

- `1%` graphite shrinkage corresponds to about:
  - `0.65% delta-k/k` in Fuels A and C
  - `1.2% delta-k/k` in Fuel B

## Flux Benchmarks

Source: `Table 3.5`, `Figs. 3.5-3.13`

Fuel C thermal fluxes at operating concentration, `10 MW`:

| Quantity | Value |
| --- | ---: |
| Maximum thermal flux | `3.29e13 n/cm^2/s` |
| Average in graphite-moderated regions | `1.42e13 n/cm^2/s` |
| Average in circulating fuel | `3.98e12 n/cm^2/s` |

Field-shape references available from the report:

- axial two-group flux shape
- radial two-group flux shape
- high-energy radial and axial flux shape
- radial and axial fission-density shape
- average flux spectrum in the largest core region

These figures should be used as qualitative or semi-quantitative field-shape anchors for the PINN.

## Neutron Balance

Source: `Table 3.6`, Fuel C clean critical, per `1e5` neutrons produced

Absorptions:

- `U-235`: `51,323`
- `U-238`: `8,967`
- Salt constituents other than uranium: `5,536`
- Graphite: `796`
- INOR: `9,682`
- Total absorptions: `76,304`

Leakage:

- Top: `2,001`
- Sides: `20,623`
- Bottom: `1,072`
- Total leakage: `23,696`

Useful derived fractions:

- Total leakage fraction: `0.23696`
- Side leakage fraction: `0.20623`

This is one of the strongest reasons buckling and leakage must be explicit in v2.

## Delayed Neutrons

Source: `Sec. 6.2-6.3`, `Table 6.1`, `Table 6.2`

### Group Data

| Group | Half-life (s) | `1e4 * beta_i` | Mean energy (MeV) | Age in MSRE (`cm^2`) |
| --- | ---: | ---: | ---: | ---: |
| 1 | `55.7` | `2.11` | `0.25` | `256` |
| 2 | `22.7` | `14.02` | `0.46` | `266` |
| 3 | `6.22` | `12.54` | `0.40` | `264` |
| 4 | `2.30` | `25.28` | `0.45` | `266` |
| 5 | `0.61` | `7.40` | `0.52` | `269` |
| 6 | `0.23` | `2.70` | `0.50` | `268` |

### Residence Times And Geometry Used

- Circulation rate: `1200 gpm`
- Core residence time: `9.37 s`
- External-loop residence time: `16.45 s`
- Thermal diffusion length used: `L^2 = 210 cm^2`

### Effective Delayed Neutron Fractions

From `Table 6.2` and ORNL summary text:

- Total precursor yield: `0.00641`
- Static effective delayed neutron fraction: `0.00666`
- Circulating effective delayed neutron fraction: `0.00362`
- Ratio `beta_eff_circulating / beta_eff_static ~= 0.5435`

Circulating effective group fractions (`1e4 * beta_i*`):

- `[0.52, 3.73, 4.99, 16.98, 7.18, 2.77]`

Static effective group fractions (`1e4 * beta_i*`):

- `[2.23, 14.57, 13.07, 26.28, 7.66, 2.80]`

These values should be supervised directly in v2 rather than inferred only from a simple heuristic.

## Kinetics

Source: `Table 3.5`

Fuel C prompt neutron lifetime:

- `2.40e-4 s`

## Power Coefficient Of Reactivity

Source: `Sec. 5.3`, `Table 5.3`

Fuel C power coefficients:

| Mode of control | ORNL units | Converted |
| --- | ---: | ---: |
| Constant `T_out` | `-0.006 % delta-k/k per MW` | `-6 pcm/MW` |
| Constant mean of `T_in` and `T_out` | `-0.024 % delta-k/k per MW` | `-24 pcm/MW` |
| Hands-off | `0` by definition | `0` |

Fuel C temperatures at `10 MW`, assuming isothermal `1200 degF` at zero power:

| Quantity | Value (`degF`) |
| --- | ---: |
| `T_out` | `1191` |
| `T_in` | `1141` |
| `T_f*` | `1177` |
| `T_g*` | `1221` |

## Control Rod Worth

Source: `Sec. 4.3`, `Table 4.1`, `Fig. 4.2`

| Fuel | Configuration | Worth (`% delta-k/k`) |
| --- | --- | ---: |
| A | 3 rods in | `5.6` |
| B | 3 rods in | `7.6` |
| C | 3 rods in | `5.7` |

Fuel A individual rod examples were also tabulated for shadowing effects.

The partially inserted rod-bank curve should be treated as a reference shape for future control-worth extensions, not as a direct field-PINN target in MVP.

## Cross-Section Workflow Context

Source: `Sec. 3.4`

ORNL workflow notes:

- Fast groups generated with `GAM-1`
- Thermal-group treatment generated with `THERMOS`
- Resonance effects and shielding were corrected for the channelized lattice
- Dancoff effects reduced the effective surface-to-volume ratio for resonance capture by about `30%`
- `Li-6`, `Li-7`, and `F-19` required special handling because of library limitations

This matters because the current v1 cross-section model is much cruder than the ORNL workflow.

## What To Hold The V2 Model Against

At minimum, v2 should recover Fuel C trends and magnitudes for:

- `k_eff` and near-critical behavior
- `alpha_T`
- `beta_eff`
- thermal flux scale and field shape
- leakage-dominated finite-cylinder behavior
- prompt lifetime order of magnitude
- power coefficient sign and rough magnitude
