# Domain primer: wall-flow diesel particulate filters

This primer sets out the physical behaviour that the synthetic data generator imitates. It is written for engineers new to after-treatment and for anyone reviewing whether the simulator is plausible. Quantities marked [verify] are typical figures that should be confirmed against manufacturer data or the DieselNet Technology Guide before any real-world use.

## 1. What a DPF does

A diesel particulate filter removes particulate matter (PM) from engine exhaust so that the vehicle or machine meets its type-approval limits on PM mass and, for recent standards, particle number (PN). Heavy-duty on-road engines (Euro VI), non-road mobile machinery (EU Stage V) and many rail and marine engines are now fitted with filters. A healthy wall-flow filter captures the large majority of solid particles; commonly quoted filtration efficiencies are above 90 % by mass and higher by number [verify].

## 2. Construction

- **Wall-flow honeycomb.** The filter is a ceramic monolith with thousands of parallel square channels. Channels are plugged alternately at each end in a checkerboard pattern: an *inlet channel* is open at the inlet face and plugged at the outlet face; an *outlet channel* is the reverse. Exhaust entering an inlet channel can only leave by passing through the porous wall into a neighbouring outlet channel, which is where particles are trapped.
- **Face appearance.** Each face therefore shows a checkerboard of open (dark) channel mouths and plugged (cement-coloured) cells. The outlet face is the complement of the inlet face.
- **Materials.**
  - *Cordierite* (a magnesium aluminosilicate): low thermal expansion and lower cost, usually a single monolith; lower melting point (around 1 450 °C [verify]) makes it vulnerable to melting during uncontrolled regenerations.
  - *Silicon carbide (SiC)*: high thermal conductivity and temperature capability, but higher thermal expansion, so it is built from square *segments* bonded with a ceramic cement; the cement joints are visible on the faces.
  - Aluminium titanate is also used; it is not modelled here.
- **Cell density** is quoted in cells per square inch (cpsi). Filters commonly use roughly 100 to 300 cpsi [verify]; 200 cpsi is about 31 cells/cm². This repository keeps cpsi as the label because it is the industry convention, with SI equivalents in the data card.
- **Packaging.** The substrate is wrapped in a mat and canned in a steel shell; the outer ceramic layer is the *skin*.

## 3. Soot and ash

| | Soot | Ash |
|---|---|---|
| Origin | Incomplete combustion of fuel: elemental carbon plus adsorbed hydrocarbons | Incombustible residue, mainly from lubricating-oil additives (calcium, zinc, phosphorus, magnesium compounds), plus wear metals |
| Removable on the vehicle? | Yes, by regeneration (oxidation) | No; it accumulates over the filter's life |
| Where it sits | Cake layer on inlet-channel walls | Plugs at the rear of inlet channels and layers on walls |
| Effect | Raises backpressure, recoverable | Raises backpressure permanently, reduces storage volume; only removed by off-vehicle cleaning |

Weight gain over the nominal clean weight is therefore a proxy for soot plus ash (plus any glassy deposits), but cannot separate them.

## 4. Regeneration

- **Passive regeneration.** NO₂ produced over an upstream diesel oxidation catalyst oxidises soot continuously at moderate exhaust temperatures (roughly 250 to 450 °C [verify]). It works well on hot, steady duty cycles such as long-haul driving.
- **Active regeneration.** The engine controller raises exhaust temperature, typically by late fuel injection burnt over the oxidation catalyst or by a burner, to roughly 550 to 650 °C [verify] so that oxygen burns the soot. Each event costs fuel.
- **Uncontrolled regeneration.** If a heavily loaded filter ignites, or a regeneration is interrupted at idle (low flow, little heat removal), the exotherm can drive local temperatures far above normal, creating steep radial and axial temperature gradients.
- **Duty cycle matters.** Low-speed stop-start work (urban buses), lightly loaded gensets and some off-highway machines run cold and rely on active regeneration, so they load more soot and see more regenerations. Rail, marine and gensets accumulate many hours and, with them, oil-derived ash.

## 5. Failure modes and how they show

| Failure mode | Typical cause | Visible on faces? | Effect on flow test |
|---|---|---|---|
| Radial crack | Thermal gradient, mechanical shock | Often, as a dark line from centre towards the skin | Opens a bypass path, *lowers* backpressure |
| Ring-off crack | Radial temperature gradient during regeneration; common in cordierite | Often, as a circumferential line | Bypass, lowers backpressure |
| Internal crack | As above, not reaching a face | No; may cause soot on the outlet face | Bypass |
| Melting | Severe uncontrolled regeneration | Glazed, distorted, fused cells | Local blockage or bypass |
| Missing or failed plug | Erosion, thermal stress, manufacturing | Plugged cell appears open | Channel open at both ends, leaks |
| Face plugging | Oil or fuel contamination, coking, heavy soot | Deposits blocking channel mouths | Raises backpressure |
| Oil / coolant contamination | Turbo seal, injector or head-gasket faults | Stains; coolant leaves glassy deposits (glazing) | Glazing resists cleaning |
| Outlet-face soot | Soot passing *through* the filter | Black deposits on the outlet face | Symptom of leakage |
| Edge chip / skin damage | Handling, canning | Missing material at the periphery | May compromise the seal |
| Foreign debris | Upstream component fragments, mat fibre | Objects on the face | Usually removed by cleaning |

Two of these interact in a way that matters for inspection: **cracks and missing plugs lower the measured backpressure**, so a damaged, heavily loaded filter can look normal on a flow test alone. Equally, an internal crack is invisible on the faces and may only reveal itself through outlet-face soot and an unexpectedly low pressure drop for its weight gain. No single measurement is sufficient.

## 6. Why it matters: emissions, backpressure and fuel

- A filter with a crack, melt or missing plugs lets particles bypass the walls; in service this can breach emission limits and fail roadside or periodic inspection. Several European countries are introducing particle-number measurement at periodic inspection [verify].
- Ash and soot raise exhaust backpressure, which increases engine pumping work and fuel consumption and triggers more frequent active regenerations, each of which also consumes fuel.
- A cleaned or remanufactured filter returned to service must therefore restore its pressure drop close to the new-part specification *and* demonstrate filtration integrity.

## 7. The cleaning and remanufacture line (as modelled)

1. **Goods-in.** Unit identified, weighed, service history recorded where known (application, hours, regeneration count, maximum regeneration temperature from the engine controller log, previous cleans).
2. **Thermal clean.** A slow, controlled bake (several hours [verify]) oxidises soot, oil residues and volatile deposits.
3. **Pneumatic clean.** Reverse-flow pulses of compressed air dislodge ash from the inlet channels.
4. **Downstream tests.**
   - *Flow restriction*: pressure drop at a reference flow on a flow bench, expressed here as a ratio to the new-part specification.
   - *Filtration integrity*: a test that reveals through-channels and cracks (for example light transmission through the substrate).
5. **Disposition.** PASS (return to service), REWORK (extended clean, then re-test) or SCRAP.

Because every unit currently gets the full clean before it is tested, units that end up scrapped have already consumed oven time, energy and labour. That is the waste this project targets.

## 8. Mapping to the synthetic generator

| Real phenomenon | Generator variable (latent unless stated) | Where it shows up |
|---|---|---|
| Duty cycle | `application` (observed) drives soot rate, ash rate, thermal stress, regeneration rate, contamination risk | Metadata |
| Ash accumulation | `ash_load_gL` grows with hours since last clean | Weight gain, flow curve, partly inlet-face plugging |
| Soot load | `soot_load_gL` | Weight gain, flow curve, inlet-face darkening |
| Thermal history | `max_regen_temp_true`, uncontrolled-regeneration events | Logged maximum temperature (often missing), cracks, melt |
| Cracks | visible radial or ring-off cracks; `internal_crack` | Images (visible only), lower ΔP, outlet-face soot |
| Plug loss | missing-plug cells per face | Images, backlight leaks, lower ΔP |
| Contamination | `oil_contaminated`, `coolant_contaminated`, `glazing` | Stains (partly), weight gain, poorer clean |
| Leakage | bypass fraction | Lower ΔP, outlet-face soot |
| Downstream outcome | post-clean ΔP ratio, integrity result, PASS/REWORK/SCRAP | Labels only, never model inputs |

## 9. Simplifications

- Faces are rendered with about 30 to 50 cells across instead of the 150 to 250 of a real filter, so cells resolve at laptop resolution. A real station needs either a high-resolution sensor or tiled inference over the face; see `docs/system_design.md` for the resolution budget.
- The backlight model treats each face independently: a missing plug transmits light only in the view where it is defined.
- Physics is replaced by documented, monotone surrogate relationships; parameters are chosen for plausibility, not fitted to field data.

## References

- DieselNet Technology Guide, sections on diesel particulate filters, filter materials, regeneration and ash (dieselnet.com). [verify section titles and dates before citing specific figures]
