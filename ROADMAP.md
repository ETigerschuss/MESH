# Roadmap — from overlap analysis to a closed-loop optomotor model

This connects the current connectomics viewer (VS/HS/MOS/MOT meshes + gap-junction
network) to the motion-vision model in `ALIstimAXET_Final3.ipynb` (a PyTorch HRBL
detector with T4/T5, LPi motion-opponency, and gradient-based stimulus
optimisation). The four requested extensions, with concrete architecture and the
main risks.

## What already exists (reuse, don't rebuild)

- **Motion front-end** (notebook): `HRBL` implements T4/T5 (ON/OFF × two
  directions) with LPi (Lpi34/Lpi43) motion-opponent inhibition, and a VS unit
  with excitatory/inhibitory X-weights. `SineGrating` renders drifting gratings.
- **Stimulus optimisation** (notebook): `ExperimentOne` drives one detector to
  max/min; `ExperimentTwo` finds the stimulus that *maximally separates* two
  models (contrastive loss + output decorrelation via `corrcoef`), by gradient
  descent on the stimulus itself. This is already exactly the "drive to extremes /
  biggest difference" engine.
- **LPTC↔motor network** (viewer, Tier-1/Tier-2 tabs): a biophysical gap-junction
  model over VS/HS/MOS/MOT with per-pair conductances derived from the (now
  confined) contact areas. It has the wiring but **no visual drive**.

The gap to close is: **connect the motion front-end's output to the LPTC dendrites,
run it through the existing GJ network, read out MOT/MOS as a motor/gaze command,
and make the whole chain differentiable so the stimulus optimiser can act on the
network, not just a single detector.**

---

## 1) T4/T5/LPi models → VS/HS by cardinal direction & layer

**Biology to encode.** T4/T5 come in 4 subtypes, one per lobula-plate layer, each
tuned to one cardinal direction (L1 front-to-back, L2 back-to-front, L3 upward,
L4 downward — check against the current FlyWire layer convention). LPi cells
provide the null-direction inhibition between opponent layers. HS (horizontal
cells) read layers 1/2; VS (vertical cells) read layers 3/4. So the mapping is
**layer → preferred direction → which LPTC dendrite it feeds**.

**Recommended architecture.**
- Do **not** load ~800 individual T4/T5 meshes into the viewer — it won't render
  and adds little. Instead model T4/T5 as a *retinotopic input field*: one HRBL
  detector per ommatidial column, tiled over each LPTC's receptive field on a hex
  lattice (the notebook already sketches the hex grid with `hexalattice`).
- Weight each column's contribution to a given VS/HS cell by the **real
  connectome**: pull T4/T5→VS and T4/T5→HS synapse counts from the FlyWire synapse
  table for these cells (same CAVE client you already use). That grounds the
  receptive-field shape and gain in data rather than assumption.
- In the viewer, represent this as a per-LPTC "input layer" overlay (a colored
  field on the dendrite) rather than individual cells — optionally load a handful
  of example T4/T5 for illustration.

**Risk.** Getting the layer→direction→LPTC sign right is the crux; validate the
assembled VS/HS direction tuning against published tuning curves before trusting
downstream results.

## 2) Biophysical network sim + eye-movement readout

**Plan.** Chain three modules with clean interfaces:

```
stimulus(t) ─▶ HRBL(T4/T5/LPi) per column ─▶ Σ connectome weights ─▶ I_VS/HS(t)
           ─▶ GJ network (VS·HS·MOS·MOT)  ─▶ V_MOT/MOS(t) ─▶ gaze integrator ─▶ θ(t)
```

- **Input coupling.** Feed the HRBL output as the dendritic input current to each
  VS/HS compartment in the existing GJ model. Start with the single-compartment
  Tier-1 model; only go multi-compartment (Tier-2) if the axon/dendrite voltage
  split matters for the motor readout.
- **Motor → gaze.** MOT/MOS are neck motor / descending; map their output to neck-
  muscle activation → head/eye angular velocity, integrate to gaze angle θ(t).
  A first-order model (θ̇ ∝ weighted MOT/MOS output, with a time constant) is
  enough to reproduce the optomotor response.
- **Closed loop (optional, powerful).** If the stimulus is defined in world
  coordinates, feed θ(t) back so the retinal image is stabilised by the animal's
  own movement — this is the actual active-vision loop (ties directly to the
  Eristalis active-vision proposal).

**Implementation.** Implement the GJ network as an unrolled ODE in PyTorch
(explicit Euler at the 1/120 s stimulus dt, or `torchdiffeq` for stiffness). Keep
it differentiable end-to-end — that's what unlocks #3.

## 3) Auto-generate extreme / discriminative stimuli for the whole network

`ExperimentTwo` already does this for single detectors; the extension is to make
the **loss read out the network/gaze**, not one HRBL unit:
- *Extreme*: maximise |θ(t)| or a chosen VS/HS/MOT response over a window.
- *Discriminative*: maximise |readout(model_A) − readout(model_B)| where A/B differ
  in a hypothesis (e.g. GJ present vs absent between MOS↔VS, or a layer-sign flip),
  plus the existing decorrelation term so the two models' *time courses* diverge.
- Because the whole chain (module 2) is differentiable, backprop flows from the
  gaze loss to the pixel stimulus — same optimiser, bigger graph.

**Payoff.** These optimised stimuli are the ones to run in the eye-tracking rig:
they are, by construction, where two competing circuit hypotheses disagree most —
maximum experimental power per trial. Keep the notebook's grid-search over loss
weights, but move it to the network readout.

## 4) EM-based automated gap-junction detection

**Framing.** Three classes at each membrane apposition: *chemical synapse*,
*gap junction*, *mere overlap*. You already have two of the three labelled:
- **Chemical (strong labels):** FlyWire-annotated pre/post synapse sites (T-bars)
  between HS and VS — known chemical, thousands of them.
- **Mere overlap (strong negatives):** appositions with no T-bar nearby and not in
  a coupled-axon strip.
- **Gap junction (weak labels):** the long parallel HS↔VS and VS↔VS *axonal*
  overlap strips that physiology says are electrically coupled — use as
  weak/positive GJ examples. (This is exactly the axonal-strip signal you noticed.)

**Model.** Center a small EM subvolume on each candidate patch (you already
generate these stacks). Train a CNN — 2.5-D (central slice + a few neighbours) is a
good start; go 3-D if GPU allows — to classify the three types. Train on the rich
HS/VS data, then **apply to the MOS/MOT↔LPTC contacts** and rank candidates; surface
the top-scoring sites in the viewer as putative GJs. The seg-adjacency mask
(`validated_patches.csv`) is a strong auxiliary input channel — it already tells the
net where the two membranes truly appose.

**Hard constraint — resolution.** The confirmed GJ ultrastructure (pentalaminar
membrane, ~2–4 nm cleft, the "thicker membrane" you saw) is **not resolvable in the
aligned 8 nm EM** we have; native 4 nm exists only in v14 space and is misaligned
with the v14.1 segmentation. So realistically:
- Use the CNN at 8 nm as a **triage/ranker** over apposition texture, membrane
  straightness/length, parallelism, and *absence* of vesicle clouds/T-bars — not as
  a final ultrastructural call.
- For the top-ranked MOS/MOT sites, pull **native 4 nm** crops for human/again-ML
  confirmation. Solving the 4 nm↔v14.1 alignment for small local patches (local
  rigid/affine registration rather than the global cross-correlation that failed
  earlier) is the enabling sub-project.

**Order of work.** (a) assemble the labelled HS/VS patch dataset from existing
stacks + synapse table; (b) train chemical-vs-overlap first (clean labels), then add
the GJ class; (c) rank MOS/MOT candidates; (d) local 4 nm alignment for
confirmation.

---

## Suggested sequencing

1. Wire the notebook HRBL front-end onto the LPTC dendrites and through the
   existing Tier-1 GJ network → first end-to-end VS/HS/MOT response to a grating
   (validates #1 + half of #2).
2. Add the gaze integrator + differentiable ODE → closed-loop optomotor (#2).
3. Point the existing stimulus optimiser at the gaze readout (#3).
4. In parallel, build the EM GJ classifier (#4) — it's independent of 1–3 and
   feeds better GJ priors back into the network conductances.
