import os
import sys
import json
import numpy as np
import pandas as pd

from bowl_display_engine import SuperBowl


def parse_eyetracker_params(argv):
    params = {}
    for item in argv:
        if "|" in item:
            key, value = item.split("|", 1)
            params[key.strip()] = value.strip()
    return params


def as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def prompt_experiment_setup():
    """Ask wave type and repetition counts BEFORE Phase 1 starts.

    Returns
    -------
    wave_types   : list[str] – any subset of ["square", "sine"]
    dir_repeats  : int       – repetitions for direction-tuning block
    edge_repeats : int       – repetitions for edge-stimuli block
    """
    print()
    print("  Stimulus parameters (enter before Phase 1 begins):")
    print()

    print("  Wave type for S(Q)WEdge stimuli (Phase 2):")
    print("    [1]  square   (SQWEdges)")
    print("    [2]  sine     (SWEdges)")
    print("    [3]  both     (SQWEdges + SWEdges)")
    while True:
        choice = input("  Enter 1 / 2 / 3 [default: 3]: ").strip()
        if choice == "" or choice == "3":
            wave_types = ["square", "sine"]
            break
        elif choice == "1":
            wave_types = ["square"]
            break
        elif choice == "2":
            wave_types = ["sine"]
            break
        else:
            print("  *** Please enter 1, 2, or 3.")

    print()
    while True:
        raw = input("  Repetitions for direction tuning [default: 2]: ").strip()
        if raw == "":
            dir_repeats = 2
            break
        try:
            dir_repeats = int(raw)
            if dir_repeats >= 1:
                break
            print("  *** Please enter a positive integer.")
        except ValueError:
            print("  *** Please enter a whole number, e.g. 2 or 3.")

    while True:
        raw = input("  Repetitions for edge stimuli   [default: 2]: ").strip()
        if raw == "":
            edge_repeats = 2
            break
        try:
            edge_repeats = int(raw)
            if edge_repeats >= 1:
                break
            print("  *** Please enter a positive integer.")
        except ValueError:
            print("  *** Please enter a whole number, e.g. 2 or 3.")

    print(f"  → wave_types={wave_types},  dir_repeats={dir_repeats},  edge_repeats={edge_repeats}")
    return wave_types, dir_repeats, edge_repeats


def prompt_pd_only():
    """Ask only for PD after direction tuning is complete.

    Returns
    -------
    pd_deg : float – preferred direction in degrees (0–360)
    nd_deg : float – null direction (pd_deg + 180, wrapped)
    """
    print("\n" + "=" * 60)
    print("  DIRECTION TUNING COMPLETE")
    print("=" * 60)
    print("  Recording is still ACTIVE.")
    print()
    while True:
        raw = input("  Enter PD (preferred direction) in degrees [0–360]: ").strip()
        try:
            pd_deg = float(raw) % 360.0
            break
        except ValueError:
            print("  *** Please enter a number, e.g. 45 or 225.5")
    nd_deg = (pd_deg + 180.0) % 360.0
    print(f"  PD = {pd_deg:.1f} deg   →   ND = {nd_deg:.1f} deg (auto)")
    print("=" * 60 + "\n")
    return pd_deg, nd_deg


def prompt_pd_and_wave():
    """Combined prompt used by run_stimulus (standalone mode).

    Returns
    -------
    pd_deg, nd_deg, wave_types, dir_repeats, edge_repeats
    """
    wave_types, dir_repeats, edge_repeats = prompt_experiment_setup()
    pd_deg, nd_deg = prompt_pd_only()
    return pd_deg, nd_deg, wave_types, dir_repeats, edge_repeats


def build_direction_tuning_scenes(sb, cfg, dir_repeats_override=None):
    """Build only the Phase-1 direction-tuning scenes (PDNDgrating)."""
    # ── shared grating parameters ────────────────────────────────────────
    wavelength_deg = float(cfg.get("WavelengthDeg",  30.0))
    speed_dps      = float(cfg.get("SpeedDegPerSec", 30.0))
    mean_lum       = float(cfg.get("MeanLuminance", 127.5))
    contrast       = float(cfg.get("Contrast",        1.0))
    dir_wave       = str(cfg.get("DirectionWave", "sine")).strip().lower()

    # ── direction-tuning parameters (mirror .sti2 PDNDgrating exactly) ──
    #   "start pause [sec]"    : 1.0  → static grating shown 1 s before motion
    #   "time of motion [sec]" : 2.0  → grating drifts 2 s per direction
    #   "pause time [sec]"     : 0.5  → grey gap between PD and ND motion
    start_pause    = float(cfg.get("StartPause",        1.0))  # [sec]
    motion_time    = float(cfg.get("MotionTime",        2.0))  # [sec]
    pause_time     = float(cfg.get("PauseTime",         0.5))  # [sec]
    dir_repeats    = dir_repeats_override if dir_repeats_override is not None else int(cfg.get("DirectionRepeats", 1))
    randomize_dirs = as_bool(cfg.get("RandomizeDirections", "True"), default=True)

    # 8 independent directions shown one at a time, randomized each repetition.
    # Timing per scene: 1 s static → 2 s drifting → 1 s static → 1 s grey
    all_directions = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    if randomize_dirs:
        np.random.shuffle(all_directions)

    scenes        = []
    protocol_rows = []

    for d in all_directions:
        static_grating = sb.grating_wave_2d(
            wavelength=wavelength_deg, speed_dps=0, direction_deg=d,
            t=0, wave_type=dir_wave, mean_lum=mean_lum, contrast=contrast,
        )
        scene_name = "DirTuning_{:03.0f}deg".format(d)
        scenes.append(sb.scene(
            name             = scene_name,
            start_delay      = 1.0,               # 1 s static grating before motion
            starting_screen  = static_grating,
            timer            = 2.0,               # 2 s drifting motion
            timer_iterations = 1,                 # one motion period per direction
            timer_break      = 1.0,               # 1 s static grating after motion
            layers_brk       = [static_grating],  # show grating, not grey, after motion
            stiCond_save     = "DirTuning",
            wl_save          = wavelength_deg,
            tF_save          = speed_dps / wavelength_deg,
            lum_save         = mean_lum,
            dir_save         = d,
            eye_save         = "both",
            update=lambda s, c, t, p, b, d=d: [
                sb.grating_wave_2d(
                    wavelength    = wavelength_deg,
                    speed_dps     = speed_dps,
                    direction_deg = d,
                    t             = t,
                    wave_type     = dir_wave,
                    mean_lum      = mean_lum,
                    contrast      = contrast,
                )
            ],
        ))
        protocol_rows.append({
            "phase"            : "direction_tuning",
            "scene_name"       : scene_name,
            "direction_deg"    : d,
            "wavelength_deg"   : wavelength_deg,
            "speed_deg_per_sec": speed_dps,
            "wave_type"        : dir_wave,
            "start_static_s"   : 1.0,
            "motion_time_s"    : 2.0,
            "end_static_s"     : 1.0,
        })

    return scenes, pd.DataFrame(protocol_rows)


def build_edge_scenes(sb, cfg, pd_deg, nd_deg, wave_types, edge_repeats_override=None):
    """Build Phase-2 S(Q)WEdge scenes given confirmed PD/ND and wave type list.

    Parameters
    ----------
    pd_deg              : float       Preferred direction in degrees.
    nd_deg              : float       Null direction in degrees (usually pd+180).
    wave_types          : list[str]   Any subset of ["square", "sine"].
    edge_repeats_override : int|None  If set, overrides EdgeRepeats from cfg.
    """
    wavelength_deg  = float(cfg.get("WavelengthDeg",  30.0))
    speed_dps       = float(cfg.get("SpeedDegPerSec", 30.0))
    mean_lum        = float(cfg.get("MeanLuminance", 127.5))
    contrast        = float(cfg.get("Contrast",        1.0))
    start_pause     = float(cfg.get("StartPause",       1.0))
    inter_trial     = float(cfg.get("EdgeInterTrial",   1.0))
    edge_repeats    = edge_repeats_override if edge_repeats_override is not None else int(cfg.get("EdgeRepeats", 2))
    randomize_edges = as_bool(cfg.get("RandomizeEdges", "True"), default=True)

    # 1 s = 30 deg / 30 deg·s⁻¹
    edge_timer = wavelength_deg / speed_dps

    # wave_types list maps to named stimulus classes
    wave_to_cond = {"square": "SQWEdges", "sine": "SWEdges"}

    edge_conditions = []
    for axis_label, axis_deg in [("PD", pd_deg), ("ND", nd_deg)]:
        for wt in wave_types:
            cond_name = wave_to_cond[wt]
            for polarity_label, close_value in [("ON", 255), ("OFF", 0)]:
                edge_conditions.append({
                    "axis_label"    : axis_label,
                    "axis_deg"      : float(axis_deg),
                    "cond_name"     : cond_name,
                    "wave_type"     : wt,
                    "polarity_label": polarity_label,
                    "close_value"   : close_value,
                })

    if randomize_edges:
        np.random.shuffle(edge_conditions)

    scenes        = []
    protocol_rows = []

    for rep in range(edge_repeats):
        for cond in edge_conditions:
            a_deg          = float(cond["axis_deg"])
            wave_type      = cond["wave_type"]
            close_value    = int(cond["close_value"])
            polarity_label = cond["polarity_label"]
            cond_name      = cond["cond_name"]
            axis_label     = cond["axis_label"]

            scene_name = "{cond}_{axis}_{pol}_rep{rep:02d}".format(
                cond=cond_name, axis=axis_label, pol=polarity_label, rep=rep,
            )

            # Static grating shown for start_pause seconds before curtain moves.
            # Identical to what SQWEdges/SWEdges renders at progress=0.
            static_edge_grating = sb.grating_wave_2d(
                wavelength=wavelength_deg, speed_dps=0, direction_deg=a_deg,
                t=0, wave_type=wave_type, mean_lum=mean_lum, contrast=contrast,
            )

            scenes.append(sb.scene(
                name             = scene_name,
                start_delay      = start_pause,        # 1 s static grating before curtain
                starting_screen  = static_edge_grating,
                timer            = edge_timer,         # 1 s curtain motion
                timer_iterations = 1,
                timer_break      = inter_trial,        # grey between trials
                layers_brk       = [sb.grey_screen(int(mean_lum))],
                stiCond_save     = "{}_{}".format(cond_name, polarity_label),
                wl_save          = wavelength_deg,
                tF_save          = speed_dps / wavelength_deg,
                lum_save         = mean_lum,
                dir_save         = a_deg,
                pha_save         = polarity_label,
                eye_save         = "both",
                update=lambda s, c, t, p, b,
                              a_deg=a_deg, close_value=close_value,
                              cond_name=cond_name: [
                    sb.SQWEdges(
                        progress=p, axis_deg=a_deg, close_value=close_value,
                        wavelength=wavelength_deg, mean_lum=mean_lum, contrast=contrast,
                    ) if cond_name == "SQWEdges" else
                    sb.SWEdges(
                        progress=p, axis_deg=a_deg, close_value=close_value,
                        wavelength=wavelength_deg, mean_lum=mean_lum, contrast=contrast,
                    )
                ],
            ))

            protocol_rows.append({
                "phase"               : cond_name,
                "scene_name"          : scene_name,
                "stimulus_type"       : cond_name,
                "axis_label"          : axis_label,
                "axis_deg"            : a_deg,
                "wave_type"           : wave_type,
                "polarity"            : polarity_label,
                "close_value"         : close_value,
                "wavelength_deg"      : wavelength_deg,
                "speed_deg_per_sec"   : speed_dps,
                "closure_deg_per_side": wavelength_deg,
                "start_pause_s"       : start_pause,
                "motion_time_s"       : edge_timer,
                "inter_trial_s"       : inter_trial,
                "repetition"          : rep,
            })

    return scenes, pd.DataFrame(protocol_rows)


def build_protocol_scenes(sb, cfg):
    """Convenience wrapper: build all scenes without interactive prompts.

    Used by MultiDeviceSyncExemple.py where PD_Deg / ND_Deg are already
    known and passed as command-line parameters.
    """
    pd_deg     = float(cfg.get("PD_Deg",  0.0))
    nd_deg     = float(cfg.get("ND_Deg", 180.0))
    # Accept comma-separated list: "square", "sine", or "square,sine"
    wt_raw     = str(cfg.get("EdgeWaveTypes", "square,sine")).strip()
    wave_types = [w.strip().lower() for w in wt_raw.split(",") if w.strip()]

    dir_scenes, dir_rows  = build_direction_tuning_scenes(sb, cfg)
    edge_scenes, edge_rows = build_edge_scenes(sb, cfg, pd_deg, nd_deg, wave_types)
    all_scenes = dir_scenes + edge_scenes
    all_rows   = pd.concat([dir_rows, edge_rows], ignore_index=True)
    return all_scenes, all_rows


def run_stimulus(eyetracker_param):
    fly_id = eyetracker_param.get("FlyID", "DefaultFlyID")
    output_directory_path = eyetracker_param.get("OutputPath", r"C:\Users\fenklab\Desktop")
    os.makedirs(output_directory_path, exist_ok=True)

    xfov     = int(  eyetracker_param.get("XFOV",    180))
    yfov     = int(  eyetracker_param.get("YFOV",    120))
    fov_ele  = int(  eyetracker_param.get("FOVEle",   20))
    margin_x = int(  eyetracker_param.get("MarginX", 270))
    margin_y = int(  eyetracker_param.get("MarginY",   0))

    sb = SuperBowl(
        screen_type="HalfBowl",
        xfov=xfov,
        yfov=yfov,
        fov_ele=fov_ele,
        margin=[margin_x, margin_y],
    )

    photodiode_trigger = sb.rectangle(
        80, 80, color=255, color_b=0,
        offset_x=-(xfov - 20) / 2,
        offset_y=-(yfov - 20) / 2,
    )

    common_loop_kwargs = dict(
        iteration          = 1,
        random_order       = False,
        EyeTracker_linked  = bool(eyetracker_param),
        start_delays       = float(eyetracker_param.get("StartDelay",     1.0)),
        timer_breaks       = float(eyetracker_param.get("InterTrialBreak", 1.0)),
        starting_screens   = photodiode_trigger,
        starting_screens_flat = True,
    )

    # ── Phase 1: direction tuning ────────────────────────────────────────
    dir_scenes, dir_rows = build_direction_tuning_scenes(sb, eyetracker_param)
    dir_log = sb.loop_scenes(scenes=dir_scenes, **common_loop_kwargs)

    # ── Interactive prompt: user decides PD, ND, wave type, repetitions ──
    # If PD_Deg is already provided as a parameter (e.g. from EyeTracker
    # launcher), skip the prompt so the script can run unattended.
    if "PD_Deg" in eyetracker_param:
        pd_deg     = float(eyetracker_param["PD_Deg"])
        nd_deg     = float(eyetracker_param.get("ND_Deg", (pd_deg + 180.0) % 360.0))
        wt_raw     = str(eyetracker_param.get("EdgeWaveTypes", "square,sine")).strip()
        wave_types = [w.strip().lower() for w in wt_raw.split(",") if w.strip()]
        dir_repeats  = int(eyetracker_param.get("DirectionRepeats", 2))
        edge_repeats = int(eyetracker_param.get("EdgeRepeats", 2))
        print(f"  Using PD={pd_deg:.1f}\u00b0, ND={nd_deg:.1f}\u00b0, wave_types={wave_types}, "
              f"dir_repeats={dir_repeats}, edge_repeats={edge_repeats} from parameters.")
    else:
        pd_deg, nd_deg, wave_types, dir_repeats, edge_repeats = prompt_pd_and_wave()

    # Store the confirmed PD/ND in the param dict so it appears in the saved JSON
    eyetracker_param["PD_Deg_confirmed"]       = pd_deg
    eyetracker_param["ND_Deg_confirmed"]       = nd_deg
    eyetracker_param["EdgeWaveTypes_confirmed"] = ",".join(wave_types)
    eyetracker_param["DirectionRepeats_used"]  = dir_repeats
    eyetracker_param["EdgeRepeats_used"]       = edge_repeats

    # ── Phase 2: S(Q)WEdge stimuli ───────────────────────────────────────
    edge_scenes, edge_rows = build_edge_scenes(
        sb, eyetracker_param, pd_deg, nd_deg, wave_types,
        edge_repeats_override=edge_repeats,
    )
    edge_log = sb.loop_scenes(scenes=edge_scenes, **common_loop_kwargs)

    # ── Save all outputs ─────────────────────────────────────────────────
    full_log      = pd.concat([dir_log,  edge_log],  ignore_index=True)
    full_protocol = pd.concat([dir_rows, edge_rows], ignore_index=True)

    stimulus_csv = os.path.join(output_directory_path, fly_id + "_stimulus_log.csv")
    protocol_csv = os.path.join(output_directory_path, fly_id + "_stimulus_protocol.csv")
    params_json  = os.path.join(output_directory_path, fly_id + "_stimulus_params.json")

    full_log.to_csv(stimulus_csv, index=False)
    full_protocol.to_csv(protocol_csv, index=False)
    with open(params_json, "w", encoding="utf-8") as f:
        json.dump(eyetracker_param, f, indent=2)

    print("Saved stimulus log to:", stimulus_csv)
    print("Saved protocol table to:", protocol_csv)
    print("Saved run params to:", params_json)


if __name__ == "__main__":
    EyeTracker_param = parse_eyetracker_params(sys.argv)
    run_stimulus(EyeTracker_param)
