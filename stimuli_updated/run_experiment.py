import os
import sys
import time
import json
import pandas as pd

# Ensure bowl_display_engine.py and stimulus_protocol.py are always found,
# regardless of the working directory the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.append(r"C:\Users\fenklab\Desktop\Tools_Fenklab\Fictrac\python_tools")
sys.path.append(r"C:\Users\fenklab\Desktop\Tools_Fenklab\Arduino_Python_Controls")

from bowl_display_engine import SuperBowl
from ArduinoControl_class import ArduinoControl
from FictracListener_class import FictracListener
from stimulus_protocol import (
    build_direction_tuning_scenes,
    build_edge_scenes,
    prompt_experiment_setup,
    prompt_pd_only,
)


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


def main():
    eyetracker_param = parse_eyetracker_params(sys.argv)

    fly_id = eyetracker_param.get("FlyID", "DefaultFlyID")
    output_directory_path = eyetracker_param.get("OutputPath", r"C:\Users\fenklab\Desktop\TEST")
    os.makedirs(output_directory_path, exist_ok=True)

    photodiode_threshold = int(eyetracker_param.get("PhotodiodThreshold", 50))
    sync_pause_s = float(eyetracker_param.get("CameraSyncPauseSec", 3.0))

    opto_on_duration = float(eyetracker_param.get("OptoOn", 3000))
    opto_off_duration = float(eyetracker_param.get("OptoOff", 3000))
    opto_ramp_duration = float(eyetracker_param.get("OptoRampCycle", 1000))
    enable_opto = as_bool(eyetracker_param.get("EnableOpto", "False"), default=False)

    xfov = int(eyetracker_param.get("XFOV", 180))
    yfov = int(eyetracker_param.get("YFOV", 150))
    fov_ele = int(eyetracker_param.get("FOVEle", 15))
    margin_x = int(eyetracker_param.get("MarginX", 260))
    margin_y = int(eyetracker_param.get("MarginY", 0))

    arduino = None
    ft_listener = None

    try:
        arduino = ArduinoControl(debug=True)
        arduino.start()

        arduino.set_opto_state("Off")
        arduino.set_photodiod_active("Off")
        arduino.set_photodiod_threshold(photodiode_threshold)

        # Stop camera trigger briefly so first post-break frame can be used for synchronization.
        arduino.set_cameras_active(on=False)
        time.sleep(sync_pause_s)
        arduino.set_cameras_active(on=True)

        ft_listener = FictracListener(["right_shift", "forward_shift"], refresh_rate=100.0)
        ft_listener.start()

        if enable_opto:
            arduino.set_opto_cycle(
                on_duration=opto_on_duration,
                off_duration=opto_off_duration,
                ramp_duration=opto_ramp_duration,
            )
            arduino.set_opto_state("Cycle")
        else:
            arduino.set_opto_state("Off")

        arduino.set_photodiod_active("On")

        sb = SuperBowl(
            screen_type="HalfBowl",
            xfov=xfov,
            yfov=yfov,
            fov_ele=fov_ele,
            margin=[margin_x, margin_y],
        )

        photodiode_trigger = sb.rectangle(
            10,
            10,
            color=255,
            color_b=0,
            offset_x=-(xfov - 10) / 2,
            offset_y=-(yfov - 10) / 2,
        )

        common_loop_kwargs = dict(
            iteration=1,
            random_order=False,
            EyeTracker_linked=bool(eyetracker_param),
            start_delays=0.0,     # each scene defines its own start_delay
            timer_breaks=1.0,     # 1 s grey screen between scenes
            starting_screens=photodiode_trigger,
            starting_screens_flat=True,
        )

        # ── Recording readiness check + pre-experiment parameters ─────────────
        print("\n" + "=" * 60)
        print("  RECORDING STATUS")
        print("=" * 60)
        print(f"  Arduino   : {'CONNECTED' if arduino is not None else 'NOT CONNECTED'}")
        print(f"  FicTrac   : {'RUNNING' if ft_listener is not None else 'NOT RUNNING'}")
        print(f"  Photodiode: ACTIVE (threshold = {photodiode_threshold})")
        print(f"  Opto      : {'CYCLE' if enable_opto else 'OFF'}")
        print()
        print("  Confirm all recording streams are running")
        print("  (ephys software, cameras, FicTrac, Arduino serial log).")
        print()

        # Ask wave type and repetitions now, before Phase 1 runs.
        if "PD_Deg" in eyetracker_param:
            wt_raw = str(eyetracker_param.get("EdgeWaveTypes", "square,sine")).strip()
            wave_types   = [w.strip().lower() for w in wt_raw.split(",") if w.strip()]
            dir_repeats  = int(eyetracker_param.get("DirectionRepeats", 2))
            edge_repeats = int(eyetracker_param.get("EdgeRepeats", 2))
            print(f"  wave_types={wave_types}, dir_repeats={dir_repeats}, edge_repeats={edge_repeats} (from parameters)")
        else:
            wave_types, dir_repeats, edge_repeats = prompt_experiment_setup()

        print()
        input("  >>> Press ENTER when recording is confirmed and you are ready to begin Phase 1... ")
        print("=" * 60 + "\n")

        # Phase 1: 8 independent directions, randomized, repeated dir_repeats times.
        dir_scenes, dir_rows = build_direction_tuning_scenes(sb, eyetracker_param)
        dir_log = sb.loop_scenes(
            scenes=dir_scenes,
            **{**common_loop_kwargs, "iteration": dir_repeats, "random_order": True},
        )

        # After Phase 1 the experimenter identifies the preferred direction.
        if "PD_Deg" in eyetracker_param:
            pd_deg = float(eyetracker_param["PD_Deg"])
            nd_deg = float(eyetracker_param.get("ND_Deg", (pd_deg + 180.0) % 360.0))
            print(f"Using PD={pd_deg:.1f} deg, ND={nd_deg:.1f} deg from parameters.")
        else:
            pd_deg, nd_deg = prompt_pd_only()

        eyetracker_param["PD_Deg_confirmed"]       = pd_deg
        eyetracker_param["ND_Deg_confirmed"]       = nd_deg
        eyetracker_param["EdgeWaveTypes_confirmed"] = ",".join(wave_types)
        eyetracker_param["DirectionRepeats_used"]  = dir_repeats
        eyetracker_param["EdgeRepeats_used"]       = edge_repeats

        # Phase 2: S(Q)WEdges while recording remains active.
        edge_scenes, edge_rows = build_edge_scenes(
            sb, eyetracker_param, pd_deg, nd_deg, wave_types,
            edge_repeats_override=edge_repeats,
        )
        edge_log = sb.loop_scenes(scenes=edge_scenes, **common_loop_kwargs)

        stimulus_log = pd.concat([dir_log, edge_log], ignore_index=True)
        protocol_df = pd.concat([dir_rows, edge_rows], ignore_index=True)

        stimulus_csv = os.path.join(output_directory_path, fly_id + "_stimulus_log.csv")
        protocol_csv = os.path.join(output_directory_path, fly_id + "_stimulus_protocol.csv")
        params_json = os.path.join(output_directory_path, fly_id + "_run_params.json")

        stimulus_log.to_csv(stimulus_csv, index=False)
        protocol_df.to_csv(protocol_csv, index=False)
        with open(params_json, "w", encoding="utf-8") as f:
            json.dump(eyetracker_param, f, indent=2)

        print("Saved stimulus log to:", stimulus_csv)
        print("Saved protocol table to:", protocol_csv)
        print("Saved run params to:", params_json)

    finally:
        if arduino is not None:
            try:
                arduino.set_opto_state("Off")
                arduino.set_photodiod_active("Off")
            except Exception:
                pass

        if ft_listener is not None:
            try:
                ft_listener.close()
                fictrac_log = ft_listener.get_logs()
                fictrac_csv = os.path.join(output_directory_path, fly_id + "_fictrac_log.csv")
                fictrac_log.to_csv(fictrac_csv, index=False)
                print("Saved FicTrac log to:", fictrac_csv)
            except Exception as exc:
                print("Warning: failed to save FicTrac log:", str(exc))

        if arduino is not None:
            try:
                arduino.close()
                arduino_log = arduino.get_logs()
                arduino_csv = os.path.join(output_directory_path, fly_id + "_arduino_log.csv")
                arduino_log.to_csv(arduino_csv, index=False)
                print("Saved Arduino log to:", arduino_csv)
            except Exception as exc:
                print("Warning: failed to save Arduino log:", str(exc))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [!] Recording interrupted by user (Ctrl+C).")
        print("  Hardware safe-state and data saving were handled by the cleanup block above.")
        print("  Partial logs (if any) should have been saved to the output directory.\n")
