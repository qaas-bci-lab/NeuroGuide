"""Isolated command-line worker. The Qt process consumes EVENT JSON lines."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def emit(kind, value):
    print("NEUROGUIDE_EVENT " + json.dumps({"kind": kind, "value": value}, ensure_ascii=False), flush=True)


def main(config_path=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    config_path = config_path or sys.argv[1]
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    try:
        action = config["action"]
        progress = lambda message: emit("progress", message)
        timing = lambda record: emit("timing",record)
        from core.runtime import timed_stage
        if action == "inspect":
            from core.individual_cache import find_reusable_task
            result = find_reusable_task(config, progress)
        elif action == "predict":
            from core.individual_pipeline import run_pipeline
            result = run_pipeline(config, progress, timing=timing)
        elif action == "atlas":
            from core.atlas import register_atlas
            with timed_stage('atlas_registration',timing,'cpu') as duration:
                result = register_atlas(config, progress)
            result['run_timing']=duration
        else:
            stage='field_synthesis' if action=='synthesize' else 'electrode_optimization'
            with timed_stage(stage,timing,'cpu') as duration:
                if config.get('stimulation_type') == 'tES':
                    from core.tes import synthesize, optimize
                    if action == 'synthesize':
                        result = synthesize(config['prediction_dir'], config['output_dir'], config['electrode_labels'],
                            config['currents'], config.get('roi'), progress, **config.get('constraints', {}))
                    elif action == 'optimize':
                        result = optimize(config['prediction_dir'], config['output_dir'], config['roi'], config['options'], progress)
                    else:
                        raise ValueError(f'Unknown worker action: {action}')
                elif action == "synthesize":
                    from core.ti import synthesize
                    result = synthesize(config["prediction_dir"], config["output_dir"], config["pairs"],
                                        config["currents"], config.get("frequencies"), config.get("roi"), progress)
                elif action == "optimize":
                    from core.ti import optimize
                    result = optimize(config["prediction_dir"], config["output_dir"], config["roi"],
                                      config["options"], config.get("frequencies"), progress)
                else:
                    raise ValueError(f"Unknown worker action: {action}")
            result['run_timing']=duration
            if result.get('result_path'):
                from core.individual_pipeline import save_json
                save_json(result['result_path'],result)
        emit("result", result)
        return 0
    except Exception as exc:
        emit("error", str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
