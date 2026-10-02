"""Command-line entrypoint: doctor / decide / train / bench / serve."""
import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="clefforge",
                                     description="Multimodal decision-model training & inference framework")
    sub = parser.add_subparsers(dest="cmd")

    p_doc = sub.add_parser("doctor", help="check components / backends / benchmarks")
    p_doc.add_argument("--config", default=None)

    p_dec = sub.add_parser("decide", help="run a single decision")
    p_dec.add_argument("query")
    p_dec.add_argument("--options", nargs="+", required=True)
    p_dec.add_argument("--image", default=None)
    p_dec.add_argument("--config", default=None)

    p_tr = sub.add_parser("train", help="RL finetune the pointer head on synthetic data")
    p_tr.add_argument("--config", default=None)
    p_tr.add_argument("--steps", type=int, default=None)

    p_bench = sub.add_parser("bench", help="run the decision benchmarks")
    p_bench.add_argument("--config", default=None)

    p_serve = sub.add_parser("serve", help="start the Jev-API-compatible HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--config", default=None)

    args = parser.parse_args(argv)
    from ..config import load_config
    cfg = load_config(args.config)

    if args.cmd == "doctor":
        from ..doctor import run_doctor
        ok = run_doctor(cfg)
        sys.exit(0 if ok else 1)
    elif args.cmd == "decide":
        from ..engine import ClefEngine
        eng = ClefEngine(cfg)
        res = eng.decide(args.query, list(args.options), image=args.image)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "train":
        from ..train.finetune import run_finetune
        run_finetune(cfg, steps=args.steps)
    elif args.cmd == "bench":
        from ..bench.runner import run_all
        run_all(cfg)
    elif args.cmd == "serve":
        from ..serving.server import serve
        serve(cfg, args.host, args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
