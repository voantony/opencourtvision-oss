"""ocv: umbrella command. `ocv eval ...`, `ocv swingcut ...`, `ocv match-start ...`, `ocv reel ...`, `ocv sv-audit ...`, `ocv highlights ...`, `ocv pose ...`, `ocv stages`."""
import sys

COMMANDS = {"eval": "eval", "swingcut": "swingcut", "match-start": "match_start", "reel": "reel", "sv-audit": "opencourtvision.importers.swingvision",
            "highlights": "opencourtvision.story.highlight", "pose": "pose"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip()); print("\ncommands: " + ", ".join(list(COMMANDS) + ["stages"])); return
    cmd, rest = argv[0], argv[1:]
    if cmd == "stages":
        from ..stages import STAGES
        print("\n".join(STAGES)); return
    if cmd not in COMMANDS:
        sys.exit(f"ocv: unknown command {cmd!r}; try `ocv --help`")
    import importlib
    mod = COMMANDS[cmd]
    importlib.import_module(mod if "." in mod else f"opencourtvision.cli.{mod}").main(rest)


if __name__ == "__main__":
    main()
