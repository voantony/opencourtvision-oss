"""Vamos app engine API (v1): the layer the Mac / iPhone apps call.

    python -m opencourtvision.app analyze VIDEO --work DIR [--final "6-1 6-2"] [--mode match|rallies] [--labels FILE]
    python -m opencourtvision.app review DIR [--n 12]            # least-certain calls for the fix screen
    python -m opencourtvision.app fix DIR --item ID --value V    # one fix (confirm / correct); stored as a labelled example
    python -m opencourtvision.app taps DIR --taps watch.json     # Apple Watch score taps -> weak labels
    python -m opencourtvision.app render DIR --aspect 9:16 [--preset vamos_clean] --out reel.mp4
    python -m opencourtvision.app retrain DIR                    # personal adapter retrain + eval gate (on device)
    python -m opencourtvision.app share DIR --on|--off|--delete  # opt-in sharing / delete my data
    python -m opencourtvision.app tiers                          # free vs pro feature map (no payments)

Every command prints one JSON object on stdout (the app parses it); progress goes to stderr. Nothing here needs an
account or the network: import, auto-edit, score and fixes are free-tier and fully on device.
"""
SESSION_SCHEMA = "vamos.session.v1"
