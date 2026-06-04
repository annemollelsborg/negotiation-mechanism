import json
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python mitigation_shares.py <policy.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    levers = ["energy", "methane", "agriculture"]
    country_totals = {
        c: sum(sum(v["lever_effort_fraction"][l]) for l in levers)
        for c, v in data.items()
    }
    grand_total = sum(country_totals.values())

    print(f"{'Agent':<12} {'Total effort':>14} {'Share':>8}")
    print("-" * 36)
    for c, total in country_totals.items():
        print(f"{c:<12} {total:>14.2f} {total / grand_total:>8.2%}")
    print("-" * 36)
    print(f"{'TOTAL':<12} {grand_total:>14.2f} {'100.00%':>8}")


if __name__ == "__main__":
    main()
