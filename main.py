import argparse

from pipeline.core import run_pipeline


def main(use_ensemble=True, use_kelly=True, universe_limit=None):
    result = run_pipeline(
        use_ensemble=use_ensemble,
        use_kelly=use_kelly,
        universe_limit=universe_limit,
        verbose=True,
    )

    portfolio = result.portfolio
    print("\n" + "=" * 50)
    print("FINAL CONSTRUCTED PORTFOLIO (Target Allocations)")
    print("=" * 50)
    print(portfolio)
    print("=" * 50)

    gross_exposure = portfolio['Weight'].abs().sum()
    net_exposure = portfolio['Weight'].sum()
    print(f"Gross Exposure: {gross_exposure:.2%}")
    print(f"Net Exposure: {net_exposure:.2%}")
    print(f"Cash Drag: {1.0 - gross_exposure:.2%}")

    if result.risk_adjustments:
        adj = result.risk_adjustments
        print(f"\nRisk: VIX={adj.get('vix', 0):.1f} ({adj.get('vol_regime', 'n/a')}), "
              f"recommended exposure={adj.get('exposure_multiplier', 1):.0%}")
    if result.should_halt:
        print(f"[WARNING] Trading halt recommended: {result.halt_reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos research pipeline")
    parser.add_argument('--no-ensemble', action='store_true', help='Disable multi-model ensemble')
    parser.add_argument('--no-kelly', action='store_true', help='Use mean-variance instead of Kelly')
    parser.add_argument('--universe-limit', type=int, default=None, help='Cap tickers after filtering')
    args = parser.parse_args()

    main(
        use_ensemble=not args.no_ensemble,
        use_kelly=not args.no_kelly,
        universe_limit=args.universe_limit,
    )
