"""
bot.py - Unified AI Stock Analysis Bot.

One entry point for everything:
    python bot.py scan                          # scan market for top picks
    python bot.py scan --tickers AAPL MSFT      # scan custom tickers
    python bot.py analyze RELIANCE.NS           # deep backtest with financial metrics
    python bot.py train RELIANCE.NS             # train model on ticker
    python bot.py advise RELIANCE.NS            # get today's advisory
"""

import argparse
import sys

from config import Config


def _banner():
    print()
    print("  ============================================")
    print("     LSTM STOCK ANALYSIS BOT")
    print("     Powered by LSTM + Attention + yfinance")
    print("  ============================================")
    print()


def cmd_scan(args, cfg):
    """Scan the market for top stocks and multibagger candidates."""
    from screener import scan_market
    _banner()
    tickers = args.tickers if args.tickers else None
    scan_market(tickers=tickers, cfg=cfg, top_n=args.top)


def cmd_analyze(args, cfg):
    """Deep analysis with real financial metrics (Sharpe, Sortino, etc.)."""
    from backtest import backtest
    _banner()
    ticker = args.ticker or cfg.DEFAULT_TICKER
    backtest(cfg, ticker=ticker, period=args.period, threshold=args.threshold)


def cmd_train(args, cfg):
    """Train the LSTM model on a specific ticker."""
    from train import train
    _banner()
    ticker = args.ticker or cfg.DEFAULT_TICKER
    train(cfg, ticker=ticker, period=args.period, epochs=args.epochs)


def cmd_advise(args, cfg):
    """Get today's signal with advisory (asks permission before action)."""
    from advisor import advise
    _banner()
    ticker = args.ticker or cfg.DEFAULT_TICKER
    advise(cfg, ticker=ticker)


def main():
    parser = argparse.ArgumentParser(
        description="AI Stock Analysis Bot",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python bot.py scan                          Scan NIFTY 50 stocks
  python bot.py scan --tickers TCS.NS INFY.NS Scan specific stocks
  python bot.py analyze RELIANCE.NS           Backtest with financial metrics
  python bot.py train RELIANCE.NS             Train model on Reliance
  python bot.py train RELIANCE.NS --epochs 50 Train with custom epochs
  python bot.py advise RELIANCE.NS            Get today's advisory
        """,
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # scan
    p_scan = sub.add_parser("scan", help="Scan market for top stocks")
    p_scan.add_argument("--tickers", nargs="+", default=None)
    p_scan.add_argument("--top", type=int, default=15)
    p_scan.set_defaults(func=cmd_scan)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Deep analysis with financial metrics")
    p_analyze.add_argument("ticker", nargs="?", default=None)
    p_analyze.add_argument("--period", type=str, default=Config.DATA_PERIOD)
    p_analyze.add_argument("--threshold", type=float, default=None)
    p_analyze.set_defaults(func=cmd_analyze)

    # train
    p_train = sub.add_parser("train", help="Train model on ticker")
    p_train.add_argument("ticker", nargs="?", default=None)
    p_train.add_argument("--period", type=str, default=Config.DATA_PERIOD)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.set_defaults(func=cmd_train)

    # advise
    p_advise = sub.add_parser("advise", help="Get today's advisory signal")
    p_advise.add_argument("ticker", nargs="?", default=None)
    p_advise.set_defaults(func=cmd_advise)

    args = parser.parse_args()
    cfg = Config()

    if args.command is None:
        parser.print_help()
        print("\n  Run 'python bot.py <command> --help' for command-specific help.\n")
        sys.exit(0)

    args.func(args, cfg)


if __name__ == "__main__":
    main()
