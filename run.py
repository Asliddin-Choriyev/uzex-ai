#!/usr/bin/env python3
"""
UZEX AI Agent — startup script
Usage:
    python run.py
    python run.py --port 8080
    python run.py --key sk-your-api-key-here
"""
import argparse
import os
import sys
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser(description="UZEX AI Agent Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--key", default="", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    args = parser.parse_args()

    if args.key:
        os.environ["OPENAI_API_KEY"] = args.key
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    # Pre-flight checks
    data_dir = os.path.join(os.path.dirname(__file__), "uzex_spot_data")
    required = ["deals_2022.csv", "deals_2023.csv", "contracts.csv", "brokers.csv", "participants.csv", "usd_rates.csv"]
    missing = [f for f in required if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print(f"❌ ERROR: Missing data files in uzex_spot_data/: {', '.join(missing)}")
        print("   Make sure all CSV files are in the uzex_spot_data/ folder.")
        sys.exit(1)

    print("=" * 60)
    print("  UZEX AI Agent — Birja Bitimlari Tahlili")
    print("=" * 60)
    print(f"  🌐 URL:   http://localhost:{args.port}")
    print(f"  🤖 Model: {args.model}")
    print(f"  🔑 Key:   {'set via --key / env' if os.environ.get('OPENAI_API_KEY') else 'NOT SET — configure in UI'}")
    print("=" * 60)
    print("  ⏳ Loading 500K deals into memory (this may take ~30s)...")
    print()

    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
