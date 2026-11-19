from pathlib import Path

from sdge_microgrid.data.synthetic import generate_year_traces, save_processed

if __name__ == "__main__":
    out = Path("data/processed")
    save_processed(out, generate_year_traces(seed=42))
    print(f"Wrote traces to {out}")
