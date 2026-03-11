from ucimlrepo import fetch_ucirepo
from pathlib import Path

# fetch dataset
wholesale_customers = fetch_ucirepo(id=292)
df = wholesale_customers.data.original
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "wholesale_customers.csv"
df.to_csv(output_path, index=False)
print("Dataset saved to:", output_path)