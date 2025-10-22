import os
import json
import random
from pathlib import Path

# ---------- CONFIG ----------
SOURCE_DIR = Path("./tobu_data")
OUTPUT_DIR = Path("./synthetic_users")
NUM_USERS = 20
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_valid_files(base_dir: Path):
    """Scan all subfolders for memory.json files that contain >10 reports."""
    valid = []
    for sub in base_dir.iterdir():
        mem_file = sub / "memory.json"
        if mem_file.exists():
            try:
                with open(mem_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if (
                    isinstance(data, dict)
                    and "reports" in data
                    and isinstance(data["reports"], list)
                    and len(data["reports"]) > 10
                ):
                    valid.append((sub.name, data))
            except Exception as e:
                print(f"[skip] {mem_file}: {e}")
    return valid


def reindex_memory_ids(data):
    """Replace id and memory_id with m01, m02, ..."""
    if "reports" not in data or not isinstance(data["reports"], list):
        return data
    for i, report in enumerate(data["reports"], start=1):
        new_id = f"m{i:02d}"
        report["id"] = new_id
        ctx = report.get("context", {})
        if isinstance(ctx, dict):
            ctx["memory_id"] = new_id
    return data


def main():
    valid_files = load_valid_files(SOURCE_DIR)
    print(f"✅ Found {len(valid_files)} valid memory.json files with >10 memories")

    if len(valid_files) == 0:
        print("❌ No files with more than 10 memories found.")
        return

    # select up to 20 random users from filtered list
    selected = random.sample(valid_files, min(NUM_USERS, len(valid_files)))
    start_index = 41

    for i, (user_folder, data) in enumerate(selected, start=start_index):
        data = reindex_memory_ids(data)
        out_path = OUTPUT_DIR / f"user_{i:03d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[done] {user_folder} → {out_path} ({len(data['reports'])} memories)")

    print("🎉 Extraction complete! Filtered & reindexed files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
