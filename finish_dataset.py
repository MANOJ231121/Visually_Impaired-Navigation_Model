import concurrent.futures
import os
import re
import sys
import time
import zipfile
from pathlib import Path
import requests

FILE_ID = "1NFMPXsI8RGnXomLqamtXp0iCXGoQZZks"
TOTAL_SIZE = 1505825181
NUM_PARTS = 16
OUTPUT_ZIP = Path("archive.zip")
PARTS_DIR = Path("download_parts")

def get_download_url():
    session = requests.Session()
    url = f"https://docs.google.com/uc?export=download&id={FILE_ID}"
    r = session.get(url)
    uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', r.text)
    if uuid_match:
        uuid = uuid_match.group(1)
        return f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t&uuid={uuid}"
    raise RuntimeError("Could not obtain download URL / UUID")

def download_subchunk(url, start, end, out_path):
    expected_size = end - start + 1
    if out_path.exists() and out_path.stat().st_size == expected_size:
        return expected_size
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            if r.status_code not in (200, 206):
                raise RuntimeError(f"Bad status: {r.status_code}")
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        f.write(chunk)
            if out_path.stat().st_size == expected_size:
                return expected_size
            time.sleep(1)
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2)
    raise RuntimeError(f"Failed subchunk {out_path}")

def main():
    chunk_size = (TOTAL_SIZE + NUM_PARTS - 1) // NUM_PARTS
    # Check which parts are missing
    missing_parts = []
    for i in range(NUM_PARTS):
        start = i * chunk_size
        end = min((i + 1) * chunk_size - 1, TOTAL_SIZE - 1)
        pf = PARTS_DIR / f"part_{i:02d}.bin"
        expected = end - start + 1
        if not pf.exists() or pf.stat().st_size != expected:
            missing_parts.append((i, start, end, pf, expected))

    print(f"Missing parts: {[p[0] for p in missing_parts]}")
    if missing_parts:
        dl_url = get_download_url()
        print("Obtained fresh download URL.")

        for (part_idx, part_start, part_end, part_file, part_expected) in missing_parts:
            print(f"Downloading part {part_idx} ({part_expected / (1024*1024):.1f} MB) with 8 sub-threads...")
            num_sub = 8
            sub_size = (part_expected + num_sub - 1) // num_sub
            sub_tasks = []
            sub_files = []
            for s_idx in range(num_sub):
                s_start = part_start + s_idx * sub_size
                s_end = min(part_start + (s_idx + 1) * sub_size - 1, part_end)
                s_file = PARTS_DIR / f"sub_{part_idx:02d}_{s_idx:02d}.bin"
                sub_tasks.append((dl_url, s_start, s_end, s_file))
                sub_files.append(s_file)

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_sub) as executor:
                futures = [executor.submit(download_subchunk, *t) for t in sub_tasks]
                for f in concurrent.futures.as_completed(futures):
                    f.result()

            print(f"Merging sub-chunks for part {part_idx}...")
            with open(part_file, "wb") as pf_out:
                for sf in sub_files:
                    with open(sf, "rb") as sf_in:
                        pf_out.write(sf_in.read())
                    sf.unlink()
            print(f"Part {part_idx} assembled: {part_file.stat().st_size} bytes (Expected: {part_expected})")
            assert part_file.stat().st_size == part_expected

    print("All 16 parts complete! Merging into archive.zip...")
    part_files = [PARTS_DIR / f"part_{i:02d}.bin" for i in range(NUM_PARTS)]
    with open(OUTPUT_ZIP, "wb") as outfile:
        for pf in part_files:
            with open(pf, "rb") as infile:
                while True:
                    b = infile.read(8 * 1024 * 1024)
                    if not b:
                        break
                    outfile.write(b)

    print(f"archive.zip ready: {OUTPUT_ZIP.stat().st_size} bytes.")
    assert OUTPUT_ZIP.stat().st_size == TOTAL_SIZE

    print("Cleaning up part files...")
    for pf in part_files:
        try:
            pf.unlink()
        except Exception:
            pass
    try:
        PARTS_DIR.rmdir()
    except Exception:
        pass

    print("Extracting archive.zip...")
    t0 = time.time()
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
        zf.extractall(".")
    print(f"Extraction finished in {time.time() - t0:.1f}s.")

    try:
        OUTPUT_ZIP.unlink()
        print("archive.zip deleted.")
    except Exception:
        pass
    print("Done!")

if __name__ == "__main__":
    main()
