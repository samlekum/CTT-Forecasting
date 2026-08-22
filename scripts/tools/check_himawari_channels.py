from pathlib import Path
import argparse
import xarray as xr


def inspect_file(nc_file: Path):
    print("=" * 80)
    print(f"FILE: {nc_file}")
    print("=" * 80)

    try:
        ds = xr.open_dataset(nc_file)

        print("\n[VARIABLES]")
        for name, var in ds.variables.items():
            print(f"  - {name}: shape={var.shape}, dtype={var.dtype}")

        print("\n[DIMENSIONS]")
        for name, size in ds.sizes.items():
            print(f"  - {name}: {size}")

        print("\n[GLOBAL ATTRIBUTES]")
        for key, value in ds.attrs.items():
            print(f"  - {key}: {value}")

        print("\n[CHANNEL-RELATED ATTRIBUTES]")

        channel_keywords = [
            "channel",
            "band",
            "wavelength",
            "central_wavelength",
            "resolution",
            "band_id",
        ]

        found = False

        for key, value in ds.attrs.items():
            key_lower = key.lower()

            if any(keyword in key_lower for keyword in channel_keywords):
                print(f"  - {key}: {value}")
                found = True

        for var_name, var in ds.variables.items():
            for key, value in var.attrs.items():
                key_lower = key.lower()

                if any(keyword in key_lower for keyword in channel_keywords):
                    print(f"  - {var_name}.{key}: {value}")
                    found = True

        if not found:
            print("  Tidak menemukan attribute channel/band yang eksplisit.")

        ds.close()

    except Exception as e:
        print(f"\nERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect channel information from Himawari-09 NetCDF files."
    )

    parser.add_argument(
        "path",
        nargs="?",
        default="data_bandung/jma/netcdf",
        help="File .nc atau folder yang berisi file .nc"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan folder secara recursive"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Jumlah file maksimum yang dicek"
    )

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        files = [path]

    elif path.is_dir():
        if args.recursive:
            files = sorted(path.rglob("*.nc"))
        else:
            files = sorted(path.glob("*.nc"))

    else:
        print(f"Path tidak ditemukan: {path}")
        return

    if not files:
        print(f"Tidak ada file .nc ditemukan di: {path}")
        return

    print(f"Ditemukan {len(files)} file .nc")

    if args.limit > 0:
        files = files[:args.limit]

    print(f"Mengecek {len(files)} file...\n")

    for nc_file in files:
        inspect_file(nc_file)


if __name__ == "__main__":
    main()