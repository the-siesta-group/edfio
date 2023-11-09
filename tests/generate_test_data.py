import urllib.request
from pathlib import Path

from tests import TEST_DATA_DIR


def shorten_remote_edf(url: str, target_path: Path, target_duration: int) -> None:
    with urllib.request.urlopen(url) as source_edf:
        total_len = source_edf.length
        source_header = source_edf.read(256)
        bytes_in_header_record = int(source_header[184:192])
        source_num_data_records = int(source_header[236:244])
        data_record_duration = float(source_header[244:252])

    body_len = total_len - bytes_in_header_record
    target_num_data_records = int(target_duration / data_record_duration)
    target_body_len = int(body_len * target_num_data_records / source_num_data_records)
    target_len = bytes_in_header_record + target_body_len

    target_header = source_header.replace(
        f"{source_num_data_records:<8}".encode("ascii"),
        f"{target_num_data_records:<8}".encode("ascii"),
    )

    with urllib.request.urlopen(url) as source_edf:
        raw_source_shortened = source_edf.read(target_len)
    raw_target_edf = target_header + raw_source_shortened[256:target_len]
    target_path.write_bytes(raw_target_edf)


if __name__ == "__main__":
    files = [
        (
            "https://www.physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf?download",
            "short_psg.edf",
            5 * 60,
        ),
    ]

    for source_url, target_name, target_duration in files:
        shorten_remote_edf(source_url, TEST_DATA_DIR / target_name, target_duration)
