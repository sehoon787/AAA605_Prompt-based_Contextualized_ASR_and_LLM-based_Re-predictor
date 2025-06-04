import os
import tarfile

# 압축파일 있는 경로
DOWNLOAD_DIR = r"C:\Users\Administrator\Desktop\ku\1-2\AAA605_Prompt-based_Contextualized_ASR_and_LLM-based_Re-predictor\data"
# 압축해제 대상 경로
EXTRACT_DIR = os.path.join(DOWNLOAD_DIR, "LibriSpeech")

# 압축파일 리스트
ARCHIVES = [
    "train-clean-100.tar.gz",
    "dev-clean.tar.gz",
    "dev-other.tar.gz",
    "test-clean.tar.gz",
    "test-other.tar.gz"
]

def extract_if_needed(archive_filename):
    archive_path = os.path.join(DOWNLOAD_DIR, archive_filename)

    # 폴더명은 압축파일명에서 ".tar.gz" 제거
    folder_name = archive_filename.replace(".tar.gz", "")
    target_dir = os.path.join(EXTRACT_DIR, folder_name)

    if os.path.exists(target_dir):
        print(f"[SKIP] {folder_name} already extracted.")
        return

    print(f"[EXTRACT] {archive_filename} ...")

    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=EXTRACT_DIR)

    print(f"[DONE] Extracted {folder_name}")

def prepare_all():
    for archive in ARCHIVES:
        extract_if_needed(archive)

if __name__ == "__main__":
    prepare_all()
