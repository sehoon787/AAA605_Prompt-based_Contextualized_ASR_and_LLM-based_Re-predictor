# Data Source: https://www.openslr.org/12
import os
import glob
import pathlib
import tarfile

# 압축파일 있는 경로
DOWNLOAD_DIR = r"C:\Users\Administrator\Desktop\ku\1-2\AAA605_Prompt-based_Contextualized_ASR_and_LLM-based_Re-predictor\data"

def extract_if_needed(archive_path):
    extract_dir = os.path.splitext(os.path.splitext(archive_path)[0])[0]

    if not os.path.exists(extract_dir):
        print(f"Extracting {archive_path}...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(archive_path))
    else:
        print(f"Already extracted: {archive_path}")

def prepare_all(data_dir):
    # 주어진 디렉토리 내부의 모든 .tar.gz 파일 검색
    archives = glob.glob(os.path.join(data_dir, '**', '*.tar.gz'), recursive=True)

    for archive in archives:
        extract_if_needed(archive)

def safe_extract(tar, path="."):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        abs_path = pathlib.Path(member_path).resolve()
        if not str(abs_path).startswith(str(pathlib.Path(path).resolve())):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path=path)

if __name__ == "__main__":
    prepare_all(DOWNLOAD_DIR)
