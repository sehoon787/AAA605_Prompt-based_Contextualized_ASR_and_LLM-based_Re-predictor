import subprocess
import os

def huggingface_login():
    # 이미 토큰이 있는지 확인
    if os.path.exists(os.path.expanduser("~/.huggingface/token")):
        print("✅ 이미 Hugging Face에 로그인되어 있습니다.")
        return

    print("🔐 Hugging Face 계정 로그인이 필요합니다.")
    print("👉 토큰을 발급받아 아래 안내에 따라 입력해주세요:")
    print("   1. https://huggingface.co/settings/tokens 접속")
    print("   2. New token 생성 (권한: read)")
    print("   3. 아래 명령어 실행하여 로그인")
    print()

    # huggingface-cli login 호출
    try:
        subprocess.run(["huggingface-cli", "login"], check=True)
    except subprocess.CalledProcessError:
        print("❌ 로그인 실패: huggingface-cli 설치 확인 필요")
