import subprocess
import os

# 필요한 라이브러리 및 버전 정의
PACKAGES = [
    "thop",
    "basicsr",
    "einops",
    "matplotlib",
    "tensorboardX",
    "numpy==1.26.4"
]

# # Git 저장소 URL 및 경로
# GIT_REPO_URL = "https://github.com/XPixelGroup/HAT.git" 
# GIT_REPO_DIR = 'HAT'

# basicsr 코드 수정 경로 및 내용
BASICS_R_FILE_PATH = "/usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py"
OLD_LINE = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW_LINE = "from torchvision.transforms.functional import rgb_to_grayscale"

# # HAT 코드 수정 경로 및 내용
# HAT_FILE_PATH = "/workspace/HAT/hat/data/imagenet_paired_dataset.py"
# OLD_importing = "from basicsr.utils.matlab_functions import imresize, rgb2ycbcr"
# NEW_importing = "from basicsr.utils.matlab_functions import imresize\nfrom basicsr.utils.color_util import rgb2ycbcr"

def run_command(command):
    """터미널 명령어를 실행하고 결과를 출력."""
    print(f"Executing: {command}")
    process = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    print(process.stdout)
    if process.stderr:
        print("Error Output:", process.stderr)


def setup_packages():
    print("--- Installing required packages ---")

    packages_cmd = f"pip install {' '.join(PACKAGES)}"
    run_command(packages_cmd)

# def clone_repo():
#     """Git 저장소를 클론"""
#     print("\n--- Cloning Git repository ---")
#     if not os.path.exists(GIT_REPO_DIR):
#         run_command(f"git clone {GIT_REPO_URL} {GIT_REPO_DIR}")
#     else:
#         print(f"Repository '{GIT_REPO_DIR}' already exists. Skipping clone.")

def modify_basicsr():
    """basicsr 코드를 수정."""
    print("\n--- Modifying basicsr code ---")
    if os.path.exists(BASICS_R_FILE_PATH):
        with open(BASICS_R_FILE_PATH, 'r') as f:
            content = f.read()

        if OLD_LINE in content:
            content = content.replace(OLD_LINE, NEW_LINE)
            with open(BASICS_R_FILE_PATH, 'w') as f:
                f.write(content)
            print(f"Successfully modified '{BASICS_R_FILE_PATH}'.")
        else:
            print(f"Modification not needed for '{BASICS_R_FILE_PATH}'.")
    else:
        print(f"basicsr file not found at '{BASICS_R_FILE_PATH}'. Skipping modification.")

# def modify_HAT():
#     """HAT 코드 수정."""
#     print("\n--- Modifying HAT code ---")
#     if os.path.exists(HAT_FILE_PATH):
#         with open(HAT_FILE_PATH, 'r') as f:
#             content = f.read()

#         if OLD_importing in content:
#             content = content.replace(OLD_importing, NEW_importing)
#             with open(HAT_FILE_PATH, 'w') as f:
#                 f.write(content)
#             print(f"Successfully modified '{HAT_FILE_PATH}'.")
#         else:
#             print(f"Modification not needed for '{HAT_FILE_PATH}'.")
#     else:
#         print(f"HAT file not found at '{HAT_FILE_PATH}'. Skipping modification.")


if __name__ == "__main__":
    setup_packages()
    # clone_repo()
    modify_basicsr()
    # modify_HAT()
    print("\n--- Environment setup complete! ---")