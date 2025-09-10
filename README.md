This repository is based on Hybrid Attention Transformer ([HAT](https://github.com/XPixelGroup/HAT))
# ZEE (Zoom Enhance Enhance)

Super Resolution 기술을 활용하여 저해상도의 위성 사진을 고해상도로 복원하는 프로젝트. 

이를 통해 기존 위성 사진의 한계를 극복하고, 지표면에 있는 객체를 더욱 정밀하게 식별하고 분할(Segmentation)하여 보다 정확한 공간 정보를 획득할 수 있도록 합니다.

# Setup
`
RunPod의 RTX 4090 GPU, `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` template 기준 환경에서 `setup_env_hat4090.py`를 실행하여 환경을 구성합니다.

'''
python setup_env_hat4090.py
'''

# Train
'''
python -m hat.train -opt options/train/custom/train_HAT_SRx4_finetune_from_ImageNet_pretrain_custom_sumi.yml
'''

# Test
'''
python -m hat.test -opt options/test/custom/HAT_SRx4_finetune_custom_sam.yml
'''
