

## tensorboard


  >> tensorboard --logdir ./output/lora_train_avengers_fp16/logs --host 0.0.0.0 --port 8610



  

* fp 16 학습 장애 
  
  * 은 SDXL 기본 VAE를 FP16 autocast로 디코딩하면서 NaN이 발생하는 문제입니다. TensorBoard 자체 문제는 아닙니다. 다만 학습 시 VAE encode는 FP32라서, 검은 validation 이미지가 곧 LoRA 학습 실패를 의미하지는 않습니다.
  * RTX A6000에서는 가장 간단하게 다음 실행부터 BF16을 권장합니

# FP16을 유지하려면 FP16용 VAE를 추가하세요.
# --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix"