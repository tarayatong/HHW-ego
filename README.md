# HHW-Ego Image Enhancement

This repository contains code and pipeline for **HHW-Ego** image enhancement, targeting low-quality wearable ego-centric images captured by EmdoorVR and RayNeo devices. The enhancement is performed in two stages: super-resolution reconstruction and hyperspectral (HSI) color correction.

---

## Dataset

The following datasets are used in this project:

- **EmdoorVR_img**: Images captured by EmdoorVR glasses.  
- **RayNeo_img**: Images captured by RayNeo glasses.  
- **HSI_data**: Hyperspectral data for color correction.  
- **HDR_img**: High-quality images captured by a digital camera.  
- **Glasses_img**: Aligned images obtained by structurally aligning EmdoorVR images with HDR images.

---

## Code Overview

### Stage 1: Super-Resolution Reconstruction

This stage enhances low-quality images using super-resolution models.

1. **General Super-Resolution**  
   - Use a general super-resolution model to reconstruct images.

2. **Real-ESRGAN Enhancement**
   - **Real-ESRGAN**: [GitHub Repository](https://github.com/xinntao/Real-ESRGAN)
   - The full Real-ESRGAN pipeline includes:
     1. **High-Quality Image Degradation**  
        - Degrade HDR images to simulate low-quality inputs.  
        - Script:  
          ```bash
          python realesrgan/models/telehyper_model.py
          ```
     2. **Training / Fine-Tuning**  
        - Train or fine-tune the Real-ESRGAN model on your dataset.  
        - Script:  
          ```bash
          python train.py
          ```
     3. **Inference on Low-Quality Glasses Images**  
        - Apply the trained model to enhance EmdoorVR / RayNeo images.  
        - Script:  
          ```bash
          python inference_realesrgan.py
          ```


---

### Stage 2: HSI Color Correction

This stage performs color correction using hyperspectral data.

- Script for HSI-based color correction with windowed mask and overexposure handling:  
  ```bash
  python hsi/hsi_pipeline_window_mask_overexposure.py
