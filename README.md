# 3-Depth SAM Qwen Reset

A comprehensive project that combines depth estimation, image segmentation, and multimodal visual understanding. This project uses Depth-Anything-V2 for depth estimation, SAM (Segment Anything Model) for image segmentation, and performs multimodal visual question answering through Qwen-VL and GPT-4o.

## Features

- **Depth Estimation**: Generate depth maps from images using Depth-Anything-V2 model
- **Image Segmentation**: Automatically generate image masks using SAM (Segment Anything Model)
- **Multimodal Understanding**: Combine depth maps and segmentation maps for visual question answering using large language models
- **Result Visualization**: Generate visualizations of depth maps, segmentation maps, and fused results

## Project Structure

```
sam_reset/
├── 3-depth_sam_qwen_reset.ipynb  # Main notebook file
├── README.md                      # Project documentation
├── notebook_utils.py             # OpenVINO utility functions (auto-downloaded)
├── cmd_helper.py                  # Command helper functions (auto-downloaded)
└── Depth-Anything-V2/            # Depth-Anything-V2 repository (auto-cloned)
```

## Requirements

### Python Dependencies

- `openvino>=2024.2.0`
- `datasets>=2.14.6`
- `nncf>=2.11.0`
- `torch` and `torchvision`
- `opencv-python`
- `segment_anything`
- `huggingface_hub`
- `openai`
- `gradio>=4.19`
- `matplotlib>=3.4`
- `scikit-image`
- `numpy`
- `PIL` (Pillow)
- `requests`

### Model Files

- **SAM Model**: `sam_vit_h_4b8939.pth` (~2.4GB, auto-downloaded)
- **Depth-Anything-V2 Model**: Auto-downloaded from Hugging Face

### API Keys

- **Qwen-VL API**: DashScope API key required
- **GPT-4o API** (optional): OpenAI API key required

## Usage

### 1. Environment Setup

The notebook automatically handles:
- Network proxy configuration (if needed)
- Downloading necessary utility functions
- Cloning the Depth-Anything-V2 repository
- Installing required Python packages

### 2. Model Preparation

- Depth-Anything-V2 model is automatically downloaded from Hugging Face
- SAM model is automatically downloaded from Facebook Research (~2.4GB)

### 3. Workflow

1. **Depth Estimation**: Generate depth map from input image using Depth-Anything-V2
2. **Image Segmentation**: Automatically generate image masks and segmentation results using SAM
3. **Result Fusion**: Overlay depth map and segmentation boundaries for visualization
4. **Multimodal Q&A**: Input depth map and segmentation map to Qwen-VL or GPT-4o for visual question answering

### 4. Output Files

After running, the following files will be generated:
- `depth.png`: Depth map visualization result
- `mask.png`: Segmentation mask visualization result
- `mask_boundary.png`: Fused result of depth map and segmentation boundaries

## Main Modules

### Depth Estimation Module

Uses Depth-Anything-V2 with ViT-Small encoder for depth estimation:
- Input: RGB image
- Output: Depth map (numpy array)
- Visualization: Uses INFERNO colormap

### Image Segmentation Module

Uses SAM (Segment Anything Model) for automatic image segmentation:
- Model: SAM ViT-H (vit_h)
- Functionality: Automatically generates masks for all objects in the image
- Output: List of dictionaries containing segmentation masks, bounding boxes, etc.

### Multimodal Q&A Module

Supports two large language models:
- **Qwen-VL**: Called via DashScope API
- **GPT-4o**: Called via OpenAI API

Can simultaneously input depth map and segmentation map for visual question answering tasks.

## Notes

1. **Network Proxy**: If using in mainland China, network proxy may be required to access Hugging Face and download models
2. **Model Size**: SAM model file is large (~2.4GB), download may take some time
3. **API Keys**: Multimodal Q&A functionality requires corresponding API keys
4. **Memory Requirements**: Running SAM and depth estimation models requires significant memory
5. **xFormers**: xFormers is disabled in the code, using standard PyTorch implementation

## Example Usage

```python
# Depth estimation
depth = model.forward(image)

# Image segmentation
masks = mask_generator.generate(raw_img)

# Multimodal Q&A
query_qwen(image_base64_1, image_base64_2, "What are in these images?")
```

## Tech Stack

- **Depth Estimation**: Depth-Anything-V2 (ViT-Small)
- **Image Segmentation**: SAM (Segment Anything Model)
- **Multimodal Understanding**: Qwen-VL-Max, GPT-4o
- **Deep Learning Framework**: PyTorch
- **Image Processing**: OpenCV, PIL
- **Visualization**: Matplotlib

## License

Please refer to the original licenses of each component:
- Depth-Anything-V2: Check its GitHub repository
- SAM: Apache 2.0
- Qwen-VL: Check DashScope terms of use
- GPT-4o: Check OpenAI terms of use

## References

- [Depth-Anything-V2](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
- [Qwen-VL](https://dashscope.aliyuncs.com/)
- [OpenAI GPT-4o](https://openai.com/)

## Changelog

- Initial version: Integrated depth estimation, image segmentation, and multimodal Q&A functionality
