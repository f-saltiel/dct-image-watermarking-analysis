# DCT-Based Digital Image Watermarking Reproduction and Analysis

This repository contains the Python implementation for an undergraduate research project based on the article **"Digital image watermarking using discrete cosine transformation based linear modulation"** by Alomoush et al.

The project implements and tests a DCT-based digital image watermarking method using Python, OpenCV, NumPy, and Matplotlib.

The main goal is to reproduce the core behavior of the watermarking method described in the article and explore how different implementation choices affect image quality and watermark recovery.

## Project Overview

The watermarking process follows this general pipeline:

1. Load the host image.
2. Convert the image from RGB to YCrCb.
3. Extract the Y channel.
4. Divide the Y channel into 8x8 blocks.
5. Apply DCT to each block.
6. Quantize the DCT coefficients.
7. Embed the watermark bits into selected DCT coefficients.
8. Reconstruct the watermarked image using inverse DCT.
9. Apply image attacks.
10. Extract the watermark and evaluate recovery.

The watermark used in this implementation is:

```text
Supercalifragilisticexpialidocious
```

The host image used is a 512x512 RGB image of Lena.

## Implementations

This repository includes two implementation versions.

### Version 1: Single-Coefficient Embedding

`version1.py` embeds each watermark bit into one fixed mid-frequency DCT coefficient per 8x8 block.

The selected coefficient is position `(2,2)`.

This version is simpler and modifies fewer coefficients, which helps preserve image quality. However, since each bit depends on only one coefficient, it is more sensitive to attacks that alter that coefficient.

### Version 2: Multi-Coefficient Redundancy

`version2.py` embeds the same watermark bit across multiple mid-frequency DCT coefficients within each 8x8 block.

The coefficients are selected using a zig-zag ordering pattern. During extraction, the bit is recovered using majority voting across those coefficients.

This version adds redundancy, which improves watermark recovery under several attacks. The tradeoff is that more coefficients are modified, which can reduce image quality.

## Main Difference Between Versions

| Version | Embedding Method | Purpose |
|--------|------------------|---------|
| Version 1 | One fixed coefficient per block | Baseline implementation with higher image quality |
| Version 2 | Multiple coefficients per block with majority voting | More robust implementation with added redundancy |

## Metrics Calculated

The code calculates the following metrics:

| Metric | Purpose |
|--------|---------|
| PSNR | Measures image distortion between original and modified image |
| SSIM | Measures structural similarity between original and modified image |
| BER | Measures bit error rate between original and extracted watermark |
| NC | Measures normalized correlation between original and extracted watermark |

## Image Attacks Tested

The implementations test the watermarked image against:

| Attack | Parameters |
|--------|------------|
| JPEG Compression | Q=90, Q=70, Q=30 |
| Salt & Pepper Noise | p=0.01 |
| Gaussian Noise | var=0.001 |
| Rotation | 1°, 5°, 45° |

For rotation attacks, the image is rotated with padding to avoid cutting off the corners. Then inverse rotation and cropping are applied before watermark extraction.

## Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## How to Run

To run Version 1:

```bash
python version1.py
```

To run Version 2:

```bash
python version2.py
```

Make sure the image path inside each script points to the correct image location.

By default, the scripts expect:

```text
images/lena_color_512.jpg
```

If your image is stored somewhere else, update the `IMAGE_PATH` variable in the script.

## Notes

This project intentionally uses a 512x512 Lena image and a 34-character text watermark. These differ from the exact experimental setup in the article, but the implementation follows the same general DCT-based watermarking process.

The repository is meant to store and distribute the implementation code. Detailed result tables, comparison with the article, and analysis are included separately in the final report.

## Image Source

The Lena test image used in this project comes from the following repository:

https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing

The repository provides a collection of standard test images for image processing and is licensed under the MIT License.

## Reference

Alomoush, W., Khashan, O. A., Alrosan, A., Attar, H. H., Almomani, A., Alhosban, F., & Makhadmeh, S. N.

**Digital image watermarking using discrete cosine transformation based linear modulation.**

Journal of Cloud Computing, 2023.
