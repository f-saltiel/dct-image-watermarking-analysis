import cv2
import numpy as np
import matplotlib.pyplot as plt

# set seed so noise experiments are reproducible
np.random.seed(123)

# =========================
# CONSTANTS
# =========================

# JPEG luminance quantization matrix
Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103,77],
    [24, 35, 55, 64, 81,104,113,92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103,99]
], dtype=np.float32)

ALPHA = 0.5  # embedding strength
BLOCK_SIZE = 8  # standard JPEG block size

# using one coefficient per block
# picked (2,2) because it’s in the mid-frequencies
EMBED_POS = (2, 2)

# inputs
IMAGE_PATH = "lena_color_512.jpg"
WATERMARK_TEXT = "Supercalifragilisticexpialidocious"


# =========================
# IMAGE UTILS
# =========================

def load_image(path):
    # load image and convert from BGR (opencv) to RGB
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_ycbcr(img):
    # convert to YCrCb
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def extract_y_channel(img_ycbcr):
    # Y channel contains brightness info
    return img_ycbcr[:, :, 0]


def split_into_blocks(channel, block_size=BLOCK_SIZE):
    # break image into 8x8 blocks
    h, w = channel.shape

    if h % block_size != 0 or w % block_size != 0:
        raise ValueError("Image dimensions must be divisible by block size.")

    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            blocks.append(channel[i:i + block_size, j:j + block_size])

    return blocks


def merge_blocks(blocks, shape, block_size=BLOCK_SIZE):
    # reconstruct full image from blocks
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)

    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            out[i:i + block_size, j:j + block_size] = blocks[idx]
            idx += 1

    return np.clip(out, 0, 255).astype(np.uint8)


def rebuild_rgb(original_ycbcr, new_y):
    # replace Y channel and convert back to RGB
    ycbcr_mod = original_ycbcr.copy()
    ycbcr_mod[:, :, 0] = new_y
    return cv2.cvtColor(ycbcr_mod, cv2.COLOR_YCrCb2RGB)


# =========================
# DCT + QUANTIZATION
# =========================

def apply_dct(block):
    # shift by 128 (standard JPEG step), then apply DCT
    block = np.float32(block) - 128.0
    return cv2.dct(block)


def apply_idct(block):
    # inverse DCT + shift back
    spatial = cv2.idct(np.float32(block)) + 128.0
    return np.clip(spatial, 0, 255)


def apply_dct_blocks(blocks):
    return [apply_dct(block) for block in blocks]


def apply_idct_blocks(blocks):
    return [apply_idct(block) for block in blocks]


def quantize_block(dct_block):
    exact = dct_block / Q
    CQ = np.round(exact)
    R = exact - CQ
    return CQ.astype(np.float32), R.astype(np.float32)


# =========================
# WATERMARK UTILS
# =========================

def text_to_bits(text):
    # convert each character into 8-bit binary
    bits = []
    for char in text:
        bits.extend(int(bit) for bit in format(ord(char), "08b"))
    return bits


def bits_to_text(bits):
    # convert bits back to characters (group into bytes)
    chars = []
    usable_length = (len(bits) // 8) * 8

    for i in range(0, usable_length, 8):
        byte = bits[i:i + 8]
        chars.append(chr(int("".join(map(str, byte)), 2)))

    return "".join(chars)


def bits_to_text_preview(bits, max_chars=200):
    # just show a small portion so output isn’t massive
    return bits_to_text(bits)[:max_chars]


# =========================
# EMBEDDING
# =========================

def embed_bit_in_block(CQ, R, bit, alpha=ALPHA):
    CQ_mod = CQ.copy()
    R_mod = R.copy()

    i, j = EMBED_POS

    # map bit to -1 or +1
    W = 1 if bit == 1 else -1

    # reconstruct coefficient
    C = CQ[i, j] + R[i, j]

    # embedding formula
    C_prime = (1.0 - alpha) * C + alpha * W

    # store new value back into quantized form
    CQ_mod[i, j] = np.round(C_prime)
    R_mod[i, j] = C_prime - CQ_mod[i, j]

    # go back to DCT domain
    return (CQ_mod + R_mod) * Q


def embed_image(image_path, watermark_text, alpha=ALPHA):
    img_rgb = load_image(image_path)
    img_ycbcr = rgb_to_ycbcr(img_rgb)
    y_channel = extract_y_channel(img_ycbcr)

    blocks = split_into_blocks(y_channel, block_size=BLOCK_SIZE)
    dct_blocks = apply_dct_blocks(blocks)

    watermark_bits = text_to_bits(watermark_text)

    modified_dct_blocks = []
    bit_idx = 0

    # embed 1 bit per block (loop watermark if we run out)
    for block in dct_blocks:
        CQ, R = quantize_block(block)
        bit = watermark_bits[bit_idx % len(watermark_bits)]
        modified_block = embed_bit_in_block(CQ, R, bit, alpha=alpha)
        modified_dct_blocks.append(modified_block)
        bit_idx += 1

    spatial_blocks = apply_idct_blocks(modified_dct_blocks)
    y_reconstructed = merge_blocks(spatial_blocks, y_channel.shape, block_size=BLOCK_SIZE)
    watermarked_rgb = rebuild_rgb(img_ycbcr, y_reconstructed)

    return img_rgb, watermarked_rgb, watermark_bits


# =========================
# EXTRACTION
# =========================

def extract_bit_from_block(CQ_orig, R_orig, CQ_mod, R_mod, alpha=ALPHA):
    i, j = EMBED_POS

    # reconstruct both coefficients
    C = CQ_orig[i, j] + R_orig[i, j]
    C_prime = CQ_mod[i, j] + R_mod[i, j]

    # reverse embedding formula to recover W
    W = (C_prime - (1.0 - alpha) * C) / alpha

    return 1 if W >= 0 else 0


def extract_image(orig_rgb, mod_rgb, alpha=ALPHA):
    # same pipeline as embedding but reversed
    y_orig = extract_y_channel(rgb_to_ycbcr(orig_rgb))
    y_mod = extract_y_channel(rgb_to_ycbcr(mod_rgb))

    blocks_orig = split_into_blocks(y_orig, block_size=BLOCK_SIZE)
    blocks_mod = split_into_blocks(y_mod, block_size=BLOCK_SIZE)

    dct_orig = apply_dct_blocks(blocks_orig)
    dct_mod = apply_dct_blocks(blocks_mod)

    raw_bits = []

    for block_orig, block_mod in zip(dct_orig, dct_mod):
        CQ_orig, R_orig = quantize_block(block_orig)
        CQ_mod, R_mod = quantize_block(block_mod)

        bit = extract_bit_from_block(CQ_orig, R_orig, CQ_mod, R_mod, alpha=alpha)
        raw_bits.append(bit)

    return raw_bits


# =========================
# METRICS
# =========================

def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else float("inf")


def ssim(a, b):
    # compares structural similarity between original and modified image in Y channel
    y_a = extract_y_channel(rgb_to_ycbcr(a)).astype(np.float32)
    y_b = extract_y_channel(rgb_to_ycbcr(b)).astype(np.float32)

    mu_a = np.mean(y_a)
    mu_b = np.mean(y_b)

    var_a = np.var(y_a)
    var_b = np.var(y_b)

    cov_ab = np.mean((y_a - mu_a) * (y_b - mu_b))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)

    return numerator / denominator


def ber(a, b):
    a = np.array(a, dtype=np.uint8)
    b = np.array(b, dtype=np.uint8)

    if len(a) != len(b):
        raise ValueError("BER inputs must have same length.")

    return np.mean(a != b)


def nc(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    if len(a) != len(b):
        raise ValueError("NC inputs must have same length.")

    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return (np.sum(a * b) / denom) if denom != 0 else 0.0


def compute_ber_all_segments(raw_bits, original_bits):
    # to evaluate the repeated embedding fairly, average BER over all full recovered segments.
    L = len(original_bits)
    if L == 0:
        raise ValueError("Original watermark bit sequence is empty.")

    num_segments = len(raw_bits) // L
    if num_segments == 0:
        raise ValueError("Not enough extracted bits to form one full segment.")

    bers = []
    for i in range(num_segments):
        segment = raw_bits[i * L:(i + 1) * L]
        bers.append(ber(original_bits, segment))

    return float(np.mean(bers))


def compute_nc_all_segments(raw_bits, original_bits):
    # average NC over all full recovered segments.
    L = len(original_bits)
    if L == 0:
        raise ValueError("Original watermark bit sequence is empty.")

    num_segments = len(raw_bits) // L
    if num_segments == 0:
        raise ValueError("Not enough extracted bits to form one full segment.")

    ncs = []
    for i in range(num_segments):
        segment = raw_bits[i * L:(i + 1) * L]
        ncs.append(nc(original_bits, segment))

    return float(np.mean(ncs))


# =========================
# ATTACKS
# =========================

def gaussian_noise_variance(img, variance):
    sigma = np.sqrt(variance) * 255.0
    noise = np.random.normal(0.0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def salt_pepper_noise(img, prob=0.01):
    out = img.copy()
    rnd = np.random.rand(*img.shape[:2])

    out[rnd < prob] = 0
    out[rnd > 1 - prob] = 255

    return out


def jpeg_attack_quality(img, quality):
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), params)
    if not success:
        raise RuntimeError("JPEG encoding failed.")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def rotate_with_padding(img, angle):
    h, w = img.shape[:2]

    # Pad image so rotation does not cut off corners
    pad = int(np.ceil(np.sqrt(h*h + w*w)))
    pad_y = (pad - h) // 2
    pad_x = (pad - w) // 2

    padded = cv2.copyMakeBorder(
        img,
        pad_y, pad - h - pad_y,
        pad_x, pad - w - pad_x,
        borderType=cv2.BORDER_REFLECT
    )

    ph, pw = padded.shape[:2]
    M = cv2.getRotationMatrix2D((pw // 2, ph // 2), angle, 1.0)

    rotated = cv2.warpAffine(
        padded,
        M,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return rotated, (pad_y, pad_x, h, w)


def inverse_rotate_and_crop(rotated_img, angle, crop_info):
    pad_y, pad_x, h, w = crop_info

    ph, pw = rotated_img.shape[:2]
    M = cv2.getRotationMatrix2D((pw // 2, ph // 2), -angle, 1.0)

    corrected = cv2.warpAffine(
        rotated_img,
        M,
        (pw, ph),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return corrected[pad_y:pad_y+h, pad_x:pad_x+w]


# =========================
# REPORTING
# =========================

def print_result_block(name, attacked_img, orig_img, watermark_bits):
    raw_bits = extract_image(orig_img, attacked_img, alpha=ALPHA)
    extracted_preview = bits_to_text_preview(raw_bits, max_chars=200)

    print(f"\n=== {name} ===")
    clean_preview = ''.join(
        c if 32 <= ord(c) <= 126 else '?'
        for c in extracted_preview
    )
    print("Extracted preview:", clean_preview)
    print("PSNR:", psnr(orig_img, attacked_img))
    print("SSIM:", ssim(orig_img, attacked_img))
    print("BER:", compute_ber_all_segments(raw_bits, watermark_bits))
    print("NC:", compute_nc_all_segments(raw_bits, watermark_bits))


# =========================
# MAIN EXPERIMENT
# =========================

if __name__ == "__main__":
    orig, watermarked, watermark_bits = embed_image(
        image_path=IMAGE_PATH,
        watermark_text=WATERMARK_TEXT,
        alpha=ALPHA
    )

    # No attack
    print_result_block(
        name="NO ATTACK",
        attacked_img=watermarked,
        orig_img=orig,
        watermark_bits=watermark_bits
    )

    # JPEG quality factors reported for color images in the paper
    for q in [90, 70, 30]:
        attacked = jpeg_attack_quality(watermarked, quality=q)
        print_result_block(
            name=f"JPEG Q={q}",
            attacked_img=attacked,
            orig_img=orig,
            watermark_bits=watermark_bits
        )

    # Salt & Pepper density reported in the paper
    attacked_sp = salt_pepper_noise(watermarked, prob=0.01)
    print_result_block(
        name="SALT & PEPPER p=0.01",
        attacked_img=attacked_sp,
        orig_img=orig,
        watermark_bits=watermark_bits
    )

    # Gaussian variance reported in the paper
    attacked_gauss = gaussian_noise_variance(watermarked, variance=0.001)
    print_result_block(
        name=f"GAUSSIAN var=0.001",
        attacked_img=attacked_gauss,
        orig_img=orig,
        watermark_bits=watermark_bits
    )

    # Rotation angles discussed in the paper results
    for angle in [1, 5, 45]:
      attacked_rot, crop_info = rotate_with_padding(watermarked, angle)
      # undo rotation before extraction
      corrected = inverse_rotate_and_crop(attacked_rot, angle, crop_info)

      print_result_block(
          name=f"ROTATION {angle} deg",
          attacked_img=corrected,
          orig_img=orig,
          watermark_bits=watermark_bits
      )

    # Visualization
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(orig)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(watermarked)
    plt.title("Watermarked")
    plt.axis("off")

    plt.tight_layout()
    plt.show()