"""
pipeline_tools.py - Clinical dermoscopy preprocessing pipeline inspection,
lesion segmentation, ABCDE feature analysis, and diagnostic decision reporting.
"""

import cv2
import numpy as np
import pandas as pd
import inference as inf

# Class details and clinical references for ISIC 2019
CLASS_INFO = {
    'MEL': {
        'name': 'Melanoma',
        'type': 'Malignant',
        'risk': 'High',
        'color': '#ef4444',
        'desc': 'Aggressive malignant skin cancer arising from melanocytes. Early diagnosis and prompt excision are crucial.',
        'hallmarks': 'Atypical pigment network, irregular dots/globules, blue-white veil, peripheral streaks, asymmetric structure.',
        'action': 'Urgent urgent dermatologist referral and full-thickness excisional biopsy with 1-2 mm margins.'
    },
    'BCC': {
        'name': 'Basal Cell Carcinoma',
        'type': 'Malignant',
        'risk': 'High',
        'color': '#f97316',
        'desc': 'Most common form of non-melanoma skin cancer. Locally invasive but very rarely metastasizes.',
        'hallmarks': 'Arborizing (branching) telangiectasia, blue-gray ovoid nests/globules, ulceration, shiny translucent hue.',
        'action': 'Referral for dermatologic evaluation, surgical excision, Mohs micrographic surgery, or cryotherapy.'
    },
    'SCC': {
        'name': 'Squamous Cell Carcinoma',
        'type': 'Malignant',
        'risk': 'High',
        'color': '#dc2626',
        'desc': 'Second most common skin cancer; arises from keratinocytes in the epidermis with metastatic potential.',
        'hallmarks': 'Keratin pearl formation, hyperkeratosis, central crusting, glomerular/hairpin blood vessels, targetoid white circles.',
        'action': 'Urgent dermatological assessment and histological biopsy; excision with appropriate margins.'
    },
    'AK': {
        'name': 'Actinic Keratosis',
        'type': 'Precancerous',
        'risk': 'Moderate',
        'color': '#f59e0b',
        'desc': 'Dysplastic intraepidermal proliferation of keratinocytes induced by chronic UV radiation. Can evolve into invasive SCC.',
        'hallmarks': 'Red pseudonetwork, prominent follicular openings with white/yellow keratotic plugs (strawberry pattern), superficial scaling.',
        'action': 'Cryotherapy with liquid nitrogen, topical fluorouracil (5-FU), imiquimod, or photodynamic therapy.'
    },
    'NV': {
        'name': 'Melanocytic Nevus',
        'type': 'Benign',
        'risk': 'Low',
        'color': '#10b981',
        'desc': 'Common benign proliferation of melanocytes (mole). Regular architecture with symmetric progression.',
        'hallmarks': 'Regular uniform pigment network, symmetric distribution of globule aggregates, homogeneous pattern.',
        'action': 'Routine monitoring. Educate patient on self-examination (ABCDE criteria).'
    },
    'BKL': {
        'name': 'Benign Keratosis (Seborrheic Keratosis / Solar Lentigo)',
        'type': 'Benign',
        'risk': 'Low',
        'color': '#059669',
        'desc': 'Very common benign epidermal tumor typically seen in older adults; non-cancerous with zero metastatic risk.',
        'hallmarks': 'Milia-like cysts, comedo-like openings, gyri/sulci (cerebriform brain-like appearance), sharp demarcated border.',
        'action': 'Reassurance. Cosmetic removal (cryotherapy/curettage) only if irritated or symptomatic.'
    },
    'DF': {
        'name': 'Dermatofibroma',
        'type': 'Benign',
        'risk': 'Low',
        'color': '#0d9488',
        'desc': 'Benign cutaneous fibrohistiocytic lesion, commonly found on extremities; positive pinch/dimple sign.',
        'hallmarks': 'Central white patch (scar-like area) surrounded by a delicate, faint peripheral pigment network.',
        'action': 'Reassurance. Biopsy only if features are atypical or history of rapid expansion.'
    }
}


def run_pipeline_stages(bgr_img):
    """
    Executes and retains every intermediate stage of the dermoscopy preprocessing pipeline:
    1. Raw image acquisition & format conversion
    2. Morphological hair detection (Blackhat filtering)
    3. DullRazor hair inpainting (Telea algorithm)
    4. Denoising (Median / Gaussian)
    5. Color constancy / Illumination correction (Shades of Gray)
    6. Resize to 320x320 model input tensor
    7. Lesion segmentation & boundary extraction
    """
    h_orig, w_orig = bgr_img.shape[:2]
    
    # 1. Downscale if max side exceeds config threshold
    hair_max = inf.CFG.get('hair_max_side', 768)
    if max(h_orig, w_orig) > hair_max:
        s = hair_max / max(h_orig, w_orig)
        working_bgr = cv2.resize(bgr_img, (int(w_orig * s), int(h_orig * s)), interpolation=cv2.INTER_AREA)
    else:
        working_bgr = bgr_img.copy()

    # 2. Blackhat filter for hair detection
    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    hair_thr = inf.CFG.get('hair_thr', 25)
    _, hair_mask = cv2.threshold(blackhat, hair_thr, 255, cv2.THRESH_BINARY)
    
    # Hair density calculation
    hair_density = float(np.count_nonzero(hair_mask)) / (hair_mask.shape[0] * hair_mask.shape[1]) * 100.0

    # 3. DullRazor inpainting
    inpainted_bgr = cv2.inpaint(working_bgr, hair_mask, 3, cv2.INPAINT_TELEA)

    # 4. Denoising
    denoised_bgr = inf._denoise(inpainted_bgr)

    # 5. Model Resize
    img_size = inf.CFG.get('img_size', 320)
    resized_bgr = cv2.resize(denoised_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # 6. Color Normalization
    if inf.CFG.get('preproc') == 'paper':
        norm_rgb = inf._stretch_contrast(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
    else:
        norm_rgb = inf._shades_of_gray(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))

    # 7. Lesion Segmentation & Contour overlay
    segmentation_info = segment_lesion(resized_bgr)

    return {
        'original_dimensions': (w_orig, h_orig),
        'raw_rgb': cv2.cvtColor(working_bgr, cv2.COLOR_BGR2RGB),
        'blackhat': blackhat,
        'hair_mask': hair_mask,
        'hair_density': hair_density,
        'inpainted_rgb': cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB),
        'denoised_rgb': cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB),
        'final_tensor': norm_rgb,
        'segmentation': segmentation_info
    }


def segment_lesion(bgr_320):
    """
    Robust dermoscopic lesion segmentation via adaptive Otsu thresholding
    on saturation/luminance and morphological refinement.
    """
    hsv = cv2.cvtColor(bgr_320, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_320, cv2.COLOR_BGR2LAB)
    
    # Lesions typically have higher saturation or lower L channel compared to normal surrounding skin
    sat = hsv[..., 1]
    l_channel = lab[..., 0]
    
    # Inverted L combines well with Saturation
    combined = cv2.addWeighted(255 - l_channel, 0.6, sat, 0.4, 0)
    
    # Smooth with Gaussian
    blurred = cv2.GaussianBlur(combined, (7, 7), 0)
    
    # Otsu thresholding
    _, raw_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove outer circular vignette if present (common in dermatoscope optical barrels)
    h, w = raw_mask.shape
    vignette_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(vignette_mask, (w // 2, h // 2), int(min(h, w) * 0.47), 255, -1)
    cleaned_mask = cv2.bitwise_and(raw_mask, raw_mask, mask=vignette_mask)
    
    # Morphological opening and closing
    k_seg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, k_seg)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, k_seg)
    
    # Keep the largest contour closest to center
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lesion_contour = None
    max_score = -1
    center = np.array([w / 2.0, h / 2.0])
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (h * w * 0.02):  # at least 2% of image
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                dist_to_center = np.linalg.norm(np.array([cx, cy]) - center)
                score = area / (dist_to_center + 50.0)
                if score > max_score:
                    max_score = score
                    lesion_contour = cnt

    final_mask = np.zeros((h, w), dtype=np.uint8)
    rgb_overlay = cv2.cvtColor(bgr_320, cv2.COLOR_BGR2RGB).copy()
    
    bbox = None
    area_px = 0
    perimeter = 0
    
    if lesion_contour is not None:
        cv2.drawContours(final_mask, [lesion_contour], -1, 255, -1)
        area_px = int(cv2.contourArea(lesion_contour))
        perimeter = float(cv2.arcLength(lesion_contour, True))
        x, y, bw, bh = cv2.boundingRect(lesion_contour)
        bbox = (x, y, bw, bh)
        
        # Draw contour on overlay with translucent fill
        overlay = rgb_overlay.copy()
        cv2.drawContours(overlay, [lesion_contour], -1, (14, 165, 233), 2)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (245, 158, 11), 1)
        # translucent mask overlay
        mask_bool = final_mask > 0
        overlay[mask_bool] = (overlay[mask_bool] * 0.75 + np.array([14, 165, 233]) * 0.25).astype(np.uint8)
        rgb_overlay = overlay
    else:
        # Fallback mask: central circle
        cv2.circle(final_mask, (w // 2, h // 2), int(min(h, w) * 0.3), 255, -1)
        area_px = int(np.pi * (min(h, w) * 0.3) ** 2)
        perimeter = float(2 * np.pi * (min(h, w) * 0.3))
        bbox = (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))
        cv2.circle(rgb_overlay, (w // 2, h // 2), int(min(h, w) * 0.3), (14, 165, 233), 2)
        
    return {
        'mask': final_mask,
        'overlay_rgb': rgb_overlay,
        'area_pixels': area_px,
        'perimeter': perimeter,
        'bbox': bbox,
        'coverage_pct': (area_px / (h * w)) * 100.0,
        'contour': lesion_contour
    }


def compute_abcde_features(rgb_320, segmentation_info):
    """
    Computes objective clinical dermoscopy ABCDE metrics from the segmented lesion:
    - A: Asymmetry Index (horizontal & vertical axis asymmetry)
    - B: Border Irregularity (Compactness P^2 / (4*pi*A) & Convexity Defect score)
    - C: Color Variegation (RGB channel standard deviations & pigment variance)
    - D: Diameter / Coverage estimation
    - E: Integrated Dermoscopy Risk Rating
    """
    mask = segmentation_info['mask']
    area = max(segmentation_info['area_pixels'], 100)
    perimeter = max(segmentation_info['perimeter'], 10.0)
    
    # ------------------ A: ASYMMETRY ------------------
    # Align mask to its center of mass
    M = cv2.moments(mask)
    if M['m00'] > 0:
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    else:
        cx, cy = mask.shape[1] // 2, mask.shape[0] // 2
        
    # Horizontal flip asymmetry
    h_flipped = cv2.flip(mask, 1)
    diff_h = np.bitwise_xor(mask, h_flipped)
    asym_h = np.count_nonzero(diff_h) / (2.0 * area)

    # Vertical flip asymmetry
    v_flipped = cv2.flip(mask, 0)
    diff_v = np.bitwise_xor(mask, v_flipped)
    asym_v = np.count_nonzero(diff_v) / (2.0 * area)

    asymmetry_score = min(100.0, max(0.0, ((asym_h + asym_v) / 2.0) * 100.0 * 1.5))
    
    # ------------------ B: BORDER IRREGULARITY ------------------
    # Isoperimetric quotient / Compactness: Circle = 1.0, highly irregular > 1.8
    compactness = (perimeter ** 2) / (4.0 * np.pi * area)
    border_score = min(100.0, max(0.0, (compactness - 1.0) * 35.0))
    
    # ------------------ C: COLOR VARIEGATION ------------------
    mask_indices = mask > 0
    if np.any(mask_indices):
        pixels = rgb_320[mask_indices]
        r_std = float(np.std(pixels[:, 0]))
        g_std = float(np.std(pixels[:, 1]))
        b_std = float(np.std(pixels[:, 2]))
        total_color_std = (r_std + g_std + b_std) / 3.0
        
        # Color score: higher std implies mottled, variegated pigmentation
        color_score = min(100.0, (total_color_std / 50.0) * 100.0)
    else:
        total_color_std = 0.0
        color_score = 0.0

    # ------------------ D: DIAMETER & COVERAGE ------------------
    equiv_diameter_px = 2.0 * np.sqrt(area / np.pi)
    coverage_pct = segmentation_info['coverage_pct']
    # Estimated physical diameter assuming standard 10x-20x contact dermatoscope (approx 10-30um/pixel)
    est_diameter_mm = equiv_diameter_px * 0.045  # heuristic dermatoscope scaling factor

    # ------------------ E: INTEGRATED CLINICAL ABCDE RATING ------------------
    abc_mean = (asymmetry_score * 0.35) + (border_score * 0.35) + (color_score * 0.30)
    
    if abc_mean > 55.0:
        abc_level = 'High Irregularity'
        abc_badge = 'High'
        badge_color = '#ef4444'
    elif abc_mean > 32.0:
        abc_level = 'Moderate Irregularity'
        abc_badge = 'Moderate'
        badge_color = '#f59e0b'
    else:
        abc_level = 'Low / Uniform'
        abc_badge = 'Low'
        badge_color = '#10b981'

    return {
        'asymmetry_score': asymmetry_score,
        'asymmetry_rating': 'Asymmetric' if asymmetry_score > 40 else 'Symmetric',
        'compactness': compactness,
        'border_score': border_score,
        'border_rating': 'Irregular / Scalloped' if border_score > 35 else 'Smooth / Regular',
        'color_score': color_score,
        'color_std': total_color_std,
        'color_rating': 'Variegated / Mottled' if color_score > 45 else 'Homogeneous',
        'diameter_px': equiv_diameter_px,
        'est_diameter_mm': est_diameter_mm,
        'coverage_pct': coverage_pct,
        'composite_abc_score': abc_mean,
        'abc_level': abc_level,
        'abc_badge': abc_badge,
        'badge_color': badge_color
    }


def generate_clinical_report(res, abcde, patient_meta, orig_shape):
    """
    Generates a structured clinical inspection and consultation summary report.
    """
    label = res['label']
    info = CLASS_INFO.get(label, {})
    mal_prob = res.get('malignant_probability', 0.0)
    
    report = []
    report.append("================================================================================")
    report.append("                 DERMOSCOPIC SCAN & DIAGNOSTIC TRIAGE REPORT                    ")
    report.append("================================================================================")
    report.append(f"Model Backbone       : EfficientNetV2-M + EfficientNet-B4 Dual Trunk Fusion")
    report.append(f"Primary Finding      : {info.get('name', label)} ({label})")
    report.append(f"Classification Type  : {info.get('type', 'Unknown')} | Risk Tier: {info.get('risk', 'Unknown')}")
    report.append(f"Malignancy Likelihood: {mal_prob:.1%}")
    report.append(f"Decision Routing     : {res.get('routing', 'hard')} traversal")
    report.append("--------------------------------------------------------------------------------")
    report.append("PATIENT METADATA:")
    report.append(f" - Age               : {patient_meta.get('age', 'Unspecified')}")
    report.append(f" - Sex               : {patient_meta.get('sex', 'Unspecified').capitalize()}")
    report.append(f" - Anatomical Site   : {patient_meta.get('site', 'Unspecified').capitalize()}")
    report.append(f" - Image Size        : {orig_shape[0]} x {orig_shape[1]} px")
    report.append("--------------------------------------------------------------------------------")
    report.append("OBJECTIVE ABCDE MORPHOMETRIC ANALYSIS:")
    report.append(f" - [A] Asymmetry Index       : {abcde['asymmetry_score']:.1f}% ({abcde['asymmetry_rating']})")
    report.append(f" - [B] Border Irregularity   : {abcde['border_score']:.1f}% ({abcde['border_rating']}, Compactness: {abcde['compactness']:.2f})")
    report.append(f" - [C] Color Variegation     : {abcde['color_score']:.1f}% ({abcde['color_rating']}, Dispersion: {abcde['color_std']:.1f})")
    report.append(f" - [D] Estimated Diameter    : ~{abcde['est_diameter_mm']:.1f} mm (Area Coverage: {abcde['coverage_pct']:.1f}%)")
    report.append(f" - [E] Morphologic Risk      : {abcde['abc_level']} ({abcde['composite_abc_score']:.1f}/100)")
    report.append("--------------------------------------------------------------------------------")
    report.append("7-CLASS POSTERIOR PROBABILITY BREAKDOWN:")
    for cls_code, prob in sorted(res['probs'].items(), key=lambda x: x[1], reverse=True):
        c_name = CLASS_INFO.get(cls_code, {}).get('name', cls_code)
        c_type = CLASS_INFO.get(cls_code, {}).get('type', 'N/A')
        report.append(f" - {cls_code:4s} | {c_name:30s} [{c_type:11s}]: {prob:6.2%}")
    report.append("--------------------------------------------------------------------------------")
    report.append("HIERARCHICAL NODE TRAVERSAL:")
    for node, dist in res.get('nodes', {}).items():
        node_str = ", ".join(f"{k}: {v:.1%}" for k, v in dist.items())
        report.append(f" - Node [{node:4s}]: {node_str}")
    report.append("--------------------------------------------------------------------------------")
    report.append("CLINICAL RECOMMENDATION & DERMOSCOPIC HALLMARKS:")
    report.append(f" - Hallmarks         : {info.get('hallmarks', 'None')}")
    report.append(f" - Recommended Action: {info.get('action', 'Consult a dermatologist')}")
    report.append("--------------------------------------------------------------------------------")
    report.append("DISCLAIMER: Research prototype trained on ISIC 2019. Not an FDA/CE cleared medical")
    report.append("device. All findings must be corroborated by a licensed board-certified dermatologist.")
    report.append("================================================================================")
    
    return "\n".join(report)
