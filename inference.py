# inference.py - self-contained inference for the skin lesion classifier
# (EfficientNetV2-M + EfficientNet-B4 trunk, hierarchical branch heads, optional metadata)
# Everything it needs (class names, preprocessing settings, metadata encoding, routing) is read from config.json.
import os, json
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetV2M, EfficientNetB4

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, 'config.json')))
IMG_SIZE = CFG['img_size']
CLASSES = CFG['classes']; NC = len(CLASSES); CI = {c: i for i, c in enumerate(CLASSES)}
NODES = CFG['nodes']; NODE_CLASSES = CFG['node_classes']; NODE_K = {n: len(v) for n, v in NODE_CLASSES.items()}
FLAT_HEAD = CFG['flat_head']; USE_META = CFG['use_meta']; META_DIM = CFG['meta_dim']


# ---------------------------------------------------------------- preprocessing (identical to training)
def _remove_hair(img_bgr):
    h, w = img_bgr.shape[:2]
    if max(h, w) > CFG['hair_max_side']:
        s = CFG['hair_max_side'] / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, CFG['hair_thr'], 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

def _denoise(img):
    if CFG['denoise'] == 'median': return cv2.medianBlur(img, 3)
    if CFG['denoise'] == 'gaussian': return cv2.GaussianBlur(img, (3, 3), 0)
    return img

def _stretch_contrast(img, lo=2, hi=98):
    out = np.empty_like(img)
    for ch in range(3):
        a, b = np.percentile(img[..., ch], (lo, hi))
        out[..., ch] = np.clip((img[..., ch].astype('float32') - a) * 255.0 / max(b - a, 1.0), 0, 255).astype('uint8')
    return out

def _shades_of_gray(img, p=6):
    x = img.astype('float32'); illum = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (np.linalg.norm(illum) + 1e-6)
    return np.clip(x / (illum * np.sqrt(3.0) + 1e-6), 0, 255).astype('uint8')

def decode(image_bytes):
    # bytes of a jpg/png -> BGR array (None if it cannot be decoded)
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

def preprocess(img_bgr):
    # BGR image of any size -> RGB uint8 (IMG_SIZE x IMG_SIZE), exactly what the network was trained on
    if CFG['preproc'] == 'paper':
        img = _denoise(_remove_hair(img_bgr))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return _stretch_contrast(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = cv2.resize(_remove_hair(img_bgr), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return _shades_of_gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------- metadata encoding
def meta_vector(age=None, sex=None, site=None):
    # age in years (or None), sex in {'male','female','unknown'}, site = one of CFG['meta']['sites'] (or None)
    if not USE_META: return np.zeros(1, 'float32')
    M = CFG['meta']; v = {c: 0.0 for c in M['columns']}
    if age is None: v['age'] = 0.0; v['age_missing'] = 1.0
    else: v['age'] = (float(age) - M['age_median']) / M['age_std']; v['age_missing'] = 0.0
    v['sex_' + (sex if sex in M['sex_cats'] else 'unknown')] = 1.0
    key = 'site_' + (site if site in M['sites'] else 'unknown')
    if key in v: v[key] = 1.0
    return np.array([v[c] for c in M['columns']], 'float32')


# ---------------------------------------------------------------- model (same layer order as training, so the weights load)
def build_model():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3), name='image')
    v2m = EfficientNetV2M(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    b4  = EfficientNetB4(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    fe = layers.GlobalAveragePooling2D(name='gap_v2m')(v2m(inp))
    fd = layers.GlobalAveragePooling2D(name='gap_b4')(b4(inp))
    x = layers.Dropout(CFG['dropout'])(layers.Concatenate(name='fusion')([fe, fd]))
    inputs = [inp]
    if USE_META:
        meta_in = layers.Input((META_DIM,), name='meta'); inputs.append(meta_in)
        mf = layers.Dense(32, activation='relu', name='meta_fc')(meta_in)
        x = layers.Concatenate(name='fusion_meta')([x, mf])
    x = layers.Dense(CFG['hidden'], activation='relu', kernel_regularizer=regularizers.l2(CFG['l2']), name='fc512')(x)
    x = layers.Dropout(CFG['dropout'])(x)
    outs = []
    for n in NODES:
        h = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(CFG['l2']), name=f'{n}_fc')(x)
        h = layers.Dropout(CFG['dropout'])(h)
        outs.append(layers.Dense(NODE_K[n], activation='softmax', dtype='float32', kernel_regularizer=regularizers.l2(CFG['l2']), name=n)(h))
    if FLAT_HEAD:
        outs.append(layers.Dense(NC, activation='softmax', dtype='float32', kernel_regularizer=regularizers.l2(CFG['l2']), name='flat')(x))
    return Model(inputs, outs, name='skin_classifier')

def ensure_weights(weights_path=None):
    target = weights_path or os.path.join(HERE, 'model.weights.h5')
    if not os.path.exists(target):
        url = os.environ.get('WEIGHTS_URL')
        if not url:
            try:
                import streamlit as st
                url = st.secrets.get('WEIGHTS_URL')
            except Exception:
                pass
        if url:
            import urllib.request
            print(f"Downloading weights from {url} to {target}...")
            urllib.request.urlretrieve(url, target)
    return target

def load_model(weights_path=None):
    m = build_model()
    path = ensure_weights(weights_path)
    if os.path.exists(path):
        m.load_weights(path)
    else:
        raise FileNotFoundError(
            f"Weights file not found at '{path}'. For deployment, set WEIGHTS_URL in Streamlit Secrets "
            f"or host model.weights.h5 on Hugging Face / GitHub Releases."
        )
    return m


# ---------------------------------------------------------------- routing through the hierarchy
def soft_route(P):
    r, m, e, b, k = (P[n] for n in NODES); out = np.zeros((len(r), NC), 'float32')
    out[:, CI['MEL']] = r[:, 1] * m[:, 0]
    out[:, CI['BCC']] = r[:, 1] * m[:, 1] * e[:, 0]
    out[:, CI['SCC']] = r[:, 1] * m[:, 1] * e[:, 1]
    out[:, CI['DF']]  = r[:, 0] * b[:, 0]
    out[:, CI['NV']]  = r[:, 0] * b[:, 1]
    out[:, CI['AK']]  = r[:, 0] * b[:, 2] * k[:, 0]
    out[:, CI['BKL']] = r[:, 0] * b[:, 2] * k[:, 1]
    return out

def hard_route(P):
    rt, m, e, b, k = (P[n].argmax(1) for n in NODES)
    mal = np.where(m == 0, CI['MEL'], np.where(e == 0, CI['BCC'], CI['SCC']))
    ben = np.where(b == 0, CI['DF'], np.where(b == 1, CI['NV'], np.where(k == 0, CI['AK'], CI['BKL'])))
    return np.where(rt == 1, mal, ben)


# ---------------------------------------------------------------- prediction
def _views(rgb, n):
    # up to 8 test-time views: bit0 = h-flip, bit1 = v-flip, bit2 = rotate 90 (same as training-time evaluation)
    out = []
    for v in range(n):
        x = rgb
        if v & 1: x = x[:, ::-1]
        if v & 2: x = x[::-1, :]
        if v & 4: x = np.rot90(x, 1)
        out.append(np.ascontiguousarray(x))
    return np.stack(out).astype('float32')

def predict_nodes(model, rgb, meta=None, n_tta=None):
    n = CFG['n_tta'] if n_tta is None else n_tta
    views = _views(rgb, n)
    names = NODES + (['flat'] if FLAT_HEAD else [])
    
    # Process views with batch_size=1 sequentially to avoid OOM crashes on cloud containers
    meta_vec = (meta if meta is not None else meta_vector())[None].astype('float32') if USE_META else None
    
    accum = {k: 0.0 for k in names}
    for i in range(n):
        single_input = {'image': views[i:i+1]}
        if USE_META:
            single_input['meta'] = meta_vec
        out = model(single_input, training=False)
        for k, o in zip(names, out):
            accum[k] = accum[k] + np.asarray(o, 'float32')
            
    return {k: accum[k] / float(n) for k in names}

def classify(model, rgb, meta=None, routing=None, n_tta=None):
    # rgb: output of preprocess(); returns the predicted class, class probabilities and per-node probabilities
    P = predict_nodes(model, rgb, meta, n_tta=n_tta)
    routing = routing or CFG['routing']; soft = soft_route(P)[0]
    if routing == 'hard': label, probs = int(hard_route(P)[0]), soft
    elif routing == 'flat': probs = P['flat'][0]; label = int(probs.argmax())
    elif routing == 'blend': probs = (soft + P['flat'][0]) / 2; label = int(probs.argmax())
    else: probs = soft; label = int(soft.argmax())
    return {'label': CLASSES[label], 'label_name': CFG['class_full'][CLASSES[label]], 'routing': routing,
            'probs': {c: float(probs[i]) for i, c in enumerate(CLASSES)},
            'malignant_probability': float(P['root'][0, 1]),
            'nodes': {n: {NODE_CLASSES[n][j]: float(P[n][0, j]) for j in range(NODE_K[n])} for n in NODES}}
