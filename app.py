from flask import Flask, request, jsonify, render_template_string
import pickle
import cv2
import dlib
import numpy as np
from PIL import Image
import tempfile
import os
import base64
from io import BytesIO

app = Flask(__name__)

# ============================================================================
# LOAD MODEL & PREDICTOR
# ============================================================================

print("Loading model...")
with open('faceshape_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

svm_model = model_data['svm_model']
mlp_model = model_data['mlp_model']
knn_model = model_data['knn_model']
scaler = model_data['scaler']
lda = model_data['lda']
label_encoder = model_data['label_encoder']
face_shapes = model_data['face_shapes']

print("✓ Model loaded successfully!")

# Print versions untuk debugging
print(f"\n{'='*60}")
print("📦 Package Versions:")
print(f"  • Python: {cv2.__version__.split()[0] if hasattr(cv2, '__version__') else 'N/A'}")
print(f"  • OpenCV: {cv2.__version__}")
print(f"  • NumPy: {np.__version__}")
print(f"  • Dlib: {dlib.__version__ if hasattr(dlib, '__version__') else 'No version attr'}")
import sklearn
print(f"  • scikit-learn: {sklearn.__version__}")
print(f"{'='*60}\n")

# Load dlib predictor
predictor_path = "shape_predictor_81_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

print("✓ Dlib predictor loaded successfully!")

# ============================================================================
# PREPROCESSING & FEATURE EXTRACTION
# ============================================================================

def preprocess_image_for_landmark(image_path, target_size=600):
    """OpenCV-only preprocessing seperti di Colab"""
    img_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_cv is None:
        raise ValueError("Failed to read image")

    # Handle transparansi/alpha (4 channel → 3 channel BGR)
    if img_cv.ndim == 3 and img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
    
    # Handle grayscale (1 channel → 3 channel BGR)
    if img_cv.ndim == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    
    # Pastikan uint8 dan contiguous
    if img_cv.dtype != np.uint8:
        img_cv = img_cv.astype(np.uint8)
    if not img_cv.flags['C_CONTIGUOUS']:
        img_cv = np.ascontiguousarray(img_cv)

    # Resize keeping aspect ratio
    h, w = img_cv.shape[:2]
    max_dim = max(h, w)
    if max_dim > target_size:
        scale = target_size / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to grayscale (seperti Colab)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Pastikan gray juga uint8 dan contiguous
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    if not gray.flags['C_CONTIGUOUS']:
        gray = np.ascontiguousarray(gray)

    return img_cv, gray


def extract_81_facial_landmarks(image_input):
    """OpenCV Cascade untuk detect wajah, Dlib hanya untuk landmarks"""
    print(f"[DEBUG] extract_81_facial_landmarks input type: {type(image_input)}")
    
    # KONDISI 1: image_input = PATH
    if isinstance(image_input, str):
        print(f"[DEBUG] Processing file path: {image_input}")
        img_cv, gray = preprocess_image_for_landmark(image_input)
        print(f"[DEBUG] From file - img_cv.shape: {img_cv.shape}, gray.shape: {gray.shape}")
    
    # KONDISI 2: image_input = NUMPY ARRAY
    elif isinstance(image_input, np.ndarray):
        img = image_input
        print(f"[DEBUG] Processing numpy array - shape: {img.shape}, dtype: {img.dtype}, ndim: {img.ndim}")
        
        # Pastikan input uint8 dan contiguous dulu
        if img.dtype != np.uint8:
            print(f"[DEBUG] Converting dtype from {img.dtype} to uint8")
            img = img.astype(np.uint8)
        if not img.flags['C_CONTIGUOUS']:
            print(f"[DEBUG] Making array contiguous")
            img = np.ascontiguousarray(img)
        
        # Handle berbagai format numpy array
        if img.ndim == 2:  # Grayscale
            print(f"[DEBUG] Input is grayscale 2D")
            gray = img.copy()
            img_cv = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:  # Grayscale 3D
            print(f"[DEBUG] Input is grayscale 3D")
            gray = img[:,:,0].copy()
            img_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:  # RGBA
            print(f"[DEBUG] Input is RGBA")
            img_cv = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 3:  # BGR dari cv2.imdecode
            print(f"[DEBUG] Input is 3-channel (BGR)")
            img_cv = img.copy()
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            print(f"[ERROR] Unsupported image dimensions: {img.shape}")
            return None
        
        print(f"[DEBUG] After conversion - img_cv.shape: {img_cv.shape}, gray.shape: {gray.shape}")
        
        # Pastikan uint8 dan contiguous untuk output
        if img_cv.dtype != np.uint8:
            print(f"[DEBUG] Converting img_cv dtype to uint8")
            img_cv = img_cv.astype(np.uint8)
        if not img_cv.flags['C_CONTIGUOUS']:
            print(f"[DEBUG] Making img_cv contiguous")
            img_cv = np.ascontiguousarray(img_cv)
            
        if gray.dtype != np.uint8:
            print(f"[DEBUG] Converting gray dtype to uint8")
            gray = gray.astype(np.uint8)
        if not gray.flags['C_CONTIGUOUS']:
            print(f"[DEBUG] Making gray contiguous")
            gray = np.ascontiguousarray(gray)
    
    else:
        print(f"[ERROR] Unsupported input type: {type(image_input)}")
        return None
    
    # Validasi ukuran minimal
    if gray.shape[0] < 80 or gray.shape[1] < 80:
        print(f"[ERROR] Image too small: {gray.shape}")
        return None
    
    # ============================================
    # GUNAKAN DLIB DETECTOR UNTUK FACE DETECTION
    # ============================================
    print(f"[DEBUG] Using dlib face detector...")
    faces = detector(gray, 1)
    
    print(f"[DEBUG] Dlib detected {len(faces)} faces")
    
    if len(faces) == 0:
        print(f"[DEBUG] No faces detected")
        return None
    
    # Pilih face terbesar
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    print(f"[DEBUG] Selected face: width={face.width()}, height={face.height()}")
    
    # ============================================
    # GUNAKAN DLIB PREDICTOR UNTUK 81 LANDMARKS
    # ============================================
    print(f"[DEBUG] About to call dlib predictor for landmarks...")
    
    try:
        gray_safe = np.array(gray, dtype=np.uint8, copy=True, order='C')

        print("[DEBUG] Passing image directly to dlib")
        print("[DEBUG] dtype:", gray_safe.dtype,
              "C_CONTIGUOUS:", gray_safe.flags['C_CONTIGUOUS'],
              "OWNDATA:", gray_safe.flags['OWNDATA'],
              "shape:", gray_safe.shape)
        
        marks = predictor(gray_safe, face)
        coords = np.zeros((81, 2), dtype=float)
        
        for i in range(81):
            coords[i] = (marks.part(i).x, marks.part(i).y)
        
        print(f"[DEBUG] Successfully extracted {len(coords)} landmarks")
        return coords
    except Exception as e:
        import traceback
        print(f"[ERROR] Dlib predictor failed: {str(e)}")
        print(traceback.format_exc())
        return None


def extract_geometric_features_81(landmarks):
    """Ekstraksi 54 fitur - EXACT COPY dari Colab FINAL"""
    if landmarks is None or len(landmarks) != 81:
        return None

    def dist(i, j):
        return np.sqrt((landmarks[i,0]-landmarks[j,0])**2 + (landmarks[i,1]-landmarks[j,1])**2)

    def get_angle(p1, p2, p3):
        v1 = landmarks[p1] - landmarks[p2]
        v2 = landmarks[p3] - landmarks[p2]
        val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(val, -1.0, 1.0))

    features = []

    # === 1. NORMALISASI ROBUST (Multiple Reference) ===
    ref_width = dist(0, 16)  # Pipi
    ref_height_1 = dist(27, 8)  # Hidung ke dagu
    ref_height_2 = dist(71, 8)  # Dahi ke dagu (extended)
    ref_diag = dist(0, 8)  # Diagonal wajah

    for ref in [ref_width, ref_height_1, ref_height_2, ref_diag]:
        if ref == 0: ref = 1.0

    # === 2. ASPECT RATIOS (5 fitur) ===
    features.append(ref_width / ref_height_1)  # 1. Classic aspect
    features.append(ref_width / ref_height_2)  # 2. Extended aspect
    features.append(ref_height_2 / ref_width)  # 3. Inverse
    features.append(dist(68, 79) / ref_width)  # 4. Forehead ratio
    features.append(dist(6, 10) / ref_width)   # 5. Chin ratio

    # === 3. LEBAR BERTINGKAT (10 fitur - Normalisasi) ===
    width_points = [
        (68, 73),  # Dahi atas
        (17, 26),  # Alis
        (0, 16),   # Pipi
        (2, 14),   # Rahang atas
        (4, 12),   # Rahang tengah
        (6, 10),   # Dagu
        (36, 45),  # Mata
        (31, 35),  # Hidung
        (48, 54),  # Mulut
        (3, 13),   # Rahang bawah
    ]
    for p1, p2 in width_points:
        features.append(dist(p1, p2) / ref_width)

    # === 4. TINGGI BERTINGKAT (8 fitur) ===
    height_points = [
        (71, 27),  # Dahi ke hidung
        (27, 33),  # Hidung ke bibir
        (33, 8),   # Bibir ke dagu
        (71, 8),   # Total tinggi
        (19, 24),  # Tinggi alis
        (37, 41),  # Tinggi mata kiri
        (43, 47),  # Tinggi mata kanan
        (51, 57),  # Tinggi mulut
    ]
    for p1, p2 in height_points:
        features.append(dist(p1, p2) / ref_height_2)

    # === 5. SUDUT KRITIS (12 fitur) ===
    angle_points = [
        (0, 4, 8),   # Rahang kiri bawah
        (16, 12, 8), # Rahang kanan bawah
        (0, 2, 4),   # Rahang kiri atas
        (16, 14, 12),# Rahang kanan atas
        (6, 8, 10),  # Sudut dagu
        (2, 4, 6),   # Lengkung kiri
        (10, 12, 14),# Lengkung kanan
        (17, 21, 26),# Alis
        (36, 39, 42),# Mata
        (48, 51, 54),# Mulut
        (27, 30, 33),# Hidung
        (0, 8, 16),  # Segitiga wajah
    ]
    for p1, p2, p3 in angle_points:
        features.append(get_angle(p1, p2, p3))

    # === 6. CURVATURE (Kelengkungan - 6 fitur) ===
    def curve_measure(p1, p_mid, p2):
        line_dist = dist(p1, p2)
        actual_dist = dist(p1, p_mid) + dist(p_mid, p2)
        return (actual_dist - line_dist) / line_dist if line_dist > 0 else 0

    features.append(curve_measure(0, 4, 8))   # Rahang kiri
    features.append(curve_measure(16, 12, 8)) # Rahang kanan
    features.append(curve_measure(0, 8, 16))  # Dagu total
    features.append(curve_measure(17, 19, 21))# Alis kiri
    features.append(curve_measure(22, 24, 26))# Alis kanan
    features.append(curve_measure(36, 39, 45))# Mata

    # === 7. RASIO SILANG (8 fitur) ===
    jaw_w = dist(4, 12)
    forehead_w = dist(68, 79) if dist(68, 79) > 0 else dist(17, 26)
    chin_w = dist(6, 10)

    features.append(jaw_w / forehead_w)      # Jaw vs Forehead
    features.append(chin_w / jaw_w)          # Chin vs Jaw
    features.append(forehead_w / chin_w)     # Forehead vs Chin
    features.append(dist(2, 14) / jaw_w)     # Upper jaw ratio
    features.append(dist(36, 45) / ref_width)# Eye spread
    features.append(dist(31, 35) / ref_width)# Nose width
    features.append(dist(48, 54) / ref_width)# Mouth width
    features.append(dist(0, 16) / dist(71, 8))# Width/Height global

    # === 8. EXTENDED LANDMARKS (5 fitur) ===
    features.append(dist(68, 70) / ref_width)  # Dahi kiri
    features.append(dist(78, 79) / ref_width)  # Dahi kanan
    features.append(dist(70, 71) / ref_height_2) # Tinggi dahi kiri
    features.append(dist(78, 71) / ref_height_2) # Tinggi dahi kanan
    features.append(dist(75, 76) / ref_width)    # Lebar dahi tengah

    return np.array(features)


def predict_face_shape(image_path):
    try:
        # Extract landmarks
        landmarks = extract_81_facial_landmarks(image_path)
        if landmarks is None:
            return {'error': 'No face detected'}
        
        # Extract features
        features = extract_geometric_features_81(landmarks)
        if features is None:
            return {'error': 'Feature extraction failed'}
        
        # Scale and transform
        features_scaled = scaler.transform([features])
        features_lda = lda.transform(features_scaled)
        
        # Predict with all models
        pred_svm = svm_model.predict(features_lda)[0]
        pred_mlp = mlp_model.predict(features_lda)[0]
        pred_knn = knn_model.predict(features_lda)[0]
        
        shape_svm = label_encoder.inverse_transform([pred_svm])[0]
        shape_mlp = label_encoder.inverse_transform([pred_mlp])[0]
        shape_knn = label_encoder.inverse_transform([pred_knn])[0]
        
        # Voting
        from collections import Counter
        predictions = [shape_svm, shape_mlp, shape_knn]
        vote_count = Counter(predictions)
        final_prediction = vote_count.most_common(1)[0][0]
        
        # Visualize
        img_cv, _ = preprocess_image_for_landmark(image_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Draw landmarks
        for i in range(68):
            cv2.circle(img_rgb, tuple(landmarks[i].astype(int)), 3, (255, 0, 0), -1)
        for i in range(68, 81):
            cv2.circle(img_rgb, tuple(landmarks[i].astype(int)), 4, (0, 255, 0), -1)
        
        # Convert to base64
        buffered = BytesIO()
        Image.fromarray(img_rgb).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        result = {
            'final_prediction': final_prediction,
            'svm': shape_svm,
            'mlp': shape_mlp,
            'knn': shape_knn,
            'confidence': f"{vote_count[final_prediction]}/3",
            'image': img_str
        }
        
        # Cleanup
        os.unlink(image_path)
        
        return result
        
    except Exception as e:
        return {'error': str(e)}


# ============================================================================
# FLASK ROUTES
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Face Shape Classifier</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-bottom: 10px; }
        p { color: #666; }
        input[type="file"] { margin: 20px 0; padding: 10px; }
        button { padding: 12px 30px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px; font-size: 16px; }
        button:hover { background: #0056b3; }
        #result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 5px; }
        img { max-width: 100%; margin-top: 20px; border-radius: 5px; }
        .prediction { font-size: 24px; color: #28a745; font-weight: bold; }
        .models { margin: 15px 0; }
        .models li { margin: 5px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Face Shape Classifiers</h1>
        <p>Upload a clear frontal face photo to detect your face shape!</p>
        <p><strong>Supported Shapes:</strong> Oblong, Oval, Round, Square</p>
        
        <form id="uploadForm">
            <input type="file" id="imageFile" accept="image/*" required>
            <button type="submit">🔍 Detect Face Shape</button>
        </form>
        
        <div id="result"></div>
    </div>
    
    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('file', document.getElementById('imageFile').files[0]);
            
            document.getElementById('result').innerHTML = '<p>⏳ Processing image...</p>';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('result').innerHTML = `<p style="color:red">❌ Error: ${data.error}</p>`;
                } else {
                    document.getElementById('result').innerHTML = `
                        <div class="prediction">🎯 Your Face Shape: ${data.final_prediction}</div>
                        <div class="models">
                            <h3>📊 Model Predictions:</h3>
                            <ul>
                                <li>SVM Model: ${data.svm}</li>
                                <li>MLP Model: ${data.mlp}</li>
                                <li>KNN Model: ${data.knn}</li>
                            </ul>
                            <p>✅ Confidence: ${data.confidence} models agree</p>
                        </div>
                        <img src="data:image/png;base64,${data.image}" alt="Detected Landmarks">
                    `;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = `<p style="color:red">❌ Error: ${error.message}</p>`;
            }
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read upload into bytes
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({'error': 'Empty file content'}), 400

        # Decode dengan OpenCV (support semua format termasuk RGBA)
        np_buf = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        print(f"Upload decoded: shape={img.shape}, dtype={img.dtype}, ndim={img.ndim}")

        # Normalisasi ke BGR 3-channel (seperti preprocessing Colab)
        if img.ndim == 2:  # Grayscale
            img_cv = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:  # RGBA
            img_cv = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:  # BGR (sudah OK)
            img_cv = img
        else:
            return jsonify({'error': f'Unsupported image format: ndim={img.ndim}'}), 400

        # Pastikan uint8 + contiguous
        if img_cv.dtype != np.uint8:
            img_cv = img_cv.astype(np.uint8)
        if not img_cv.flags['C_CONTIGUOUS']:
            img_cv = np.ascontiguousarray(img_cv)

        # RESIZE dulu seperti preprocessing Colab (target_size=600)
        h, w = img_cv.shape[:2]
        max_dim = max(h, w)
        if max_dim > 600:
            scale = 600 / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized to: {img_cv.shape}")

        # LANGSUNG extract landmarks dari numpy array (seperti Colab)
        print("Extracting landmarks from array...")
        landmarks = extract_81_facial_landmarks(img_cv)
        cv2.imwrite("debug_upload.jpg", img_cv)
        if landmarks is None:
            return jsonify({'error': 'No face detected in image'}), 400
        
        print(f"Landmarks extracted: {landmarks.shape}")
        
        # Extract features
        features = extract_geometric_features_81(landmarks)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 400
        
        print(f"Features extracted: {len(features)} features")
        
        # Scale and transform
        features_scaled = scaler.transform([features])
        features_lda = lda.transform(features_scaled)
        
        # Predict with all models
        pred_svm = svm_model.predict(features_lda)[0]
        pred_mlp = mlp_model.predict(features_lda)[0]
        pred_knn = knn_model.predict(features_lda)[0]
        
        shape_svm = label_encoder.inverse_transform([pred_svm])[0]
        shape_mlp = label_encoder.inverse_transform([pred_mlp])[0]
        shape_knn = label_encoder.inverse_transform([pred_knn])[0]
        
        print(f"Predictions: SVM={shape_svm}, MLP={shape_mlp}, KNN={shape_knn}")
        
        # Voting
        from collections import Counter
        predictions = [shape_svm, shape_mlp, shape_knn]
        vote_count = Counter(predictions)
        final_prediction = vote_count.most_common(1)[0][0]
        
        # Visualize - convert BGR ke RGB
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Draw landmarks
        for i in range(68):
            cv2.circle(img_rgb, tuple(landmarks[i].astype(int)), 3, (255, 0, 0), -1)
        for i in range(68, 81):
            cv2.circle(img_rgb, tuple(landmarks[i].astype(int)), 4, (0, 255, 0), -1)
        
        # Convert to base64
        buffered = BytesIO()
        Image.fromarray(img_rgb).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        result = {
            'final_prediction': final_prediction,
            'svm': shape_svm,
            'mlp': shape_mlp,
            'knn': shape_knn,
            'confidence': f"{vote_count[final_prediction]}/3",
            'image': img_str
        }
        
        print(f"✓ Success! Prediction: {final_prediction}")
        return jsonify(result)

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print("="*60)
        print("ERROR OCCURRED:")
        print(error_detail)
        print("="*60)
        return jsonify({'error': f'{str(e)}'}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Flask server starting...")
    print("📍 Open: http://0.0.0.0:7860")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=7860, debug=False)
