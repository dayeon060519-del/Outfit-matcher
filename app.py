import os
import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from flask_cors import CORS 
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# TensorFlow/Keras 라이브러리 (AI 모델 로드 및 이미지 처리)
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# --- 1. 환경 설정 및 경로 지정 ---
app = Flask(__name__)
CORS(app)

# 🚨🚨🚨 중요: 사용자님의 로컬 PC 경로로 수정되었습니다. 🚨🚨🚨
# 모든 AI 모델, 데이터 파일이 이 폴더 안에 있어야 합니다.
ROOT_PATH = os.getcwd()
# 🚨🚨🚨 새로 추가: 추천 아이템 이미지들이 저장된 폴더 경로 🚨🚨🚨
# 이 폴더 안에 recommendation_metadata.csv에 등록된 모든 이미지가 있어야 합니다.
#IMAGE_DIR = os.path.join(ROOT_PATH, "dataset_main") 


# 데이터 파일
CSV_FILE = os.path.join(ROOT_PATH, "recommendation_metadata.csv")
EMBEDDING_FILE = os.path.join(ROOT_PATH, "all_embeddings.npy")

# 모델 파일
CATEGORY_MODEL_PATH = os.path.join(ROOT_PATH, "classifier_category.h5")
COLOR_MODEL_PATH = os.path.join(ROOT_PATH, "classifier_color.h5")
STYLE_MODEL_PATH = os.path.join(ROOT_PATH, "classifier_style.h5")
# 🚨 새로 추가: 계절 모델 경로 🚨
SEASON_MODEL_PATH = os.path.join(ROOT_PATH, "classifier_season.h5")


# 매핑 파일 (AI 예측 결과 숫자를 실제 이름으로 변환하기 위해 필요)
CATEGORY_MAPPING_PATH = os.path.join(ROOT_PATH, "classifier_category_mapping.txt")
COLOR_MAPPING_PATH = os.path.join(ROOT_PATH, "classifier_color_mapping.txt")
STYLE_MAPPING_PATH = os.path.join(ROOT_PATH, "classifier_style_mapping.txt")
# 🚨 새로 추가: 계절 매핑 경로 🚨
SEASON_MAPPING_PATH = os.path.join(ROOT_PATH, "classifier_season_mapping.txt")


# --- 2. 전역 변수 설정 및 모델 로드 ---
global df, all_embeddings, mobile_net, category_model, color_model, style_model, season_model
global category_map, color_map, style_map, season_map # season_map 추가

def load_all_assets():
    """서버 시작 시 모든 데이터 및 AI 모델을 로드합니다."""
    
    global df, all_embeddings, mobile_net, category_model, color_model, style_model, season_model
    global category_map, color_map, style_map, season_map
    
    print("--- 데이터셋 로드 시작 ---")
    try:
        # 이 부분에서 ROOT_PATH를 사용하여 파일을 찾습니다.
        df = pd.read_csv(CSV_FILE)
        all_embeddings = np.load(EMBEDDING_FILE)
        print(f"데이터 로드 완료. 샘플 수: {len(df)}, 임베딩 Shape: {all_embeddings.shape}")
    except Exception as e:
        print(f"🚨 오류: 데이터 파일 로드 실패. 서버를 시작할 수 없습니다. ({e})")
        return False
        
    print("--- 특징 추출기 및 분류기 로드 시작 ---")
    try:
        # 1. MobileNetV2 특징 추출기 (임베딩 추출용)
        mobile_net = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        
        # 2. 학습된 자동 분류기 4종 로드
        category_model = load_model(CATEGORY_MODEL_PATH)
        color_model = load_model(COLOR_MODEL_PATH)
        style_model = load_model(STYLE_MODEL_PATH)
        # 🚨 계절 모델 로드 🚨
        season_model = load_model(SEASON_MODEL_PATH)

        # 3. 매핑 정보 로드
        def load_mapping(path):
            with open(path, 'r') as f:
                # 텍스트 파일에 저장된 문자열을 파이썬 딕셔너리로 변환
                mapping_str = f.read().strip()
                # eval 사용은 위험할 수 있으나, Colab에서 만든 파일이므로 가정
                class_indices = eval(mapping_str)
                # 인덱스(숫자)를 클래스(이름)로 변환하는 맵 생성
                return {v: k for k, v in class_indices.items()}

        category_map = load_mapping(CATEGORY_MAPPING_PATH)
        color_map = load_mapping(COLOR_MAPPING_PATH)
        style_map = load_mapping(STYLE_MAPPING_PATH)
        # 🚨 계절 매핑 로드 🚨
        season_map = load_mapping(SEASON_MAPPING_PATH)

        print("모든 AI 모델 및 매핑 정보 로드 완료.")
        return True
    except Exception as e:
        print(f"🚨 오류: AI 모델 로드 실패. 서버를 시작할 수 없습니다. 필요한 파일이 ROOT_PATH에 있는지 확인하세요. ({e})")
        return False

# --- 3. 핵심 로직 함수 (Notebook 로직 재사용) ---

# 색상 그룹 정의
color_groups = {
    'neutral': ['Black', 'White', 'Gray', 'Beige'],
    'cool': ['Blue', 'Green', 'Purple'],
    'warm': ['Red', 'Orange', 'Yellow', 'Pink']
}
def get_group(color):
    for group, colors in color_groups.items():
        if color in colors:
            return group
    return 'neutral'

# Notebook의 가중치
W_COLOR = 0.40
W_STYLE = 0.25
W_SEASON = 0.20
W_SIM = 0.15

def get_outfit_pair_score(query_attrs, target_idx):
    """추천 점수 계산 로직 (Notebook의 로직 사용)"""
    
    q = query_attrs # 쿼리 아이템 속성 (업로드 사진)
    t = df.iloc[target_idx] # 타겟 아이템 속성 (DB 아이템)

    # 1. 색상 점수 (w_color=0.40)
    q_group = get_group(q['color'])
    t_group = get_group(t['color'])
    
    if q_group == t_group: color_score = 1.0
    elif q_group == 'neutral' or t_group == 'neutral': color_score = 0.8
    else: color_score = 0.5
    
    # 2. 스타일 점수 (w_style=0.25)
    style_score = 1.0 if q['style'] == t['style'] else 0.5
    
    # 3. 계절 점수 (w_season=0.20)
    # 이제 q['season']은 실제 모델 예측값임
    season_score = 1.0 if q['season'] == t['season'] else 0.5
    
    # 4. 시각적 유사도 (w_sim=0.15) - 임베딩 추출 후 외부에서 계산됨
    
    # 여기서는 속성 점수 합계만 계산
    attribute_score = (
        W_COLOR * color_score +
        W_STYLE * style_score +
        W_SEASON * season_score 
    )
    
    return attribute_score, color_score, style_score, season_score

def extract_embedding(img_pil):
    """PIL 이미지 객체에서 MobileNetV2 임베딩을 추출합니다."""
    img = img_pil.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    embedding = mobile_net.predict(x, verbose=0)
    return embedding[0]

def predict_attributes(img_pil):
    """4가지 AI 모델을 사용하여 속성을 예측합니다."""
    
    # 이미지 전처리 (224x224, 정규화)
    img = img_pil.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    results = {}
    confidence = {}
    
    # 예측 및 변환 (Category)
    pred_cat = category_model.predict(x, verbose=0)[0]
    cat_index = np.argmax(pred_cat)
    results['category'] = category_map[cat_index]
    confidence['category'] = float(pred_cat[cat_index])

    # 예측 및 변환 (Color)
    pred_col = color_model.predict(x, verbose=0)[0]
    col_index = np.argmax(pred_col)
    results['color'] = color_map[col_index]
    confidence['color'] = float(pred_col[col_index])

    # 예측 및 변환 (Style)
    pred_sty = style_model.predict(x, verbose=0)[0]
    sty_index = np.argmax(pred_sty)
    results['style'] = style_map[sty_index]
    confidence['style'] = float(pred_sty[sty_index])
    
    # 🚨 수정 4: 계절 예측을 모델 기반으로 변경 🚨
    pred_sea = season_model.predict(x, verbose=0)[0]
    sea_index = np.argmax(pred_sea)
    results['season'] = season_map[sea_index]
    confidence['season'] = float(pred_sea[sea_index])

    return results, confidence

# --- 4. API 엔드포인트 ---

@app.route('/recommend', methods=['POST'])
def recommend_outfit():
    if 'file' not in request.files:
        return jsonify({"error": "No file part", "message": "파일을 첨부해주세요."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file", "message": "파일을 선택하지 않았습니다."}), 400

    try:
        # 1. 이미지 로드
        img_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # 2. **자동 속성 예측 (AI 브레인 사용)**
        query_attrs, confidence = predict_attributes(img_pil)
        
        # 3. 쿼리 아이템의 임베딩 추출
        query_embedding = extract_embedding(img_pil)
        
        # 4. 추천 로직 실행
        query_category = query_attrs['category']
        
        # 보완 카테고리 추천 로직
        
        # 1순위: 보완 카테고리 설정 (Top이면 Bottom, Bottom이면 Top)
        complementary_category = 'Bottom' if query_category == 'Top' else 'Top'
        
        candidate_indices = [] # 후보 인덱스 리스트 초기화
        available_categories = df['category'].unique().tolist()
        final_recommend_category = None
        guidance_category = ""
        
        if complementary_category in available_categories:
            # 1순위: 보완 카테고리가 DB에 존재하면 해당 카테고리 선택
            final_recommend_category = complementary_category
        else:
            # 2순위: 보완 카테고리가 없는 경우, 쿼리 카테고리를 제외한 다른 카테고리 탐색 (예: Outer)
            other_categories = [cat for cat in available_categories if cat != query_category]
            
            if other_categories:
                # 쿼리 카테고리가 아닌 다른 카테고리 중 첫 번째를 선택
                final_recommend_category = other_categories[0]
            else:
                # 3순위: DB에 보완할 아이템이 전혀 없는 경우 (Top만 있는 경우 등)
                guidance_category = f"DB에 {query_category} 외의 보완할 수 있는 다른 카테고리 아이템이 부족하여 추천 목록을 생성하기 어렵습니다."

        if final_recommend_category:
            candidate_indices = df[df['category'] == final_recommend_category].index.tolist()
        
        if not candidate_indices:
             # 후보 아이템이 없으면 빈 목록 반환 및 경고 처리
             top_k = []
             # 기존 정확도 기반 경고가 없으면 DB 관련 경고를 우선 사용합니다.
             # confidence 딕셔너리에 'season'이 추가되었으므로 min() 사용 가능
             min_confidence = min(confidence.values()) 
             guidance = guidance_category or f"AI 분석 정확도가 낮습니다 (최저 {(min_confidence*100):.0f}%). 옷이 잘 보이도록 다른 각도/배경에서 다시 시도해 보세요."
        
             return jsonify({
                "status": "success",
                "query_attributes": query_attrs,
                "confidence": confidence,
                "guidance": guidance, 
                "recommendations": top_k
            })
            
        scores = {}
        
        # 모든 후보 아이템과 점수 계산
        for target_idx in candidate_indices:
            # 속성 기반 점수 (W_COLOR + W_STYLE + W_SEASON)
            attr_score, c_s, s_s, sea_s = get_outfit_pair_score(query_attrs, target_idx)
            
            # 시각적 유사도 점수 (W_SIM)
            target_embedding = all_embeddings[target_idx].reshape(1, -1)
            query_emb_reshaped = query_embedding.reshape(1, -1)
            
            # 코사인 유사도 계산 후 0.0 ~ 1.0으로 스케일링
            similarity = cosine_similarity(query_emb_reshaped, target_embedding)[0][0]
            similarity_score = (similarity + 1) / 2
            
            # 최종 점수 합산
            total_score = attr_score + (W_SIM * similarity_score)
            
            # 추천 아이템의 season, style 정보를 scores에 추가
            scores[target_idx] = {
                'total_score': float(total_score),
                'filename': df.iloc[target_idx]['filename'],
                'category': df.iloc[target_idx]['category'],
                'color': df.iloc[target_idx]['color'],
                'season': df.iloc[target_idx]['season'],
                'style': df.iloc[target_idx]['style'],
                'details': {
                    'color_score': float(c_s), 'style_score': float(s_s), 'season_score': float(sea_s), 'sim_score': float(similarity_score)
                }
            }
        
        # 점수 기준 상위 3개 추출
        top_k = sorted(scores.values(), key=lambda x: x['total_score'], reverse=True)[:3]

        # 5. 피드백 및 에러 처리
        min_confidence = min(confidence.values())
        guidance = guidance_category or "" # DB 관련 경고가 있으면 사용
        
        if min_confidence < 0.65 and not guidance: # 정확도 65% 미만일 때 경고
            guidance = f"AI 분석 정확도가 낮습니다 (최저 {(min_confidence*100):.0f}%). 옷이 잘 보이도록 다른 각도/배경에서 다시 시도해 보세요."
        
        return jsonify({
            "status": "success",
            "query_attributes": query_attrs,
            "confidence": confidence,
            "guidance": guidance,
            "recommendations": top_k
        })

    except Exception as e:
        # Error states
        return jsonify({
            "error": f"Internal Server Error: {str(e)}", 
            "message": "서버 내부 처리 중 문제가 발생했습니다. 관리자에게 문의하세요.",
            "error_type": "MODEL_INFERENCE_FAILED"
        }), 500

@app.route('/image/<filename>')
def serve_image(filename):
    """
    클라이언트(프론트엔드)가 요청하는 이미지를 IMAGE_DIR에서 찾아서 전송하는 엔드포인트
    """
    try:
        # IMAGE_DIR 폴더에서 filename에 해당하는 파일을 찾아서 전송
        return send_from_directory(IMAGE_DIR, filename)
    except FileNotFoundError:
        # 파일이 없을 경우 404 에러 반환
        return jsonify({"error": "Image not found", "message": f"파일 {filename}을 찾을 수 없습니다."}), 404


@app.route('/')
def home():
    from flask import send_file
    # render_template을 사용하려면 index.html을 'templates' 폴더에 넣어야 합니다.
    # 가장 간단하게는, 루트에 있는 index.html을 바로 보냅니다.
    return send_file('index.html')

if __name__ == '__main__':
    # 서버 시작 전에 모든 AI 모델과 데이터를 로드
    if load_all_assets():
        print("✅ 모든 에셋 로드 완료. 서버를 시작합니다.")
        # 배포 시에는 host와 port를 변경하고 debug=False로 설정해야 합니다.
        app.run(host='0.0.0.0', port=5000, debug=True)