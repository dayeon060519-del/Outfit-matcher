import os
import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from flask_cors import CORS 
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import json

# TensorFlow/Keras 라이브러리 (AI 모델 로드 및 이미지 처리)
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# --- 1. 환경 설정 및 경로 지정 ---
app = Flask(__name__)
CORS(app)

# 🚨🚨🚨 중요: Render 서버에 맞게 절대 경로 대신 현재 파일의 디렉토리로 ROOT_PATH 설정 🚨🚨🚨
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# 🚨🚨🚨 새로 추가: 추천 아이템 이미지들이 저장된 폴더 경로 🚨🚨🚨
IMAGE_DIR = os.path.join(ROOT_PATH, "dataset_main") 

# 데이터 파일
CSV_FILE = os.path.join(ROOT_PATH, "recommendation_metadata.csv")
EMBEDDING_FILE = os.path.join(ROOT_PATH, "all_embeddings.npy")

# 모델 파일 (사용자님 프로젝트에 맞춰 경로 수정 필요)
# Render에 업로드한 경로를 기준으로 설정해 주세요.
CATEGORY_MODEL_PATH = os.path.join(ROOT_PATH, "classifier_category.h5")
COLOR_MODEL_PATH = os.path.join(ROOT_PATH, "classifier_color.h5")
STYLE_MODEL_PATH = os.path.join(ROOT_PATH, "classifier_style.h5")
SEASON_MODEL_PATH = os.path.join(ROOT_PATH, "classifier_season.h5")
FEATURE_EXTRACTOR_PATH = os.path.join(ROOT_PATH, "MobileNetV2.h5")

# --- 2. 전역 변수 초기화 (NameError 방지) ---
# 이 변수들이 Flask 라우트에서 사용될 수 있도록 None으로 초기화합니다.
category_model = None
color_model = None
style_model = None
season_model = None
feature_extractor = None 
df_metadata = None
all_embeddings = None
LABEL_MAPS = None

# --- 3. 상수 정의 ---
# 실제 모델 학습 시 사용된 클래스 이름 리스트로 대체해야 합니다.
# (이전에 제공된 더미 데이터 사용)
CLASSES = {
    'category': ['Top', 'Bottom', 'Outerwear', 'Dress', 'Shoes', 'Accessory'],
    'color': ['Black', 'White', 'Red', 'Blue', 'Green', 'Light Gray', 'Dark Gray', 'Beige', 'Brown', 'Yellow', 'Pink', 'Orange', 'Purple', 'Mint', 'Navy', 'Sky Blue', 'Khaki'],
    'style': ['Casual', 'Street', 'Business', 'Formal', 'Sporty', 'Romantic', 'Vintage'],
    'season': ['Spring', 'Summer', 'Fall', 'Winter']
}

# --- 4. 모델 및 데이터 로드 함수 (서버 시작 시 단 1회 실행) ---
def load_all_assets():
    """ 모든 AI 모델과 데이터를 메모리에 로드 """
    print(f"ROOT_PATH: {ROOT_PATH}")
    global category_model, color_model, style_model, season_model, feature_extractor, df_metadata, all_embeddings, LABEL_MAPS
    
    try:
        # 1. 메타데이터 로드
        df_metadata = pd.read_csv(CSV_FILE)
        print(f"메타데이터 로드 완료. 총 {len(df_metadata)}개 아이템.")

        # 2. 임베딩 데이터 로드
        all_embeddings = np.load(EMBEDDING_FILE)
        print(f"임베딩 데이터 로드 완료. 형태: {all_embeddings.shape}")
        
        # 3. Keras 모델 로드
        category_model = load_model(CATEGORY_MODEL_PATH)
        color_model = load_model(COLOR_MODEL_PATH)
        style_model = load_model(STYLE_MODEL_PATH)
        season_model = load_model(SEASON_MODEL_PATH)
        
        # 4. 특징 추출기 로드 (MobileNetV2)
        # MobileNetV2.h5를 로드하거나, weights='imagenet'으로 MobileNetV2 기본 모델 사용
        # 여기서는 고객님의 MobileNetV2.h5 경로를 사용합니다.
        feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
        
        # 5. 모델 라벨 맵 (선택 사항: 필요한 경우 로드)
        LABEL_MAPS = CLASSES # 모델의 출력 순서와 라벨이 일치한다고 가정합니다.

        print("✅ 모든 에셋 로드 완료. 서버를 시작합니다.")
        return True

    except Exception as e:
        print(f"🚨 Fatal Error: 모델 또는 데이터 로드 실패. {e}")
        # Render에서 이 오류가 발생하면 서버가 즉시 종료되므로 메모리 부족 문제가 아니라는 것을 알 수 있습니다.
        return False

# 서버 시작 시 로드 함수 실행
load_all_assets() 

# --- 5. 이미지 전처리 함수 ---
def preprocess_query_image(image_bytes):
    """ 이미지 바이트를 받아 MobileNetV2 입력 형태(224x224)로 전처리 """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # MobileNetV2 전용 전처리 함수
    return preprocess_input(img_array)

# --- 6. AI 속성 예측 함수 ---
def predict_attributes(processed_img):
    """ 전처리된 이미지 배열을 기반으로 4가지 속성 예측 및 가장 높은 확률 반환 """
    
    # NameError 방지를 위해 로드된 모델 객체를 사용하기 전에 다시 한 번 검증
    if category_model is None:
        raise ValueError("AI 모델이 로드되지 않았습니다.")
    
    predictions = {
        'category': category_model.predict(processed_img)[0],
        'color': color_model.predict(processed_img)[0],
        'style': style_model.predict(processed_img)[0],
        'season': season_model.predict(processed_img)[0]
    }
    
    results = {}
    confidence = {}
    
    for key, pred in predictions.items():
        # 가장 높은 확률을 가진 인덱스 찾기
        max_index = np.argmax(pred)
        # 해당 라벨과 신뢰도(확률) 저장
        results[key] = LABEL_MAPS[key][max_index]
        confidence[key] = float(pred[max_index])
        
    return results, confidence

# --- 7. 코디 추천 핵심 로직 함수 ---
def recommend_outfits(query_vector, query_attrs, df, k=10):
    """ 쿼리 벡터와 속성을 기반으로 상위 K개 아이템 추천 """
    
    # 1. 유사도 계산 (코사인 유사도)
    similarities = cosine_similarity(query_vector, all_embeddings)[0]
    
    # 2. 메타데이터에 유사도 점수 추가
    df['similarity_score'] = similarities
    
    # 3. 속성 매칭 점수 계산
    # 카테고리/색상/스타일/계절이 일치하면 추가 점수 부여
    df['attribute_score'] = 0.0
    
    # 각 속성별 일치 점수
    attr_scores = defaultdict(float)

    # 색상 일치 점수 (0.4점)
    df.loc[df['color'] == query_attrs['color'], 'attribute_score'] += 0.4
    df.loc[df['color'] == query_attrs['color'], 'color_score'] = 0.4
    
    # 스타일 일치 점수 (0.25점)
    df.loc[df['style'] == query_attrs['style'], 'attribute_score'] += 0.25
    df.loc[df['style'] == query_attrs['style'], 'style_score'] = 0.25
    
    # 시각적 유사도 점수는 이미 similarity_score에 저장됨 (최대 0.5)
    
    # 계절 일치 점수 (0.1점)
    df.loc[df['season'] == query_attrs['season'], 'attribute_score'] += 0.1
    df.loc[df['season'] == query_attrs['season'], 'season_score'] = 0.1

    # 최종 점수 계산: 시각적 유사도(Max 0.5) + 속성 일치 점수(Max 0.75)
    df['total_score'] = df['similarity_score'] + df['attribute_score']
    
    # 4. 카테고리 필터링: 입력된 옷과 동일한 카테고리는 제외
    filtered_df = df[df['category'] != query_attrs['category']]

    # 5. 최종 점수를 기준으로 정렬 및 상위 K개 선택
    top_k_results = filtered_df.sort_values(by='total_score', ascending=False).head(k)
    
    # 결과를 JSON 형태로 변환
    recommendations_list = []
    for index, row in top_k_results.iterrows():
        recommendations_list.append({
            'filename': row['filename'],
            'category': row['category'],
            'color': row['color'],
            'style': row['style'],
            'season': row['season'],
            'total_score': row['total_score'],
            'details': {
                'sim_score': row['similarity_score'],
                # 점수 기록이 없으면 0으로 처리 (일치하지 않았을 경우)
                'color_score': row.get('color_score', 0.0), 
                'style_score': row.get('style_score', 0.0),
                'season_score': row.get('season_score', 0.0)
            }
        })
        
    return recommendations_list


# ==========================================================
# FLASK API 라우트
# ==========================================================

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    이미지를 받아 AI 분석 및 코디 추천 결과를 반환하는 엔드포인트
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # 1. 이미지 전처리
        image_bytes = file.read()
        processed_img = preprocess_query_image(image_bytes)
        
        # 2. 특징 추출 (Feature Extraction)
        query_vector = feature_extractor.predict(processed_img)
        
        # 3. 속성 예측 (Attribute Prediction)
        query_attrs, confidence = predict_attributes(processed_img)
        
        # 4. 코디 추천 실행
        top_k = recommend_outfits(query_vector, query_attrs, df_metadata.copy())

        # 5. 가이드 메시지 생성
        min_confidence = min(confidence.values())
        guidance = ""
        if min_confidence < 0.75:
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
        print(f"Fatal Error in /recommend ---: {str(e)}")
        # 🚨 여기서 NameError가 발생한다면, load_all_assets()가 메모리 부족으로 인해 실패했거나, 
        # 혹은 변수 설정이 잘못된 것입니다.
        return jsonify({
            "error": f"Internal Server Error: {str(e)}", 
            "message": "서버 내부 처리 중 문제가 발생했습니다. (모델 로드 실패 가능성)",
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
    # index.html 파일을 클라이언트에게 전송
    return send_file(os.path.join(ROOT_PATH, 'index.html'))

if __name__ == '__main__':
    # Render 환경에서는 이 부분이 실행되지 않습니다. (gunicorn 등이 실행)
    # 로컬 테스트 용도로만 사용됩니다.
    app.run(host='0.0.0.0', port=5000)