import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 한글 폰트 설정 (옵션)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows: 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 로드 및 확인
file_path = './data/7.lpg_leakage.xlsx.csv'  # 데이터 파일 경로
data = pd.read_csv(file_path)  # CSV 파일 읽기

# 데이터 구조 확인
print(data.head())
print(data.info())

# 데이터 요약: 통계 정보 확인
print(data.describe())

# 결측값 처리: 결측값이 있다면 제거
data = data.dropna()

# 데이터 분포 시각화
plt.figure(figsize=(12, 8))
sns.countplot(x='LPG_Leakage', data=data)
plt.title('LPG 가스 누출 여부 분포')
plt.show()

# 변수 간 상관관계 시각화 (히트맵)
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('변수 간 상관관계')
plt.show()

# 독립 변수(X)와 종속 변수(y) 분리
X = data.drop(columns=['LPG_Leakage'])  # 독립 변수
y = data['LPG_Leakage']  # 종속 변수

# 데이터 스케일링: 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할: 훈련 데이터와 테스트 데이터로 분리 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 머신러닝 모델: 랜덤 포레스트 분류기
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {accuracy:.2f}")

# 분류 보고서 출력
print("\n분류 보고서:")
print(classification_report(y_test, y_pred))

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Leak', 'Leak'], yticklabels=['No Leak', 'Leak'])
plt.title("혼동 행렬")
plt.xlabel("예측 값")
plt.ylabel("실제 값")
plt.show()

# 피처 중요도 시각화 (랜덤 포레스트의 피처 중요도)
feature_importance = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features)
plt.title('특성 중요도')
plt.show()

