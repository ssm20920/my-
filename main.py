import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")

# 데이터 불러오기 함수
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_and_process_data():
    # 데이터 로드 (인코딩: UTF-8)
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜 데이터 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일수 및 평균기온 계산
    yearly_summary = (
        df.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 필터링: 2025년 이하 & 관측일수 300일 이상인 해만 추출
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].copy()

    # 독립변수: 1908년부터 지난 연수 (Year - 1908)
    filtered_df["지난연수"] = filtered_df["연도"] - 1908

    return filtered_df


# 데이터 가공
df = load_and_process_data()

# 1차 선형 회귀 계산 (1908년 기준 지난 연수)
X = df["지난연수"].values
Y = df["평균기온"].values

slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 계산
corr = np.corrcoef(df["연도"], Y)[0, 1]

# 정보 출력 (직선을 만든 해의 개수, 시작 연도, 끝 연도)
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 수", f"{len(df)}개 해")
col2.metric("시작 연도", f"{int(df['연도'].min())}년")
col3.metric("끝 연도", f"{int(df['연도'].max())}년")
col4.metric("상관계수 (연도-기온)", f"{corr:.4f}")

st.markdown("---")

# 예측용 슬라이더 설정
selected_year = st.slider(
    "예측하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 선택한 연도의 예상 기온 계산
# 지난 연수 = selected_year - 1908
predicted_temp = slope * (selected_year - 1908) + intercept

# 예상 기온 크게 표시
st.subheader(f"🔮 {selected_year}년 예상 연평균 기온")
st.markdown(
    f"<h1 style='text-align: center; color: #FF4B4B;'>{predicted_temp:.2f} °C</h1>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Plotly 그래프 시각화
fig = go.Figure()

# 1. 실제 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(color="blue", size=7, opacity=0.7),
    )
)

# 2. 회귀 직선 (1900~2100년 전체 범위 표현)
years_range = np.arange(1900, 2101)
trend_y = slope * (years_range - 1908) + intercept

fig.add_trace(
    go.Scatter(
        x=years_range,
        y=trend_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="red", width=2),
    )
)

# 3. 슬라이더로 선택한 연도 강조 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers+text",
        name="선택 연도 예측값",
        marker=dict(color="orange", size=14, symbol="star"),
        text=[f"{predicted_temp:.2f}°C"],
        textposition="top center",
    )
)

# 그래프 레이아웃 설정
fig.update_layout(
    title="서울 연도별 연평균 기온 및 선형 회귀 추세선",
    xaxis_title="연도",
    yaxis_title="평균 기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    xaxis=dict(range=[1895, 2105]),
)

st.plotly_chart(fig, use_container_width=True)
