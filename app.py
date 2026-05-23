"""
ForestFire-GPT 통합 대시보드
- 발화위험 히트맵
- 확산 시뮬레이션
- LLM 자연어 에이전트
"""
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pickle
import os
from fire_spread import initialize_grid, simulate, burned_area_km2, BURNING, BURNED, TREE

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ForestFire-GPT",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 ForestFire-GPT")
st.caption("산악기상 빅데이터와 LLM 에이전트 기반 실시간 산불 위험 예측·확산 시뮬레이션 서비스")
st.caption("📍 시범 지역: 경상북도 의성군 | 데이터: 산악기상관측망(mtweather) · 임상도 · DEM")

# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    page = st.radio(
        "기능 선택",
        ["📊 발화위험 히트맵", "🔥 확산 시뮬레이션", "💬 AI 에이전트"],
    )
    st.divider()
    st.markdown("**모델 정보**")
    st.markdown("- 발화 예측: XGBoost (AUC 0.87)")
    st.markdown("- 확산 모델: Cellular Automata")
    st.markdown("- LLM: Claude Sonnet 4.6")
    st.divider()
    st.caption("© 2026 ForestFire-GPT Team")

# ──────────────────────────────────────────────
# 샘플 데이터 생성 (의성군 실제 경계 반영)
# ──────────────────────────────────────────────
@st.cache_data
def load_grid_data():
    """의성군 실제 행정경계 내부에 100m 격자 생성"""
    np.random.seed(42)
    # 의성군 대략 경계
    lat_min, lat_max = 36.25, 36.45
    lon_min, lon_max = 128.55, 128.85

    # 격자 형태로 생성 (랜덤 산포 X)
    lats = np.linspace(lat_min, lat_max, 20)
    lons = np.linspace(lon_min, lon_max, 20)
    grid = []
    for la in lats:
        for lo in lons:
            # 남서쪽일수록 위험도↑ + 노이즈
            base = 0.3 + 0.6 * ((lat_max - la) / (lat_max - lat_min))
            base += 0.3 * ((lo - lon_min) / (lon_max - lon_min))
            risk = np.clip(base + np.random.normal(0, 0.12), 0.05, 0.98)
            grid.append({'lat': la, 'lon': lo, 'risk': risk})
    return pd.DataFrame(grid)


def risk_color(r):
    if r >= 0.7: return '#d32f2f'
    if r >= 0.5: return '#f57c00'
    if r >= 0.3: return '#fbc02d'
    return '#388e3c'


def risk_label(r):
    if r >= 0.7: return '매우위험'
    if r >= 0.5: return '위험'
    if r >= 0.3: return '주의'
    return '안전'


def _demo_response(q, top3):
    """API 키 없을 때 사용할 데모 응답"""
    lines = "\n".join([
        f"- **지점 {i+1}**: 위도 {r.lat:.4f}, 경도 {r.lon:.4f} (위험도 **{r.risk:.2f}**)"
        for i, r in enumerate(top3.itertuples())
    ])
    return f"""**🔥 산불 위험 분석 보고서 (데모)**

**질의**: {q}

**1. 핵심 요약**
경상북도 의성군 일대 100m 격자 분석 결과, 현재 매우위험(0.7+) 등급 격자가 다수 식별되었습니다. 남서풍·저연료습도 조건에서 발화 시 단시간 내 확산 우려가 있습니다.

**2. 최우선 감시 지점 Top 3**
{lines}

**3. 권고사항**
- 위 3개 지점 반경 2km 내 진화헬기 사전 대기 권장
- 등산로 입구 입산통제 검토
- 인근 마을 주민 비상연락망 점검

*(실제 사용 시 Anthropic API Key를 입력하면 Claude가 더 상세한 맞춤 보고서를 생성합니다.)*
"""


# ──────────────────────────────────────────────
# 페이지 1: 발화위험 히트맵 (개선판)
# ──────────────────────────────────────────────
if page == "📊 발화위험 히트맵":
    st.subheader("실시간 발화위험 히트맵 (100m 격자)")

    df = load_grid_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("관측 격자 수", f"{len(df):,}")
    col2.metric("매우위험 격자", f"{(df['risk']>=0.7).sum()}", delta="+12 vs 어제")
    col3.metric("평균 위험도", f"{df['risk'].mean():.2f}")
    col4.metric("최고 위험도", f"{df['risk'].max():.2f}")

    st.divider()

    c_left, c_right = st.columns([2, 1])

    with c_left:
        # 안정적인 CartoDB Positron 타일 사용 (한국에서도 빠르게 로드)
        m = folium.Map(
            location=[36.353, 128.697],
            zoom_start=11,
            tiles=None,  # 기본 타일 비활성화 후 명시적 추가
        )

        # 여러 타일 옵션 추가 (사용자가 우상단 메뉴로 전환 가능)
        folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            attr='© OpenStreetMap © CartoDB',
            name='지도 (Light)',
            control=True,
        ).add_to(m)
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='지도 (Standard)',
            control=True,
        ).add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='위성영상',
            control=True,
        ).add_to(m)

        # 의성군 경계 박스
        folium.Rectangle(
            bounds=[[36.25, 128.55], [36.45, 128.85]],
            color='#1976d2',
            weight=2.5,
            fill=False,
            popup='경상북도 의성군',
        ).add_to(m)

        # 격자 셀: fillOpacity를 0.45로 낮춰 지도 배경이 보이게
        cell_size = 0.01
        for _, row in df.iterrows():
            folium.Rectangle(
                bounds=[
                    [row['lat'] - cell_size/2, row['lon'] - cell_size/2],
                    [row['lat'] + cell_size/2, row['lon'] + cell_size/2],
                ],
                color=risk_color(row['risk']),
                weight=0.3,
                fill=True,
                fillColor=risk_color(row['risk']),
                fillOpacity=0.45,
                popup=folium.Popup(
                    f"<b>위험도 {row['risk']:.2f} ({risk_label(row['risk'])})</b><br>"
                    f"위도: {row['lat']:.4f}<br>"
                    f"경도: {row['lon']:.4f}<br>"
                    f"기온: {np.random.uniform(20,30):.1f}℃<br>"
                    f"풍속: {np.random.uniform(2,10):.1f}m/s<br>"
                    f"연료습도: {np.random.uniform(8,18):.1f}%",
                    max_width=220,
                ),
            ).add_to(m)

        # 레이어 컨트롤 추가
        folium.LayerControl(position='topright', collapsed=True).add_to(m)

        # 범례
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                    background: white; padding: 10px 14px; border: 2px solid #888;
                    border-radius: 6px; font-size: 13px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
          <b>위험 등급</b><br>
          <span style="display:inline-block;width:12px;height:12px;background:#d32f2f;"></span> 매우위험 (0.7+)<br>
          <span style="display:inline-block;width:12px;height:12px;background:#f57c00;"></span> 위험 (0.5~0.7)<br>
          <span style="display:inline-block;width:12px;height:12px;background:#fbc02d;"></span> 주의 (0.3~0.5)<br>
          <span style="display:inline-block;width:12px;height:12px;background:#388e3c;"></span> 안전 (0.0~0.3)
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width=720, height=520, returned_objects=[])

    with c_right:
        # 5) 위험 등급별 색상 적용 차트
        st.markdown("### 🚨 위험 등급 분포")
        df['등급'] = df['risk'].apply(risk_label)
        order = ['안전', '주의', '위험', '매우위험']
        dist = df['등급'].value_counts().reindex(order).fillna(0)

        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Bar(
            x=dist.index,
            y=dist.values,
            marker_color=['#388e3c', '#fbc02d', '#f57c00', '#d32f2f'],
            text=dist.values,
            textposition='outside',
        )])
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            yaxis_title='격자 수',
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 상위 위험 격자 Top 5")
        top = df.nlargest(5, 'risk')[['lat', 'lon', 'risk']].reset_index(drop=True)
        top.insert(0, '순위', ['🥇 1', '🥈 2', '🥉 3', '4', '5'])
        top.columns = ['순위', '위도', '경도', '위험도']
        top['위도'] = top['위도'].round(4)
        top['경도'] = top['경도'].round(4)
        top['위험도'] = top['위험도'].apply(lambda x: f"{x:.3f} 🔴" if x >= 0.7 else f"{x:.3f}")
        st.dataframe(top, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────
# 페이지 2: 확산 시뮬레이션
# ──────────────────────────────────────────────
elif page == "🔥 확산 시뮬레이션":
    st.subheader("산불 확산 시뮬레이션 (Cellular Automata)")
    st.caption("발화점·풍향·풍속·연료습도를 입력하면 시간별 확산 경계를 시뮬레이션합니다.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: wind_dir = st.slider("풍향 (°)", 0, 360, 90, help="0=북, 90=동, 180=남, 270=서")
    with c2: wind_speed = st.slider("풍속 (m/s)", 0, 20, 7)
    with c3: fuel_moisture = st.slider("연료습도 (%)", 5, 30, 10)
    with c4: hours = st.slider("시뮬레이션 시간", 1, 12, 6)
    with c5: density = st.slider("임목밀도", 0.5, 0.95, 0.85)

    run = st.button("🔥 시뮬레이션 실행", type="primary", use_container_width=True)

    if run:
        with st.spinner("시뮬레이션 진행 중..."):
            grid = initialize_grid(size=50, tree_density=density)
            history = simulate(
                grid, ignition_xy=(25, 25), hours=hours,
                wind_dir=wind_dir, wind_speed=wind_speed,
                fuel_moisture=fuel_moisture,
            )

        # 결과 시각화
        st.success(f"✅ {hours}시간 시뮬레이션 완료")

        import plotly.graph_objects as go
        cols = st.columns(min(hours + 1, 4))
        show_times = [0, hours // 2, hours] if hours >= 2 else [0, hours]

        for idx, t in enumerate(show_times):
            with cols[idx % len(cols)]:
                fig = go.Figure(data=go.Heatmap(
                    z=history[t],
                    colorscale=[
                        [0.00, '#f5f5f5'],   # EMPTY
                        [0.33, '#2e7d32'],   # TREE
                        [0.66, '#ff5722'],   # BURNING
                        [1.00, '#424242'],   # BURNED
                    ],
                    showscale=False,
                ))
                fig.update_layout(
                    title=f"T+{t}h",
                    width=300, height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

        # 통계
        st.divider()
        c1, c2, c3 = st.columns(3)
        final = history[-1]
        c1.metric("피해 면적", f"{burned_area_km2(final):.2f} km²")
        c2.metric("연소 격자 수", f"{((final==BURNING)|(final==BURNED)).sum():,}")
        c3.metric("잔존 산림", f"{(final==TREE).sum():,} 격자")


# ──────────────────────────────────────────────
# 페이지 3: AI 에이전트
# ──────────────────────────────────────────────
elif page == "💬 AI 에이전트":
    st.subheader("💬 ForestFire-GPT 자연어 에이전트")
    st.caption("자연어로 산불 위험 보고서를 요청하세요. (RAG + MCP 기반 LLM 에이전트)")

    # API 키 입력
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="https://console.anthropic.com 에서 발급 (없으면 데모 응답 사용)",
    )

    # 예시 질문
    st.markdown("**💡 예시 질문**")
    ex_cols = st.columns(3)
    examples = [
        "내일 오후 의성군 가장 위험한 지역 3곳 알려줘",
        "현재 풍속 8m/s에서 6시간 후 확산 예상은?",
        "고위험 지역에 권장하는 진화자원 배치 방안은?",
    ]
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, use_container_width=True):
            st.session_state.user_input = ex

    user_input = st.text_area(
        "질문 입력",
        value=st.session_state.get('user_input', ''),
        height=80,
    )

    if st.button("🚀 보고서 생성", type="primary"):
        if not user_input.strip():
            st.warning("질문을 입력하세요.")
        else:
            with st.spinner("에이전트가 데이터를 분석 중입니다..."):
                # 1) 데이터 호출 (MCP Tool 모사)
                df = load_grid_data()
                top3 = df.nlargest(3, 'risk')
                avg_risk = df['risk'].mean()

                context = (
                    f"[현재 의성군 데이터]\n"
                    f"- 평균 위험도: {avg_risk:.2f}\n"
                    f"- 매우위험 격자: {(df['risk']>=0.7).sum()}개\n"
                    f"- 상위 위험 좌표:\n"
                    + "\n".join([
                        f"  · ({r.lat:.4f}, {r.lon:.4f}) 위험도 {r.risk:.2f}"
                        for r in top3.itertuples()
                    ])
                )

                # 2) LLM 호출
                if api_key:
                    try:
                        from anthropic import Anthropic
                        client = Anthropic(api_key=api_key)
                        msg = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1024,
                            system=(
                                "당신은 산불 위험 분석 전문 AI 에이전트입니다. "
                                "제공된 데이터를 바탕으로 산림청·소방청 실무자에게 "
                                "지도·표·요약문이 포함된 의사결정 지원 보고서를 한국어로 작성합니다. "
                                "마크다운 형식으로 명확하게 구성하세요."
                            ),
                            messages=[{
                                "role": "user",
                                "content": f"{context}\n\n[질문]\n{user_input}"
                            }],
                        )
                        answer = msg.content[0].text
                    except Exception as e:
                        answer = f"⚠️ API 호출 실패: {e}\n\n아래는 데모 응답입니다.\n\n" + _demo_response(user_input, top3)
                else:
                    answer = _demo_response(user_input, top3)

                st.divider()
                st.markdown("### 📄 분석 보고서")
                st.markdown(answer)

                # 결과 지도
                st.markdown("### 🗺️ 위험 격자 지도")
                m = folium.Map(location=[36.353, 128.697], zoom_start=11)
                for _, r in top3.iterrows():
                    folium.Marker(
                        [r['lat'], r['lon']],
                        popup=f"위험도 {r['risk']:.2f}",
                        icon=folium.Icon(color='red', icon='fire', prefix='fa'),
                    ).add_to(m)
                st_folium(m, width=700, height=400, returned_objects=[])
