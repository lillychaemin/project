# -*- coding: utf-8 -*-
"""
전국 고령화 지도 (Streamlit 앱)

- 시군구별 65세 이상 인구 비율(고령화율)을 5단계 색으로 칠한 단계구분도입니다.
- 인구 데이터와 경계 데이터는 GitHub에서 바로 내려받아 사용합니다.
- 지역은 '이름'이 아니라 '코드(5자리)'로 맞춥니다.
  (예: '남구'는 여러 시도에 있어서 이름으로 맞추면 어긋납니다.)
"""

import io
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────
# 1. 설정값 (바꾸고 싶으면 여기만 고치면 됩니다)
# ─────────────────────────────────────────────
POP_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/population_yearly.csv.gz"
GEO_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/boundaries/sigungu_kr.geojson"

# 단계를 나누는 경계값(%) — 전국 시군구를 다섯 덩어리로 나눈 값입니다.
BREAKS = [19, 23, 28, 38]

# 낮은 쪽은 옅게, 높은 쪽은 진하게 (5단계 색)
COLORS = ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000"]

# 범례에 보일 글자: '19% 미만', '19% 이상 ~ 23% 미만', ..., '38% 이상'
LABELS = (
    [f"{BREAKS[0]}% 미만"]
    + [f"{BREAKS[i]}% 이상 ~ {BREAKS[i + 1]}% 미만" for i in range(len(BREAKS) - 1)]
    + [f"{BREAKS[-1]}% 이상"]
)

# '계_65세', '계_100세 이상' 같은 열 이름에서 나이 숫자를 뽑는 규칙
AGE_COLUMN = re.compile(r"^계_(\d+)세")

ONE_DAY = 60 * 60 * 24  # 캐시 유지 시간(초)


# ─────────────────────────────────────────────
# 2. 데이터 불러오기
# ─────────────────────────────────────────────
@st.cache_data(ttl=ONE_DAY, show_spinner=False)
def load_population():
    """인구 CSV를 읽어 '가장 최신 연도'의 시군구별 고령화율을 계산합니다."""
    resp = requests.get(POP_URL, timeout=120)
    resp.raise_for_status()
    raw = resp.content

    # gz(압축) 파일이면 압축을 풀면서 읽습니다.
    compression = "gzip" if raw[:2] == b"\x1f\x8b" else None

    # 필요한 열(연도, 코드, '계_'로 시작하는 나이별 열)만 읽어서 메모리를 아낍니다.
    def need_column(name):
        name = name.strip()
        return name in ("연도", "코드") or name.startswith("계_")

    df = pd.read_csv(
        io.BytesIO(raw),
        compression=compression,
        encoding="utf-8-sig",
        dtype={"코드": str},  # ★ 코드는 계산용 숫자가 아니라 이름표 → 글자로 읽기
        usecols=need_column,
    )
    df.columns = df.columns.str.strip()

    # 가장 최신 연도만 남기기
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce")
    latest_year = int(df["연도"].max())
    df = df[df["연도"] == latest_year].copy()

    # 나이별 열 찾기: {'계_0세': 0, ..., '계_100세 이상': 100}
    age_columns = {}
    for col in df.columns:
        m = AGE_COLUMN.match(col)
        if m:
            age_columns[col] = int(m.group(1))
    if not age_columns:
        raise ValueError("'계_○세' 형식의 나이별 열을 찾지 못했습니다.")

    # 숫자로 바꾸기 (혹시 1,234처럼 쉼표가 있어도 처리)
    for col in age_columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    all_cols = list(age_columns)
    old_cols = [col for col, age in age_columns.items() if age >= 65]  # 65세 이상 열
    df[all_cols] = df[all_cols].fillna(0)

    # 읍·면·동 단위 → 시군구 단위로 합치기 (코드 앞 5자리 = 시군구)
    df["시군구코드"] = df["코드"].astype(str).str.strip().str[:5]
    df["전체"] = df[all_cols].sum(axis=1)
    df["고령"] = df[old_cols].sum(axis=1)

    agg = df.groupby("시군구코드", as_index=False)[["전체", "고령"]].sum()
    agg = agg[agg["전체"] > 0].copy()
    agg["고령화율"] = agg["고령"] / agg["전체"] * 100
    return agg, latest_year


@st.cache_data(ttl=ONE_DAY, show_spinner=False)
def load_geojson():
    """시군구 경계(GeoJSON)를 읽어 옵니다. 지역 코드는 글자로 통일합니다."""
    resp = requests.get(GEO_URL, timeout=120)
    resp.raise_for_status()
    geo = json.loads(resp.content.decode("utf-8"))
    for feature in geo["features"]:
        feature["properties"]["코드"] = str(feature["properties"]["코드"]).strip()
    return geo


# ─────────────────────────────────────────────
# 3. 지도 그리기
# ─────────────────────────────────────────────
def build_map(df, geo):
    """5단계 구간마다 트레이스를 하나씩 만들어 색과 범례(글자)를 붙입니다."""
    fig = go.Figure()

    for label, color in zip(LABELS, COLORS):
        part = df[df["구간"] == label]
        if part.empty:
            continue

        # 이 구간에 해당하는 지역의 경계만 골라 담습니다. (전송량을 줄이기 위해)
        codes = set(part["코드"])
        sub_geo = {
            "type": "FeatureCollection",
            "features": [f for f in geo["features"] if f["properties"]["코드"] in codes],
        }

        fig.add_trace(
            go.Choropleth(
                geojson=sub_geo,
                locations=part["코드"],
                featureidkey="properties.코드",  # 지역을 '코드'로 연결
                z=np.ones(len(part)),
                colorscale=[[0, color], [1, color]],  # 한 트레이스 = 한 가지 색
                showscale=False,  # 이어지는 색 막대는 쓰지 않음
                name=label,
                showlegend=True,  # 범례에 구간 글자 표시
                marker_line_color="#888888",
                marker_line_width=0.4,
                customdata=part[["시군구", "시도", "고령화율"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "시도: %{customdata[1]}<br>"
                    "고령화율: %{customdata[2]:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    # 배경 지도 타일 없이 경계선만 보이게 (축·바다·육지 배경 모두 숨김)
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(title="65세 이상 인구 비율", itemsizing="constant"),
    )
    return fig


# ─────────────────────────────────────────────
# 4. 순위 표 만들기
# ─────────────────────────────────────────────
def make_rank_table(part):
    """순위 · 시도 · 시군구 · 고령화율 · 65세 이상 인구 표를 만듭니다."""
    table = pd.DataFrame(
        {
            "순위": range(1, len(part) + 1),
            "시도": part["시도"].to_numpy(),
            "시군구": part["시군구"].to_numpy(),
            "고령화율(%)": part["고령화율"].round(1).to_numpy(),
            "65세 이상(명)": [f"{int(n):,}" for n in part["고령"]],
        }
    )
    return table


def show_table(table):
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "고령화율(%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )


# ─────────────────────────────────────────────
# 5. 화면 구성
# ─────────────────────────────────────────────
st.set_page_config(page_title="전국 고령화 지도", page_icon="🗺️", layout="wide")

st.title("🗺️ 전국 고령화 지도")

try:
    with st.spinner("데이터를 불러오는 중입니다..."):
        pop, year = load_population()
        geo = load_geojson()
except (requests.RequestException, ValueError, KeyError) as err:
    st.error(f"데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.\n\n({err})")
    st.stop()

# 경계 데이터의 속성(코드·시군구·시도) 표 만들기
regions = pd.DataFrame([f["properties"] for f in geo["features"]])[["코드", "시군구", "시도"]]

# 지역 '코드'로 인구 데이터와 경계 데이터를 연결 (이름으로 연결하지 않음!)
data = regions.merge(pop, left_on="코드", right_on="시군구코드", how="inner")

# 고령화율을 5단계로 나누기 (19 이상이면 다음 구간: 예) 19.0% → '19% 이상 ~ 23% 미만')
data["구간"] = pd.cut(
    data["고령화율"], bins=[-np.inf, *BREAKS, np.inf], labels=LABELS, right=False
).astype(str)

st.caption(
    f"{year}년 기준 · 시군구별 65세 이상 인구 비율(%) · 지도에 표시된 시군구 {len(data)}곳"
)

# 코드가 서로 맞지 않는 지역이 있으면 알려 주기
no_pop = len(regions) - len(data)
no_geo = len(set(pop["시군구코드"]) - set(regions["코드"]))
if no_pop or no_geo:
    st.warning(
        f"코드가 맞지 않아 지도에 나타나지 않는 지역이 있습니다. "
        f"(인구 자료가 없는 경계 {no_pop}곳, 경계가 없는 인구 자료 {no_geo}곳)"
    )

# 지도
st.plotly_chart(build_map(data, geo))

# 지도 아래: 높은 곳 10 / 낮은 곳 10 을 표 두 개로 나란히
st.subheader("고령화율 순위")
left, right = st.columns(2)

with left:
    st.markdown("**🔺 고령화율이 높은 곳 10곳**")
    show_table(make_rank_table(data.sort_values("고령화율", ascending=False).head(10)))

with right:
    st.markdown("**🔻 고령화율이 낮은 곳 10곳**")
    show_table(make_rank_table(data.sort_values("고령화율", ascending=True).head(10)))

st.caption("출처: greatsong/modudata (인구: population_yearly.csv.gz, 경계: sigungu_kr.geojson)")
