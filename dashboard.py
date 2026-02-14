import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import koreanize_matplotlib

# 페이지 설정
st.set_page_config(page_title="쇼핑 트렌드 & 블로그 인사이트", layout="wide")

# 스타일 설정
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_data():
    output_dir = 'output'
    files = glob.glob(os.path.join(output_dir, "*.csv"))
    
    trend_files = [f for f in files if "ShoppingTrend" in f]
    search_files = [f for f in files if "ShoppingSearch" in f]
    blog_files = [f for f in files if "BlogPost" in f]
    
    trend_df = pd.concat([pd.read_csv(f) for f in trend_files], ignore_index=True) if trend_files else pd.DataFrame()
    search_df = pd.concat([pd.read_csv(f) for f in search_files], ignore_index=True) if search_files else pd.DataFrame()
    blog_df = pd.concat([pd.read_csv(f) for f in blog_files], ignore_index=True) if blog_files else pd.DataFrame()
    
    # 전처리
    if not trend_df.empty:
        trend_df['Date'] = pd.to_datetime(trend_df['Date'])
    if not search_df.empty:
        search_df['lprice'] = pd.to_numeric(search_df['lprice'], errors='coerce')
    
    return trend_df, search_df, blog_df

# 메인 실행
def main():
    st.title("🛍️ 통합 쇼핑 트렌드 & 블로그 인사이트 대시보드")
    
    trend_df, search_df, blog_df = load_data()
    
    if trend_df.empty or search_df.empty or blog_df.empty:
        st.error("데이터 파일이 부족합니다. 'output/' 폴더를 확인해주세요.")
        return

    # 사이드바 구성
    with st.sidebar:
        st.header("🔍 분석 필터")
        all_keywords = sorted(trend_df['Title'].unique())
        selected_keywords = st.multiselect("분석 키워드 선택", options=all_keywords, default=all_keywords[:2])
        
        price_range = st.slider("가격 범위 선택", 
                                int(search_df['lprice'].min()), 
                                int(search_df['lprice'].max()), 
                                (int(search_df['lprice'].min()), int(search_df['lprice'].max())))
        
        selected_malls = st.multiselect("쇼핑몰 필터", options=sorted(search_df['mallName'].unique()), default=[])

    # 필터링 적용
    filtered_trend = trend_df[trend_df['Title'].isin(selected_keywords)]
    filtered_search = search_df[search_df['title'].str.contains('|'.join(selected_keywords), case=False, na=False)]
    filtered_search = filtered_search[(filtered_search['lprice'] >= price_range[0]) & (filtered_search['lprice'] <= price_range[1])]
    if selected_malls:
        filtered_search = filtered_search[filtered_search['mallName'].isin(selected_malls)]
    
    # 상단 지표 (Metric)
    cols = st.columns(len(selected_keywords) + 1)
    for i, kw in enumerate(selected_keywords):
        kw_trend = filtered_trend[filtered_trend['Title'] == kw]
        if not kw_trend.empty:
            avg_ratio = kw_trend['Ratio'].mean()
            max_ratio = kw_trend['Ratio'].max()
            cols[i].metric(label=f"{kw} 평균 비율", value=f"{avg_ratio:.2f}", delta=f"최대 {max_ratio:.1f}")

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📈 트렌드 분석", "💰 쇼핑/가격 분석", "📝 텍스트 인사이트", "👥 인구통계 분석"])

    # 탭 1: 트렌드 비교
    with tab1:
        st.subheader("키워드별 검색 추이 비교")
        fig1 = px.line(filtered_trend, x='Date', y='Ratio', color='Title', 
                      title="일별 클릭 상대비율 추이", markers=True)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("#### [분석 표] 키워드 요약 통계")
        summary_table = filtered_trend.groupby('Title')['Ratio'].agg(['mean', 'max', 'min', 'std']).reset_index()
        summary_table.columns = ['키워드', '평균 비율', '최대 비율', '최소 비율', '표준편차']
        st.table(summary_table)

    # 탭 2: 쇼핑 및 가격 분석
    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("주요 쇼핑몰 점유율")
            mall_counts = filtered_search['mallName'].value_counts().head(10)
            fig2 = px.pie(values=mall_counts.values, names=mall_counts.index, hole=.3)
            st.plotly_chart(fig2, use_container_width=True)
        with col_right:
            st.subheader("몰별 가격 분포")
            top_malls = filtered_search['mallName'].value_counts().head(5).index
            fig3 = px.box(filtered_search[filtered_search['mallName'].isin(top_malls)], 
                         x='mallName', y='lprice', color='mallName', points="all")
            st.plotly_chart(fig3, use_container_width=True)
        st.subheader("최저가 상품 리스트")
        st.dataframe(filtered_search[['title', 'mallName', 'lprice', 'brand']].sort_values('lprice').head(10), use_container_width=True)
        st.subheader("쇼핑몰별 입점 브랜드 현황")
        cross_tab = pd.crosstab(filtered_search['mallName'], filtered_search['brand'].fillna('미지정'))
        st.write(cross_tab.head(10))

    # 탭 3: 텍스트 인사이트 (블로그)
    with tab3:
        st.subheader("활발한 블로거 TOP 15")
        filtered_blog = blog_df[blog_df['title'].str.contains('|'.join(selected_keywords), case=False, na=False)]
        if not filtered_blog.empty:
            blogger_rank = filtered_blog['bloggername'].value_counts().head(15).reset_index()
            blogger_rank.columns = ['블로거명', '포스팅 수']
            fig4 = px.bar(blogger_rank, x='포스팅 수', y='블로거명', orientation='h', color='포스팅 수')
            st.plotly_chart(fig4, use_container_width=True)
            st.subheader("블로그 제목 키워드 가중치 분석 (TF-IDF)")
            clean_titles = filtered_blog['title'].str.replace('<b>', '').str.replace('</b>', '').str.replace('&quot;', '')
            vectorizer = TfidfVectorizer(max_features=20)
            tfidf_matrix = vectorizer.fit_transform(clean_titles)
            word_weights = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 
                                         'weight': tfidf_matrix.sum(axis=0).tolist()[0]})
            word_weights = word_weights.sort_values('weight', ascending=False)
            fig5 = px.bar(word_weights, x='weight', y='word', orientation='h', title="블로그 핵심 단어 가중치")
            st.plotly_chart(fig5, use_container_width=True)
            st.dataframe(word_weights, use_container_width=True)
            st.subheader("관련 최신 블로그 포스트")
            st.dataframe(filtered_blog[['postdate', 'title', 'bloggername', 'link']].sort_values('postdate', ascending=False).head(10))
        else:
            st.info("해당 키워드와 관련된 블로그 포스트 데이터가 없습니다.")

    # 탭 4: 인구통계 분석 (성별/연령)
    with tab4:
        st.subheader("성별 및 연령별 검색 분포")
        has_gender = '성별' in trend_df.columns or 'gender' in trend_df.columns
        has_age = '연령' in trend_df.columns or 'ages' in trend_df.columns
        if has_gender:
            gender_col = '성별' if '성별' in trend_df.columns else 'gender'
            st.markdown("#### [그래프] 성별 클릭 비중")
            gender_counts = filtered_trend.groupby(gender_col)['Ratio'].sum().reset_index()
            fig_gender = px.bar(gender_counts, x=gender_col, y='Ratio', color=gender_col, 
                               title="성별 누적 클릭 지수", color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_gender, use_container_width=True)
            st.markdown("#### [분석 표] 성별 평균 및 최대 클릭 지수")
            gender_summary = filtered_trend.groupby(gender_col)['Ratio'].agg(['mean', 'max', 'count']).reset_index()
            gender_summary.columns = ['성별', '평균 클릭', '최대 클릭', '데이터 수']
            st.dataframe(gender_summary, use_container_width=True)
        else:
            st.info("데이터에 '성별' 정보가 없습니다. API 수집 시 gender 파라미터를 사용해 주세요.")
        if has_age:
            age_col = '연령' if '연령' in trend_df.columns else 'ages'
            st.markdown("#### [그래프] 연령대별 검색 분포")
            age_data = filtered_trend.groupby(age_col)['Ratio'].mean().reset_index()
            fig_age = px.line(age_data, x=age_col, y='Ratio', title="연령대별 평균 클릭 지수 추이")
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.divider()
            st.info("데이터에 '연령' 정보가 없습니다. API 수집 시 ages 파라미터를 사용해 주세요.")

if __name__ == "__main__":
    main()
