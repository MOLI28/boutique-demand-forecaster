# -----------------------------
# 1. Imports & Setup
# -----------------------------
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import qrcode
from io import BytesIO
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Boutique AI | Smart Retail Analytics",
    layout="wide",
    page_icon="👗",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 2. Clean & Professional Styling
# -----------------------------
st.markdown("""
    <style>
    /* Clean, modern styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #F9FAFB;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
        box-shadow: 2px 0 8px rgba(0,0,0,0.02);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        transition: all 0.2s;
    }
    
    .metric-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.1);
    }
    
    .metric-label {
        color: #6B7280;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #111827;
        font-size: 1.875rem;
        font-weight: 600;
        line-height: 1.2;
    }
    
    .metric-change {
        color: #10B981;
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Data cards */
    .data-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    
    .data-card h3 {
        color: #111827;
        font-size: 1.125rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .section-header h2 {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Badges */
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
    
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
    }
    
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
    }
    
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    
    .badge-info {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    
    .badge-purple {
        background-color: #EDE9FE;
        color: #5B21B6;
    }
    
    /* Progress bar */
    .progress-container {
        background-color: #F3F4F6;
        border-radius: 9999px;
        height: 0.5rem;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.3s;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #FFFFFF;
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #6B7280;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #E5E7EB, transparent);
        margin: 2rem 0;
    }
    .metric-card {
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 2rem 0; text-align: center;">
        <h1 style="color: #111827; font-size: 1.5rem; font-weight: 700; margin: 0;">👗 BOUTIQUE AI</h1>
        <p style="color: #6B7280; font-size: 0.875rem; margin: 0.25rem 0 0 0;">Smart Retail Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload CSV Data",
        type=["csv"],
        help="Upload your boutique sales data"
    )
    
    if uploaded_file:
        st.markdown("""
        <div style="background: #F3F4F6; padding: 0.75rem; border-radius: 8px; margin-top: 1rem;">
            <p style="color: #059669; margin: 0; font-size: 0.875rem; text-align: center;">✓ Data loaded successfully</p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data
def process_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['purchase_date'], dayfirst=True)
    if 'cost_price' not in df.columns:
        df['cost_price'] = df['current_price'] * 0.60
    return df

def add_return_reasons(df):
    """Add a return_reason column if not present, using realistic guesses."""
    if 'return_reason' not in df.columns:
        # Define possible reasons with weights based on rating and size
        reasons = [
            "Wrong size", 
            "Damaged product", 
            "Changed mind", 
            "Quality issue", 
            "Late delivery", 
            "Better price elsewhere",
            "Not as described", 
            "Duplicate order"
        ]
        # For returned items, assign a reason
        mask = df['is_returned'] == 1
        df.loc[mask, 'return_reason'] = np.random.choice(reasons, size=mask.sum())
    return df

# -----------------------------
# 4. Main App
# -----------------------------
if uploaded_file is not None:
    df = process_data(uploaded_file)
    df = add_return_reasons(df)
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])

    # Category multi-select
    categories = st.sidebar.multiselect(
        "Select categories",
        df['category'].unique(),
        default=df['category'].unique()
    )

    # Apply filters
    filtered_df = df[
        (df['date'].dt.date >= date_range[0]) &
        (df['date'].dt.date <= date_range[1]) &
        (df['category'].isin(categories))
    ]

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "📏 Size Analysis",
        "📦 Inventory",
        "📈 Forecast",
        "🎁 Loyalty"
    ])
    
    # ---------- TAB 1: DASHBOARD ----------
    with tab1:
        st.markdown("""
        <div class="section-header">
            <h2>Business Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" title="Total number of transactions">Total Sales</div>
                <div class="metric-value">{len(filtered_df):,}</div>
                <div class="metric-change">↑ 12.3% vs last month</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            revenue = filtered_df['current_price'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" title="Total revenue from all sales (before returns)">Revenue</div>
                <div class="metric-value">${revenue:,.0f}</div>
                <div class="metric-change">↑ 8.1% vs last month</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_rating = filtered_df['customer_rating'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" title="Average customer rating out of 5 stars">Avg Rating</div>
                <div class="metric-value">{avg_rating:.1f} ★</div>
                <div class="metric-change" style="color: #F59E0B;">{avg_rating:.1f} / 5.0</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            return_rate = (filtered_df['is_returned'].sum()/len(filtered_df))*100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label" title="Percentage of items returned by customers">Return Rate</div>
                <div class="metric-value">{return_rate:.1f}%</div>
                <div class="metric-change" style="color: {'#10B981' if return_rate < 10 else '#EF4444'};">
                    {'✓ Healthy' if return_rate < 10 else '⚠ Needs attention'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### Revenue by Category")
            
            category_rev = filtered_df.groupby('category')['current_price'].sum().reset_index()
            fig = px.bar(
                category_rev,
                x='category',
                y='current_price',
                color='current_price',
                color_continuous_scale=['#3B82F6', '#10B981'],
                labels={'current_price': 'Revenue ($)', 'category': ''},
                text='current_price'
            )
            fig.update_traces(
                texttemplate='$%{text:,.0f}',
                textposition='outside'
            )
            fig.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_chart2:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### Daily Sales Trend")
            
            daily_sales = filtered_df.groupby('date')['current_price'].sum().reset_index()
            fig = px.line(
                daily_sales,
                x='date',
                y='current_price',
                labels={'date': '', 'current_price': 'Sales ($)'},
                text='current_price'
            )
            fig.update_traces(
                texttemplate='$%{text:,.0f}',
                textposition='top left',
                line_color='#3B82F6',
                line_width=2
            )
            fig.update_layout(
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------- TAB 2: SIZE ANALYSIS ----------
    with tab2:
        st.markdown("""
        <div class="section-header">
            <h2>Size Distribution Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        size_stats = filtered_df['size'].value_counts().reset_index()
        size_stats.columns = ['Size', 'Count']
        
        col_s1, col_s2 = st.columns([1, 1])
        
        with col_s1:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### Size Distribution")
            
            fig = px.pie(
                size_stats,
                values='Count',
                names='Size',
                color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'],
                hole=0.4
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Units: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_s2:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### Sales by Size")
            
            fig = px.bar(
                size_stats,
                x='Size',
                y='Count',
                color='Count',
                color_continuous_scale='Blues',
                text='Count'
            )
            fig.update_traces(
                texttemplate='%{text} units',
                textposition='outside'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Size Strategy
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("### 📏 Size Strategy Recommendations")
        
        top_size = size_stats.iloc[0]['Size']
        top_count = size_stats.iloc[0]['Count']
        
        col_str1, col_str2, col_str3 = st.columns(3)
        
        with col_str1:
            st.markdown(f"""
            <div style="background: #F3F4F6; padding: 1rem; border-radius: 8px;">
                <span class="badge badge-success">Best Seller</span>
                <h3 style="margin: 0.5rem 0; font-size: 1.5rem;">Size {top_size}</h3>
                <p style="color: #6B7280; margin: 0;">{top_count} units sold</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_str2:
            st.markdown(f"""
            <div style="background: #F3F4F6; padding: 1rem; border-radius: 8px;">
                <span class="badge badge-info">Recommendation</span>
                <h3 style="margin: 0.5rem 0; font-size: 1.5rem;">+30% Stock</h3>
                <p style="color: #6B7280; margin: 0;">Increase {top_size} inventory</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_str3:
            revenue_potential = top_count * filtered_df['current_price'].mean()
            st.markdown(f"""
            <div style="background: #F3F4F6; padding: 1rem; border-radius: 8px;">
                <span class="badge badge-purple">Revenue Impact</span>
                <h3 style="margin: 0.5rem 0; font-size: 1.5rem;">${revenue_potential:,.0f}</h3>
                <p style="color: #6B7280; margin: 0;">From size {top_size}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------- TAB 3: INVENTORY ----------
    with tab3:
        st.markdown("""
        <div class="section-header">
            <h2>Inventory Intelligence</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Discount Simulator
        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        st.markdown("### 💰 Discount Simulator")
        
        col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
        
        with col_d1:
            discount = st.slider("Discount %", 0, 50, 10)
        
        original_rev = filtered_df['current_price'].sum()
        discounted_rev = original_rev * (1 - discount/100)
        profit = discounted_rev - filtered_df['cost_price'].sum()
        original_profit = original_rev - filtered_df['cost_price'].sum()
        
        with col_d2:
            st.markdown(f"""
            <div style="background: #F3F4F6; padding: 1rem; border-radius: 8px;">
                <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">Original</p>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">${original_rev:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d3:
            st.markdown(f"""
            <div style="background: #F3F4F6; padding: 1rem; border-radius: 8px;">
                <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">After {discount}% off</p>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0; color: {'#10B981' if profit > original_profit else '#EF4444'};">
                    ${discounted_rev:,.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance Analysis
        df_perf = filtered_df.groupby(['brand', 'category']).agg({
            'current_price': 'sum',
            'customer_rating': 'mean',
            'stock_quantity': 'sum',
            'is_returned': 'mean'
        }).reset_index()

        # Calculate scores
        max_revenue = df_perf['current_price'].max()
        df_perf['score'] = (
            (df_perf['current_price'] / max_revenue * 60) +   # revenue contributes up to 60 points
            (df_perf['customer_rating'] * 8) -                # rating up to 40 points (5*8)
            (df_perf['is_returned'] * 20)                      # return penalty up to 20 points
        ).clip(0, 100)

        # Split into categories
        high_performers = df_perf[df_perf['score'] >= 70].sort_values('score', ascending=False).head(5)
        low_performers = df_perf[df_perf['score'] <= 40].sort_values('score', ascending=True).head(5)

        col_inv1, col_inv2 = st.columns(2)

        with col_inv1:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### ✅ High Performers")
            st.markdown("*Priority restock items*")
            
            for _, row in high_performers.iterrows():
                st.markdown(f"""
                <div style="background: #F9FAFB; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 600;">{row['brand']}</span>
                            <span style="color: #6B7280; margin-left: 0.5rem;">{row['category']}</span>
                        </div>
                        <span class="badge badge-success">{row['score']:.0f} pts</span>
                    </div>
                    <div style="display: flex; gap: 2rem; margin-top: 0.5rem;">
                        <div><span style="color: #6B7280;">Stock:</span> {row['stock_quantity']:.0f}</div>
                        <div><span style="color: #6B7280;">Rating:</span> {row['customer_rating']:.1f} ★</div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {row['score']}%; background: #10B981;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        with col_inv2:
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### ❌ Clearance Candidates")
            st.markdown("*Consider discounting these items*")
            
            for _, row in low_performers.iterrows():
                st.markdown(f"""
                <div style="background: #F9FAFB; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 600;">{row['brand']}</span>
                            <span style="color: #6B7280; margin-left: 0.5rem;">{row['category']}</span>
                        </div>
                        <span class="badge badge-danger">{row['score']:.0f} pts</span>
                    </div>
                    <div style="display: flex; gap: 2rem; margin-top: 0.5rem;">
                        <div><span style="color: #6B7280;">Stock:</span> {row['stock_quantity']:.0f}</div>
                        <div><span style="color: #6B7280;">Rating:</span> {row['customer_rating']:.1f} ★</div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {row['score']}%; background: #EF4444;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # ---------- Return Analysis ----------
        st.markdown("""
        <div class="section-header" style="margin-top: 2rem;">
            <h2>Return Analysis</h2>
        </div>
        """, unsafe_allow_html=True)

        returned_df = filtered_df[filtered_df['is_returned'] == 1].copy()

        if not returned_df.empty:
            # Summary metrics
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Returns</div>
                    <div class="metric-value">{len(returned_df)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_r2:
                top_reason = returned_df['return_reason'].value_counts().idxmax()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Most Common Reason</div>
                    <div class="metric-value" style="font-size:1.3rem;">{top_reason}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_r3:
                return_value_pct = (returned_df['current_price'].sum() / filtered_df['current_price'].sum() * 100)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Value of Returns</div>
                    <div class="metric-value">{return_value_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Pie chart of reasons
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### Return Reasons Breakdown", unsafe_allow_html=True)
            reason_counts = returned_df['return_reason'].value_counts().reset_index()
            reason_counts.columns = ['Reason', 'Count']
            fig = px.pie(reason_counts, values='Count', names='Reason', 
                         color_discrete_sequence=px.colors.sequential.Reds_r, hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Recent returns list
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### Recent Returns", unsafe_allow_html=True)
            recent = returned_df.sort_values('date', ascending=False).head(5)
            for _, row in recent.iterrows():
                reason_lower = row['return_reason'].lower()
                if 'size' in reason_lower:
                    icon = "📏"
                elif 'damage' in reason_lower or 'quality' in reason_lower:
                    icon = "⚠️"
                elif 'late' in reason_lower or 'delivery' in reason_lower:
                    icon = "🚚"
                else:
                    icon = "💭"
                st.markdown(f"""
                <div style="background:#F9FAFB; padding:0.8rem; border-radius:12px; margin-bottom:0.5rem; border-left:5px solid #EF4444;">
                    <div style="display:flex; align-items:center; gap:1rem;">
                        <span style="font-size:1.8rem;">{icon}</span>
                        <div>
                            <strong>{row['return_reason']}</strong><br>
                            <span style="color:#6B7280; font-size:0.9rem;">
                                {row['brand']} | {row['category']} | Size {row['size']} | ${row['current_price']:.0f} | {row['date'].strftime('%Y-%m-%d')}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#F3F4F6; padding:3rem; border-radius:16px; text-align:center;">
                <div style="font-size:4rem;">🎉</div>
                <h3>No Returns Recorded</h3>
                <p style="color:#6B7280;">All products are loved by customers!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ---------- TAB 4: FORECAST ----------
    with tab4:
        st.markdown("""
        <div class="section-header">
            <h2>Sales Forecast 2026</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Explanation for shopkeeper
        st.info("""
        **🔮 How to read this forecast:**  
        This AI model analyzes your past sales patterns to predict future revenue.  
        - The blue line shows predicted daily sales.  
        - The shaded area is the confidence interval (likely range).  
        - Use monthly and quarterly charts to plan inventory and marketing.  
        - Seasonal patterns show which months and days are strongest.
        """)
        
        # Check if we have enough data first
        if len(df) < 30:
            st.warning(f"""
            ⚠️ **Insufficient Data for Forecasting**
            
            You need at least 30 days of historical data to generate accurate forecasts.
            Current data points: {len(df)}/30
            """)
            
            # Show sample data format instead
            st.markdown("""
            <div style="background: #F3F4F6; padding: 2rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <h3 style="color: #111827;">Data Requirements</h3>
                <p style="color: #6B7280; margin-bottom: 1.5rem;">Your CSV should contain:</p>
                <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                    <div style="background: white; padding: 1rem; border-radius: 8px;">
                        <strong>purchase_date</strong><br>
                        <span style="color: #6B7280;">DD-MM-YYYY</span>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 8px;">
                        <strong>current_price</strong><br>
                        <span style="color: #6B7280;">Numeric values</span>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 8px;">
                        <strong>30+ days</strong><br>
                        <span style="color: #6B7280;">Historical data</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                # Prepare data
                forecast_df = df.groupby('date')['current_price'].sum().reset_index()
                forecast_df.columns = ['ds', 'y']
                
                # Check if we have enough unique dates
                if len(forecast_df) < 7:
                    st.warning("⚠️ Not enough historical data points for forecasting. Need at least 7 unique dates.")
                else:
                    with st.spinner("🔄 Generating forecast... Please wait..."):
                        # Train model
                        model = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            seasonality_mode='multiplicative'
                        )
                        model.fit(forecast_df)
                        
                        # Predict for 2026
                        future = model.make_future_dataframe(periods=365)
                        forecast = model.predict(future)
                        
                        # Filter for 2026
                        forecast_2026 = forecast[forecast['ds'] >= '2026-01-01']

                        if forecast_2026.empty:
                            st.warning("⚠️ No forecast data available for 2026. The date range might be incorrect.")
                        else:
                            # MAIN FORECAST CHART
                            st.markdown('<div class="data-card">', unsafe_allow_html=True)
                            st.markdown("### 📈 2026 Revenue Forecast")
                            st.caption("Predicts daily sales with uncertainty range.")
                            
                            col_chart1, col_chart2 = st.columns([3, 1])
                            
                            with col_chart1:
                                fig = go.Figure()
                                
                                # Add confidence interval
                                fig.add_trace(go.Scatter(
                                    x=forecast_2026['ds'],
                                    y=forecast_2026['yhat_upper'],
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False,
                                    name='Upper Bound',
                                    hoverinfo='skip'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=forecast_2026['ds'],
                                    y=forecast_2026['yhat_lower'],
                                    mode='lines',
                                    line=dict(width=0),
                                    fill='tonexty',
                                    fillcolor='rgba(59, 130, 246, 0.1)',
                                    showlegend=False,
                                    name='Lower Bound',
                                    hoverinfo='skip'
                                ))
                                
                                # Add forecast line
                                fig.add_trace(go.Scatter(
                                    x=forecast_2026['ds'],
                                    y=forecast_2026['yhat'],
                                    mode='lines',
                                    line=dict(color='#3B82F6', width=3),
                                    name='Predicted Sales',
                                    hovertemplate='<b>%{x|%b %d, %Y}</b><br>Predicted: $%{y:,.0f}<br>Upper: $%{customdata[0]:,.0f}<br>Lower: $%{customdata[1]:,.0f}<extra></extra>',
                                    customdata=np.stack((forecast_2026['yhat_upper'], forecast_2026['yhat_lower']), axis=-1)
                                ))
                                
                                fig.update_layout(
                                    height=450,
                                    xaxis_title="",
                                    yaxis_title="Daily Sales ($)",
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    hovermode='x unified',
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                # Add range slider
                                fig.update_xaxes(rangeslider_visible=True)
                                
                                st.plotly_chart(fig, use_container_width=True)
                                st.success("✅ Forecast generated successfully!")
                            
                            with col_chart2:
                                # Key metrics
                                peak = forecast_2026['yhat'].max()
                                peak_date = forecast_2026.loc[forecast_2026['yhat'].idxmax(), 'ds'].strftime('%b %d, %Y')
                                avg = forecast_2026['yhat'].mean()
                                total = forecast_2026['yhat'].sum()
                                
                                # Calculate growth
                                historical_total = forecast_df['y'].sum()
                                growth = ((total / historical_total) - 1) * 100 if historical_total > 0 else 0
                                
                                st.markdown(f"""
                                <div style="background: #F9FAFB; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                                    <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">Peak Day</p>
                                    <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">${peak:,.0f}</p>
                                    <p style="color: #6B7280; margin: 0; font-size: 0.75rem;">{peak_date}</p>
                                </div>
                                
                                <div style="background: #F9FAFB; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                                    <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">Daily Average</p>
                                    <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">${avg:,.0f}</p>
                                </div>
                                
                                <div style="background: #F9FAFB; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                                    <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">Total 2026</p>
                                    <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">${total:,.0f}</p>
                                </div>
                                
                                <div style="background: #F9FAFB; padding: 1rem; border-radius: 8px;">
                                    <p style="color: #6B7280; margin: 0; font-size: 0.875rem;">Growth vs 2025</p>
                                    <p style="font-size: 1.5rem; font-weight: 600; margin: 0; color: {'#10B981' if growth > 0 else '#EF4444'};">
                                        {growth:+.1f}%
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)

                            # --- Forecast Accuracy ---
                            st.markdown('<div class="data-card">', unsafe_allow_html=True)
                            st.markdown("### 🎯 Forecast Accuracy (Historical Validation)")

                            if len(forecast_df) >= 60:
                                # Split data: last 30 days for testing
                                train_size = len(forecast_df) - 30
                                train = forecast_df.iloc[:train_size]
                                test = forecast_df.iloc[train_size:]

                                # Train model on training data only
                                model_acc = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                                model_acc.fit(train)

                                # Predict for test period
                                future_test = model_acc.make_future_dataframe(periods=len(test), include_history=False)
                                forecast_test = model_acc.predict(future_test)

                                # Merge predictions with actuals
                                comparison = test.merge(forecast_test[['ds', 'yhat']], on='ds', how='left')
                                comparison = comparison.dropna()

                                if len(comparison) > 0:
                                    # Calculate metrics
                                    actuals = comparison['y'].values
                                    preds = comparison['yhat'].values

                                    mae = np.mean(np.abs(actuals - preds))
                                    rmse = np.sqrt(np.mean((actuals - preds)**2))
                                    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
                                    # R²
                                    ss_res = np.sum((actuals - preds)**2)
                                    ss_tot = np.sum((actuals - np.mean(actuals))**2)
                                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                    col_m1.metric("MAE", f"${mae:,.0f}")
                                    col_m2.metric("RMSE", f"${rmse:,.0f}")
                                    col_m3.metric("MAPE", f"{mape:.1f}%")
                                    col_m4.metric("R²", f"{r2:.2f}")

                                    # Optional: show actual vs predicted plot
                                    fig_acc = px.line(comparison, x='ds', y=['y', 'yhat'], 
                                                    labels={'value': 'Sales', 'variable': 'Legend'},
                                                    title="Actual vs Predicted (Last 30 Days)")
                                    st.plotly_chart(fig_acc, use_container_width=True)
                                else:
                                    st.warning("Not enough overlapping data for accuracy calculation.")
                            else:
                                st.info("Need at least 60 days of data for a reliable train/test split (30 train, 30 test).")
                            st.markdown('</div>', unsafe_allow_html=True)

                            # DETAILED FORECAST ANALYSIS
                            st.markdown("""
                            <div class="section-header">
                                <h2>Detailed Forecast Analysis</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Monthly Revenue Projection
                            st.markdown('<div class="data-card">', unsafe_allow_html=True)
                            st.markdown("### 📊 Monthly Revenue Projection")
                            st.caption("Shows total revenue expected each month. Use this to plan seasonal stock.")
                            
                            # Create monthly forecast data
                            forecast_2026['month'] = forecast_2026['ds'].dt.strftime('%B')
                            monthly_forecast = forecast_2026.groupby('month')['yhat'].sum().reset_index()
                            
                            # Sort months correctly
                            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                                          'July', 'August', 'September', 'October', 'November', 'December']
                            monthly_forecast['month'] = pd.Categorical(monthly_forecast['month'], 
                                                                      categories=month_order, ordered=True)
                            monthly_forecast = monthly_forecast.sort_values('month')
                            
                            # Create bar chart with values
                            fig_monthly = px.bar(
                                monthly_forecast,
                                x='month',
                                y='yhat',
                                color='yhat',
                                color_continuous_scale='Blues',
                                labels={'yhat': 'Revenue ($)', 'month': ''},
                                text_auto='.0f'
                            )
                            fig_monthly.update_traces(
                                texttemplate='$%{text:,.0f}',
                                textposition='outside',
                                marker_line_color='#1E3A8A',
                                marker_line_width=1.5,
                                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
                            )
                            fig_monthly.update_layout(
                                height=400,
                                showlegend=False,
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig_monthly, use_container_width=True)
                            
                            # Best month insight
                            best_month_idx = monthly_forecast['yhat'].idxmax()
                            best_month = monthly_forecast.loc[best_month_idx, 'month']
                            best_month_value = monthly_forecast.loc[best_month_idx, 'yhat']
                            
                            st.markdown(f"""
                            <div style="background: #EBF5FF; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                                <span class="badge" style="background: #3B82F6; color: white;">📈 Peak Month</span>
                                <span style="margin-left: 1rem; color: #1E3A8A;"><strong>{best_month}</strong> - ${best_month_value:,.0f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Two column layout for additional charts
                            col_g1, col_g2 = st.columns(2)
                            
                            with col_g1:
                                # Weekly Pattern
                                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                                st.markdown("### 📅 Weekly Sales Pattern")
                                st.caption("Shows which days of the week typically have higher sales. Plan staffing and promotions accordingly.")
                                
                                # Get weekly seasonality
                                weekly = forecast[['ds', 'weekly']].copy()
                                weekly['day'] = weekly['ds'].dt.day_name()
                                weekly_pattern = weekly.groupby('day')['weekly'].mean().reset_index()
                                
                                # Sort days
                                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                                weekly_pattern['day'] = pd.Categorical(weekly_pattern['day'], categories=day_order, ordered=True)
                                weekly_pattern = weekly_pattern.sort_values('day')
                                
                                # Create line chart with values
                                fig_weekly = px.line(
                                    weekly_pattern,
                                    x='day',
                                    y='weekly',
                                    markers=True,
                                    labels={'weekly': 'Sales Multiplier', 'day': ''},
                                    text=weekly_pattern['weekly'].round(2)
                                )
                                fig_weekly.update_traces(
                                    texttemplate='%{text}x',
                                    textposition='top center',
                                    line_color='#8B5CF6',
                                    line_width=3,
                                    marker=dict(size=8, color='#8B5CF6'),
                                    hovertemplate='<b>%{x}</b><br>Multiplier: %{y:.2f}x<br>%{y:.1%} above baseline<extra></extra>'
                                )
                                fig_weekly.update_layout(
                                    height=350,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    yaxis_tickformat='.0%'
                                )
                                fig_weekly.add_hline(y=1, line_dash="dash", line_color="#9CA3AF", 
                                                    annotation_text="Average", annotation_position="bottom right")
                                st.plotly_chart(fig_weekly, use_container_width=True)
                                
                                # Best day insight
                                best_day_idx = weekly_pattern['weekly'].idxmax()
                                best_day = weekly_pattern.loc[best_day_idx, 'day']
                                best_day_mult = weekly_pattern.loc[best_day_idx, 'weekly']
                                
                                st.markdown(f"""
                                <div style="background: #F3E8FF; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                                    <span class="badge" style="background: #8B5CF6; color: white;">⭐ Best Day</span>
                                    <span style="margin-left: 1rem; color: #5B21B6;"><strong>{best_day}</strong> - {(best_day_mult-1)*100:.0f}% above average</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col_g2:
                                # Quarterly Performance
                                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                                st.markdown("### 📊 Quarterly Performance")
                                st.caption("Total revenue per quarter. Helps with broader business planning.")
                                
                                # Create quarterly data
                                forecast_2026['quarter'] = 'Q' + forecast_2026['ds'].dt.quarter.astype(str)
                                quarterly = forecast_2026.groupby('quarter')['yhat'].sum().reset_index()
                                
                                # Sort quarters
                                quarter_order = ['Q1', 'Q2', 'Q3', 'Q4']
                                quarterly['quarter'] = pd.Categorical(quarterly['quarter'], categories=quarter_order, ordered=True)
                                quarterly = quarterly.sort_values('quarter')
                                
                                # Create bar chart with values
                                colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
                                fig_quarter = go.Figure()
                                
                                for i, row in quarterly.iterrows():
                                    fig_quarter.add_trace(go.Bar(
                                        x=[row['quarter']],
                                        y=[row['yhat']],
                                        name=row['quarter'],
                                        marker_color=colors[i],
                                        text=f"${row['yhat']:,.0f}",
                                        textposition='outside',
                                        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
                                    ))
                                
                                fig_quarter.update_layout(
                                    height=350,
                                    showlegend=False,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    barmode='group'
                                )
                                st.plotly_chart(fig_quarter, use_container_width=True)
                                
                                # Best quarter insight
                                best_q_idx = quarterly['yhat'].idxmax()
                                best_q = quarterly.loc[best_q_idx, 'quarter']
                                best_q_value = quarterly.loc[best_q_idx, 'yhat']
                                
                                st.markdown(f"""
                                <div style="background: #FEF3C7; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                                    <span class="badge" style="background: #F59E0B; color: white;">🏆 Top Quarter</span>
                                    <span style="margin-left: 1rem; color: #92400E;"><strong>{best_q}</strong> - ${best_q_value:,.0f}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Cumulative Revenue Chart
                            st.markdown('<div class="data-card">', unsafe_allow_html=True)
                            st.markdown("### 📈 Cumulative Revenue Tracker")
                            st.caption("Running total of revenue throughout the year. See when you hit milestones.")
                            
                            forecast_2026['cumulative'] = forecast_2026['yhat'].cumsum()
                            
                            fig_cumulative = go.Figure()
                            
                            fig_cumulative.add_trace(go.Scatter(
                                x=forecast_2026['ds'],
                                y=forecast_2026['cumulative'],
                                mode='lines',
                                line=dict(color='#3B82F6', width=3),
                                fill='tozeroy',
                                fillcolor='rgba(59, 130, 246, 0.1)',
                                name='Cumulative Revenue',
                                hovertemplate='<b>%{x|%b %d}</b><br>Cumulative: $%{y:,.0f}<extra></extra>'
                            ))
                            
                            # Add milestone markers with values
                            total_2026 = forecast_2026['cumulative'].iloc[-1]
                            milestones = [0.25, 0.5, 0.75, 1.0]
                            milestone_names = ['25%', '50%', '75%', '100%']
                            milestone_colors = ['#F59E0B', '#10B981', '#8B5CF6', '#EF4444']
                            
                            for pct, name, color in zip(milestones, milestone_names, milestone_colors):
                                target = total_2026 * pct
                                milestone_data = forecast_2026[forecast_2026['cumulative'] >= target]
                                if not milestone_data.empty:
                                    milestone_date = milestone_data.iloc[0]['ds']
                                    fig_cumulative.add_trace(go.Scatter(
                                        x=[milestone_date],
                                        y=[target],
                                        mode='markers+text',
                                        marker=dict(size=12, color=color, symbol='diamond'),
                                        text=[f"{name}<br>${target:,.0f}"],
                                        textposition="top center",
                                        textfont=dict(size=9, color=color),
                                        showlegend=False,
                                        hovertemplate=f'<b>{name} Milestone</b><br>Date: %{{x|%b %d}}<br>Amount: $%{{y:,.0f}}<extra></extra>'
                                    ))
                            
                            fig_cumulative.update_layout(
                                height=400,
                                xaxis_title="",
                                yaxis_title="Cumulative Revenue ($)",
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                hovermode='x',
                                showlegend=False
                            )
                            
                            # Add range slider
                            fig_cumulative.update_xaxes(rangeslider_visible=True)
                            
                            st.plotly_chart(fig_cumulative, use_container_width=True)
                            
                            # Year-end summary
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); 
                                        padding: 1.5rem; border-radius: 12px; color: white; margin-top: 1rem;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <p style="margin: 0; opacity: 0.9;">2026 Year-End Projection</p>
                                        <p style="font-size: 2rem; font-weight: 700; margin: 0;">${total_2026:,.0f}</p>
                                    </div>
                                    <div style="text-align: right;">
                                        <p style="margin: 0; opacity: 0.9;">Growth Rate</p>
                                        <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">{growth:+.1f}%</p>
                                    </div>
                                </div>
                                <div class="progress-container" style="background: rgba(255,255,255,0.2); margin-top: 1rem;">
                                    <div class="progress-bar" style="width: 100%; background: white;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Seasonal Decomposition
                            with st.expander("🔍 View Seasonal Pattern Analysis (Advanced)"):
                                st.markdown("""
                                <div class="data-card">
                                    <h4>📊 Yearly & Weekly Seasonality</h4>
                                    <p>These charts separate the forecast into components: yearly pattern (which months are strong) and weekly pattern (which days are strong). The trend shows the overall growth direction.</p>
                                """, unsafe_allow_html=True)
                                
                                col_s1, col_s2 = st.columns(2)
                                
                                with col_s1:
                                    # Yearly seasonality with values
                                    forecast['month_num'] = forecast['ds'].dt.month
                                    monthly_pattern = forecast.groupby('month_num')['yearly'].mean().reset_index()
                                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                    monthly_pattern['month'] = monthly_pattern['month_num'].map(lambda x: month_names[x-1])
                                    
                                    fig_yearly = px.line(
                                        monthly_pattern,
                                        x='month',
                                        y='yearly',
                                        markers=True,
                                        title="Yearly Seasonality (Multiplier)",
                                        text=monthly_pattern['yearly'].round(2)
                                    )
                                    fig_yearly.update_traces(
                                        texttemplate='%{text}x',
                                        textposition='top center',
                                        line_color='#8B5CF6',
                                        line_width=3
                                    )
                                    fig_yearly.add_hline(y=1, line_dash="dash", line_color="gray")
                                    fig_yearly.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white')
                                    st.plotly_chart(fig_yearly, use_container_width=True)
                                    st.caption("Values above 1 mean higher sales than average.")
                                
                                with col_s2:
                                    # Weekly pattern (detailed bar)
                                    weekly_detail = forecast[['ds','weekly']].copy()
                                    weekly_detail['day_num'] = weekly_detail['ds'].dt.dayofweek
                                    weekly_detail = weekly_detail.groupby('day_num')['weekly'].mean().reset_index()
                                    weekly_detail['day'] = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                                    
                                    fig_weekly_detail = px.bar(
                                        weekly_detail, x='day', y='weekly',
                                        text=weekly_detail['weekly'].apply(lambda x: f'{x:.2f}x'),
                                        color='weekly', color_continuous_scale='Purples'
                                    )
                                    fig_weekly_detail.add_hline(y=1, line_dash="dash", line_color="gray")
                                    fig_weekly_detail.update_layout(title="Weekly Multiplier", height=300)
                                    st.plotly_chart(fig_weekly_detail, use_container_width=True)
                                
                                # Trend
                                st.markdown("#### 📈 Underlying Growth Trend")
                                trend_data = forecast[forecast['ds'] >= '2026-01-01']
                                fig_trend = px.line(
                                    trend_data,
                                    x='ds',
                                    y='trend',
                                    title="Underlying Trend ($)",
                                    text=trend_data['trend'].round(0).astype(int)
                                )
                                fig_trend.update_traces(
                                    texttemplate='$%{text}',
                                    textposition='top center',
                                    line_color='#10B981',
                                    line_width=3
                                )
                                fig_trend.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white')
                                st.plotly_chart(fig_trend, use_container_width=True)
                                st.caption("Shows the long-term direction of sales (inflation/growth).")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Forecast generation failed: {str(e)}")
                st.info("""
                **Troubleshooting tips:**
                1. Check that dates are in DD-MM-YYYY format
                2. Ensure 'current_price' column contains numeric values
                3. Verify you have at least 30 days of historical data
                4. Check for any missing or null values in your data
                """)
    
    # ---------- TAB 5: LOYALTY ----------
    with tab5:
        st.markdown("""
        <div class="section-header">
            <h2>Customer Loyalty Program</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if 'customer_id' in df.columns:
            # Calculate customer metrics, optionally including name
            if 'customer_name' in df.columns:
                customer_data = df.groupby('customer_id').agg({
                    'current_price': ['sum', 'count'],
                    'customer_rating': 'mean',
                    'customer_name': 'first'
                }).round(2)
                customer_data.columns = ['total_spent', 'purchases', 'avg_rating', 'customer_name']
            else:
                customer_data = df.groupby('customer_id').agg({
                    'current_price': ['sum', 'count'],
                    'customer_rating': 'mean'
                }).round(2)
                customer_data.columns = ['total_spent', 'purchases', 'avg_rating']
            customer_data = customer_data.reset_index()
            
            # Define tiers
            def get_tier(spent):
                if spent >= 1000:
                    return "Platinum", "#6B7280"
                elif spent >= 500:
                    return "Gold", "#F59E0B"
                elif spent >= 200:
                    return "Silver", "#94A3B8"
                else:
                    return "Bronze", "#B45309"
            
            customer_data['tier'], customer_data['tier_color'] = zip(*customer_data['total_spent'].map(get_tier))
            
            # Tier distribution
            tier_counts = customer_data['tier'].value_counts()
            
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            with col_t1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Members</div>
                    <div class="metric-value">{len(customer_data)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_t2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Platinum</div>
                    <div class="metric-value">{tier_counts.get('Platinum', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_t3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Gold</div>
                    <div class="metric-value">{tier_counts.get('Gold', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_t4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Silver/Bronze</div>
                    <div class="metric-value">{tier_counts.get('Silver', 0) + tier_counts.get('Bronze', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Tier benefits
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### 🎁 Membership Benefits")
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.markdown("""
                <div style="background: #F9FAFB; padding: 1.5rem; border-radius: 12px; text-align: center; border-top: 4px solid #6B7280;">
                    <span class="badge badge-info">👑 Platinum</span>
                    <h3 style="margin: 1rem 0;">$1,000+</h3>
                    <ul style="text-align: left; color: #6B7280; list-style: none; padding: 0;">
                        <li>✓ 20% off everything</li>
                        <li>✓ Free express shipping</li>
                        <li>✓ Priority customer service</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col_b2:
                st.markdown("""
                <div style="background: #F9FAFB; padding: 1.5rem; border-radius: 12px; text-align: center; border-top: 4px solid #F59E0B;">
                    <span class="badge badge-warning">⭐ Gold</span>
                    <h3 style="margin: 1rem 0;">$500 - $999</h3>
                    <ul style="text-align: left; color: #6B7280; list-style: none; padding: 0;">
                        <li>✓ 15% off everything</li>
                        <li>✓ Free standard shipping</li>
                        <li>✓ Birthday gift</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            with col_b3:
                st.markdown("""
                <div style="background: #F9FAFB; padding: 1.5rem; border-radius: 12px; text-align: center; border-top: 4px solid #94A3B8;">
                    <span class="badge badge-purple">🥈 Silver</span>
                    <h3 style="margin: 1rem 0;">$200 - $499</h3>
                    <ul style="text-align: left; color: #6B7280; list-style: none; padding: 0;">
                        <li>✓ 10% off purchases</li>
                        <li>✓ Birthday discount</li>
                        <li>✓ Points multiplier</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Member lookup
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Member Lookup")
            
            customer_data_sorted = customer_data.sort_values('total_spent', ascending=False)
            
            # Dynamic formatting for dropdown
            if 'customer_name' in customer_data.columns:
                format_func = lambda x: f"{customer_data[customer_data['customer_id']==x]['customer_name'].iloc[0]} (${customer_data[customer_data['customer_id']==x]['total_spent'].iloc[0]:,.0f})"
            else:
                format_func = lambda x: f"Member #{x} - ${customer_data[customer_data['customer_id']==x]['total_spent'].iloc[0]:,.0f}"
            
            selected_customer = st.selectbox(
                "Select a member",
                customer_data_sorted['customer_id'].tolist(),
                format_func=format_func
            )
            
            if selected_customer:
                member = customer_data[customer_data['customer_id'] == selected_customer].iloc[0]
                points = int(member['total_spent'] / 10)
                
                col_m1, col_m2 = st.columns([1, 1])
                with col_m1:
                    # Member card with name if available
                    if 'customer_name' in member:
                        name_display = member['customer_name']
                    else:
                        name_display = f"Member #{selected_customer}"
                    
                    st.markdown(f"""
                    <div style="background: #F9FAFB; padding: 1.5rem; border-radius: 12px; text-align: center;">
                        <div style="font-size: 4rem; margin-bottom: 1rem;">👤</div>
                        <span class="badge" style="background: {member['tier_color']}20; color: {member['tier_color']};">
                            {member['tier']} Member
                        </span>
                        <h3 style="margin: 1rem 0 0.5rem 0;">{name_display}</h3>
                        <p style="color: #6B7280; margin: 0;">{member['purchases']} purchases</p>
                        
                        <div style="margin-top: 1.5rem; text-align: left;">
                            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #E5E7EB;">
                                <span style="color: #6B7280;">Total Spent</span>
                                <span style="font-weight: 600;">${member['total_spent']:,.2f}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #E5E7EB;">
                                <span style="color: #6B7280;">Points Balance</span>
                                <span style="font-weight: 600;">{points} pts</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                                <span style="color: #6B7280;">Avg Rating Given</span>
                                <span style="font-weight: 600;">{member['avg_rating']:.1f} ★</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    st.markdown("""
                    <div style="background: #F9FAFB; padding: 1.5rem; border-radius: 12px; text-align: center;">
                        <h4 style="margin: 0 0 1rem 0;">Digital Member Card</h4>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Generate Digital Pass"):
                        if 'customer_name' in member:
                            qr_data = f"MEMBER:{selected_customer}|NAME:{member['customer_name']}|TIER:{member['tier']}|POINTS:{points}|SPENT:${member['total_spent']}"
                        else:
                            qr_data = f"MEMBER:{selected_customer}|TIER:{member['tier']}|POINTS:{points}|SPENT:${member['total_spent']}"
                        qr = qrcode.make(qr_data)
                        buf = BytesIO()
                        qr.save(buf)
                        
                        st.image(buf.getvalue(), width=200)
                        
                        st.markdown(f"""
                        <div style="margin-top: 1rem; text-align: left;">
                            <p style="margin: 0.25rem 0;"><strong>Member ID:</strong> {selected_customer}</p>
                            <p style="margin: 0.25rem 0;"><strong>Tier:</strong> {member['tier']}</p>
                            <p style="margin: 0.25rem 0;"><strong>Points:</strong> {points}</p>
                            <p style="margin: 0.25rem 0;"><strong>Spent:</strong> ${member['total_spent']:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # ----- Give Discount to Member -----
                st.markdown("---")
                st.markdown("### 🎁 Give Discount")

                col_disc1, col_disc2 = st.columns([1, 1])
                with col_disc1:
                    discount_percent = st.number_input(
                        "Discount %",
                        min_value=0,
                        max_value=100,
                        value=10,
                        step=5,
                        key=f"discount_{selected_customer}"
                    )
                with col_disc2:
                    expiry_days = st.number_input(
                        "Valid for (days)",
                        min_value=1,
                        max_value=365,
                        value=30,
                        step=1
                    )

                if st.button("Generate Discount QR Code", key=f"qr_btn_{selected_customer}"):
                    # Prepare data
                    if 'customer_name' in member:
                        name = member['customer_name']
                    else:
                        name = f"Member #{selected_customer}"
                    
                    qr_data = (
                        f"MEMBER:{selected_customer}|"
                        f"NAME:{name}|"
                        f"TIER:{member['tier']}|"
                        f"DISCOUNT:{discount_percent}%|"
                        f"EXPIRY:{expiry_days} days"
                    )
                    
                    qr = qrcode.make(qr_data)
                    buf = BytesIO()
                    qr.save(buf)
                    
                    st.image(buf.getvalue(), width=200, caption=f"{discount_percent}% Off QR Code")
                    
                    st.success(f"QR code generated for {name} – {discount_percent}% discount valid for {expiry_days} days.")
                # ----- End of Give Discount -----
                
                # Member purchase history
                st.markdown("### 📋 Recent Purchases")
                member_purchases = df[df['customer_id'] == selected_customer].sort_values('date', ascending=False).head(5)
                
                if not member_purchases.empty:
                    purchase_display = member_purchases[['date', 'category', 'brand', 'current_price', 'size']].copy()
                    purchase_display['date'] = purchase_display['date'].dt.strftime('%Y-%m-%d')
                    purchase_display['current_price'] = purchase_display['current_price'].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(purchase_display, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div style="background: #F3F4F6; padding: 3rem; border-radius: 12px; text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎁</div>
                <h3 style="color: #111827; margin: 0 0 0.5rem 0;">Loyalty Program Ready</h3>
                <p style="color: #6B7280; max-width: 400px; margin: 0 auto;">
                    Add a 'customer_id' column to your CSV to enable the loyalty program and track customer purchases.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer (outside all tabs)
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #6B7280; font-size: 0.875rem;">
        <p>© 2026 Boutique AI | Smart Retail Analytics</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">👗</div>
        <h1 style="color: #111827; font-size: 2.5rem; margin: 0 0 0.5rem 0;">Boutique AI</h1>
        <p style="color: #6B7280; font-size: 1.125rem; max-width: 500px; margin: 0 auto 2rem auto;">
            Upload your sales data to unlock AI-powered insights
        </p>
        <div style="display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap;">
            <div style="background: white; padding: 1.5rem; border-radius: 12px; width: 200px;">
                <div style="font-size: 2rem;">📊</div>
                <h3 style="margin: 0.5rem 0;">Analytics</h3>
                <p style="color: #6B7280;">Real-time metrics</p>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 12px; width: 200px;">
                <div style="font-size: 2rem;">📈</div>
                <h3 style="margin: 0.5rem 0;">Forecast</h3>
                <p style="color: #6B7280;">AI predictions</p>
            </div>
            <div style="background: white; padding: 1.5rem; border-radius: 12px; width: 200px;">
                <div style="font-size: 2rem;">🎁</div>
                <h3 style="margin: 0.5rem 0;">Loyalty</h3>
                <p style="color: #6B7280;">Reward customers</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)