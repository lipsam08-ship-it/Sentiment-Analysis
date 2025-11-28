import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Stakeholder Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-card {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .positive { color: #2ecc71; }
    .negative { color: #e74c3c; }
    .neutral { color: #f39c12; }
    .critical-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    def __init__(self):
        # Enhanced sentiment word database
        self.positive_words = {
            'excellent', 'outstanding', 'amazing', 'great', 'good', 'awesome', 'fantastic',
            'wonderful', 'perfect', 'brilliant', 'superb', 'exceptional', 'satisfied',
            'happy', 'pleased', 'delighted', 'impressed', 'recommend', 'love', 'like',
            'helpful', 'responsive', 'quick', 'fast', 'easy', 'smooth', 'reliable',
            'professional', 'friendly', 'supportive', 'valuable', 'effective', 'efficient',
            'outstanding', 'remarkable', 'terrific', 'fantastic', 'awesome', 'perfect'
        }
        
        self.negative_words = {
            'poor', 'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrating',
            'annoying', 'angry', 'upset', 'displeased', 'unsatisfied', 'slow', 'delayed',
            'difficult', 'complicated', 'confusing', 'broken', 'failed', 'error', 'issue',
            'problem', 'bug', 'crash', 'unreliable', 'unprofessional', 'rude', 'ignored',
            'expensive', 'overpriced', 'waste', 'useless', 'worthless', 'hate', 'dislike'
        }
        
        self.strong_positive = {'excellent', 'outstanding', 'amazing', 'perfect', 'brilliant', 'fantastic'}
        self.strong_negative = {'terrible', 'awful', 'horrible', 'useless', 'worthless', 'hate'}
    
    def analyze_text(self, text):
        """Advanced rule-based sentiment analysis"""
        if pd.isna(text) or text == '':
            return 0.0, 'neutral'
        
        text_lower = str(text).lower()
        
        # Check for negation patterns
        negations = {'not', 'no', 'never', 'none', 'nothing', 'without'}
        words = text_lower.split()
        
        positive_score = 0
        negative_score = 0
        negation_active = False
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^a-zA-Z]', '', word)
            
            if word_clean in negations:
                negation_active = True
                continue
                
            if word_clean in self.strong_positive:
                score = 2 if not negation_active else -2
                positive_score += score
            elif word_clean in self.positive_words:
                score = 1 if not negation_active else -1
                positive_score += score
            elif word_clean in self.strong_negative:
                score = -2 if not negation_active else 2
                negative_score += score
            elif word_clean in self.negative_words:
                score = -1 if not negation_active else 1
                negative_score += score
                
            # Reset negation after one word
            if negation_active and word_clean:
                negation_active = False
        
        # Calculate final sentiment
        total_impact = positive_score + abs(negative_score)
        if total_impact == 0:
            return 0.0, 'neutral'
        
        sentiment_score = (positive_score + negative_score) / 10.0  # Normalize to -1 to 1
        
        # Categorize sentiment
        if sentiment_score > 0.1:
            return min(sentiment_score, 1.0), 'positive'
        elif sentiment_score < -0.1:
            return max(sentiment_score, -1.0), 'negative'
        else:
            return sentiment_score, 'neutral'
    
    def analyze_dataframe(self, df, text_column='feedback_text'):
        """Analyze sentiment for entire dataframe"""
        results = df[text_column].apply(self.analyze_text)
        df['sentiment_score'] = [r[0] for r in results]
        df['sentiment_category'] = [r[1] for r in results]
        return df

class SentimentDashboard:
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        
    def load_sample_data(self):
        """Generate comprehensive sample data"""
        np.random.seed(42)
        
        stakeholders = ['Customers', 'Employees', 'Investors', 'Partners', 'Regulators']
        sources = ['Survey', 'Social Media', 'Support Tickets', 'Reviews', 'Interviews']
        
        data = []
        for i in range(200):
            stakeholder = np.random.choice(stakeholders)
            source = np.random.choice(sources)
            sentiment_score = np.random.normal(0.2, 0.5)
            
            if sentiment_score > 0.3:
                feedbacks = [
                    "Excellent service and very responsive support team!",
                    "Great product with outstanding features that work perfectly",
                    "Very satisfied with the quick resolution and professional help",
                    "Amazing customer experience overall, highly recommended",
                    "Outstanding performance and reliable service delivery",
                    "Fantastic support team that solved my issues quickly",
                    "Perfect solution for our business needs and requirements",
                    "Brilliant features and excellent customer service"
                ]
            elif sentiment_score < -0.3:
                feedbacks = [
                    "Poor customer service experience and slow response",
                    "Terrible issues with the latest update that broke everything",
                    "Very frustrating billing process with hidden charges",
                    "Disappointed with product reliability and frequent crashes",
                    "Awful support experience with unhelpful staff",
                    "Horrible user interface that is difficult to navigate",
                    "Useless features that don't work as advertised",
                    "Waste of money and time with this terrible service"
                ]
            else:
                feedbacks = [
                    "Average experience, could be better but works okay",
                    "The product meets basic requirements for our needs",
                    "Satisfactory performance overall with some limitations",
                    "Adequate solution for current business requirements",
                    "Standard features work as expected most of the time",
                    "Reasonable pricing for what you get from the service",
                    "Acceptable quality but needs improvement in some areas",
                    "Moderate experience with both good and bad aspects"
                ]
            
            data.append({
                'date': datetime.now() - timedelta(days=np.random.randint(1, 90)),
                'stakeholder_group': stakeholder,
                'source': source,
                'feedback_text': np.random.choice(feedbacks),
                'priority': np.random.choice(['High', 'Medium', 'Low']),
                'sentiment_score': max(-1, min(1, sentiment_score))
            })
        
        df = pd.DataFrame(data)
        return self.analyzer.analyze_dataframe(df)
    
    def calculate_metrics(self, df):
        """Calculate key sentiment metrics"""
        avg_sentiment = df['sentiment_score'].mean()
        positive_pct = len(df[df['sentiment_score'] > 0.1]) / len(df) * 100
        negative_pct = len(df[df['sentiment_score'] < -0.1]) / len(df) * 100
        neutral_pct = 100 - positive_pct - negative_pct
        
        return {
            'avg_sentiment': avg_sentiment,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'neutral_pct': neutral_pct,
            'total_feedback': len(df)
        }
    
    def create_sentiment_gauge(self, score):
        """Create sentiment gauge using native Streamlit"""
        st.write(f"**Overall Sentiment Score: {score:.2f}**")
        
        # Create a visual gauge using progress bar
        normalized_score = (score + 1) / 2  # Convert -1 to 1 range to 0 to 1
        
        if score > 0.3:
            color = "green"
        elif score < -0.3:
            color = "red"
        else:
            color = "orange"
            
        st.progress(normalized_score, text=f"Sentiment Level")
        
        # Interpretation
        if score > 0.3:
            st.success("✅ Positive sentiment - Strong stakeholder satisfaction")
        elif score > 0.1:
            st.info("ℹ️  Moderately positive - Good stakeholder relations")
        elif score > -0.1:
            st.warning("⚠️  Neutral - Room for improvement")
        elif score > -0.3:
            st.error("❌ Moderately negative - Attention required")
        else:
            st.error("🚨 Critical - Immediate action needed")
    
    def create_sentiment_distribution(self, metrics):
        """Create sentiment distribution chart"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Positive Feedback",
                f"{metrics['positive_pct']:.1f}%",
                delta=f"{metrics['positive_pct'] - 33:.1f}%"
            )
        
        with col2:
            st.metric(
                "Neutral Feedback", 
                f"{metrics['neutral_pct']:.1f}%",
                delta=f"{metrics['neutral_pct'] - 33:.1f}%"
            )
        
        with col3:
            st.metric(
                "Negative Feedback",
                f"{metrics['negative_pct']:.1f}%",
                delta=f"{metrics['negative_pct'] - 33:.1f}%",
                delta_color="inverse"
            )
        
        # Simple bar chart using st.bar_chart
        dist_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Percentage': [metrics['positive_pct'], metrics['neutral_pct'], metrics['negative_pct']]
        })
        
        st.bar_chart(dist_data.set_index('Sentiment'))
    
    def create_trend_chart(self, df):
        """Create sentiment trend over time"""
        df['date'] = pd.to_datetime(df['date'])
        daily_avg = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        
        st.write("**Sentiment Trend Over Time**")
        st.line_chart(daily_avg.set_index('date'))
    
    def create_stakeholder_analysis(self, df):
        """Create stakeholder group analysis"""
        group_sentiment = df.groupby('stakeholder_group')['sentiment_score'].agg(['mean', 'count']).reset_index()
        
        st.write("**Average Sentiment by Stakeholder Group**")
        
        for _, row in group_sentiment.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                score = row['mean']
                if score > 0.3:
                    color = "🟢"
                elif score > 0.1:
                    color = "🟡"
                elif score > -0.1:
                    color = "🟠"
                elif score > -0.3:
                    color = "🔴"
                else:
                    color = "💀"
                    
                st.write(f"{color} **{row['stakeholder_group']}**: {score:.2f}")
            
            with col2:
                st.write(f"({row['count']} feedback)")
    
    def create_top_words_chart(self, df, sentiment_type='positive', top_n=8):
        """Create top words analysis"""
        if sentiment_type == 'positive':
            text_data = ' '.join(df[df['sentiment_score'] > 0.1]['feedback_text'])
            title = "🔹 Top Positive Words"
        elif sentiment_type == 'negative':
            text_data = ' '.join(df[df['sentiment_score'] < -0.1]['feedback_text'])
            title = "🔸 Top Negative Words"
        else:
            text_data = ' '.join(df[(df['sentiment_score'] >= -0.1) & (df['sentiment_score'] <= 0.1)]['feedback_text'])
            title = "🔸 Top Neutral Words"
        
        if not text_data.strip():
            st.write(f"{title}")
            st.write("No data available")
            return
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_data.lower())
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'have', 'has', 'was', 
                     'were', 'are', 'is', 'you', 'your', 'our', 'they', 'their', 'been'}
        filtered_words = [word for word in words if word not in stop_words]
        
        word_freq = Counter(filtered_words).most_common(top_n)
        
        st.write(f"**{title}**")
        for word, freq in word_freq:
            st.write(f"• {word} ({freq})")

    def generate_action_plan(self, df):
        """Generate automated action plans based on sentiment analysis"""
        actions = []
        
        for group in df['stakeholder_group'].unique():
            group_data = df[df['stakeholder_group'] == group]
            avg_sentiment = group_data['sentiment_score'].mean()
            negative_count = len(group_data[group_data['sentiment_score'] < -0.3])
            total_feedback = len(group_data)
            
            if avg_sentiment < -0.2 or (negative_count / total_feedback > 0.3 and total_feedback > 5):
                priority = "HIGH" if avg_sentiment < -0.4 else "MEDIUM"
                
                # Get top negative themes
                negative_feedback = ' '.join(group_data[group_data['sentiment_score'] < -0.1]['feedback_text'])
                negative_words = re.findall(r'\b[a-zA-Z]{4,}\b', negative_feedback.lower())
                stop_words = {'this', 'that', 'with', 'have', 'been', 'they', 'what', 'your'}
                filtered_words = [word for word in negative_words if word not in stop_words]
                common_issues = Counter(filtered_words).most_common(3)
                
                issues = [f"{word} ({count})" for word, count in common_issues]
                
                actions.append({
                    'stakeholder_group': group,
                    'priority': priority,
                    'avg_sentiment': avg_sentiment,
                    'negative_feedback_count': negative_count,
                    'common_issues': ', '.join(issues) if issues else 'General dissatisfaction',
                    'recommended_actions': [
                        f"Schedule meeting with {group} representatives",
                        f"Conduct root cause analysis for identified issues",
                        f"Develop improvement action plan with timeline",
                        f"Assign dedicated relationship manager"
                    ],
                    'timeline': "1-2 weeks" if priority == "HIGH" else "2-4 weeks"
                })
        
        return actions

def show_introduction():
    """Show the introduction page"""
    st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
    st.markdown('<h1>🎯 Stakeholder Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h3>Transform Feedback into Actionable Insights</h3>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 📋 Welcome to Your Stakeholder Intelligence Hub!
    
    This dashboard helps you **monitor, analyze, and act** on stakeholder feedback across your organization.
    """)
    
    # Key Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="step-card">
        <h4>🔍 Smart Analysis</h4>
        <p>Automated sentiment analysis using advanced NLP techniques</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-card">
        <h4>📊 Visual Insights</h4>
        <p>Interactive charts and metrics for easy understanding</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-card">
        <h4>🎯 Action Plans</h4>
        <p>Automated recommendations based on sentiment patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
    ## 🚀 How It Works
    
    1. **Upload Data** - Provide your stakeholder feedback data
    2. **Automatic Analysis** - Our system analyzes sentiment and patterns
    3. **Get Insights** - View comprehensive dashboards and reports
    4. **Take Action** - Follow automated action plans
    """)
    
    # Start Analysis Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Start Sentiment Analysis", use_container_width=True, type="primary"):
            st.session_state.current_step = "upload_data"
            st.rerun()

def show_upload_section():
    """Show data upload section"""
    st.markdown('<h2>📁 Step 1: Upload Your Data</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-section">
    <h3>📊 Upload Stakeholder Feedback Data</h3>
    <p>Upload a CSV file containing your stakeholder feedback data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file with columns: date, stakeholder_group, feedback_text"
    )
    
    # CSV format requirements
    with st.expander("📋 Required CSV Format"):
        st.markdown("""
        Your CSV file should include these columns:
        
        - **date**: Feedback date (YYYY-MM-DD)
        - **stakeholder_group**: Customer, Employee, Investor, Partner, etc.
        - **feedback_text**: The actual feedback comments
        - **source** (optional): Survey, Support, Social Media, etc.
        - **priority** (optional): High, Medium, Low
        
        ### Example CSV:
        ```csv
        date,stakeholder_group,feedback_text,source,priority
        2024-01-15,Customer,"Great service and quick response",Survey,High
        2024-01-16,Employee,"Need better tools for remote work",Internal,Medium
        2024-01-17,Investor,"Satisfied with quarterly results",Meeting,High
        ```
        """)
    
    # Sample data option
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        use_sample = st.checkbox("Use sample data for demonstration", value=True)
    
    with col2:
        if st.button("📊 Use Sample Data", use_container_width=True):
            st.session_state.use_sample_data = True
            st.session_state.current_step = "analysis"
            st.rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully uploaded {len(df)} records!")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Validate required columns
            required_columns = ['date', 'stakeholder_group', 'feedback_text']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            else:
                st.session_state.uploaded_data = df
                if st.button("🔍 Analyze This Data", use_container_width=True, type="primary"):
                    st.session_state.current_step = "analysis"
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    # Back button
    if st.button("⬅️ Back to Introduction"):
        st.session_state.current_step = "introduction"
        st.rerun()

def show_analysis_dashboard():
    """Show the main analysis dashboard"""
    dashboard = SentimentDashboard()
    
    # Load data
    if st.session_state.get('use_sample_data'):
        df = dashboard.load_sample_data()
        st.info("📊 Using sample demonstration data")
    else:
        df = st.session_state.get('uploaded_data')
        if df is None:
            st.error("No data available. Please go back and upload data.")
            if st.button("⬅️ Back to Data Upload"):
                st.session_state.current_step = "upload_data"
                st.rerun()
            return
    
    # Analyze data if not already analyzed
    if 'sentiment_score' not in df.columns:
        df = dashboard.analyzer.analyze_dataframe(df)
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔧 Analysis Controls")
        
        if st.button("🔄 Start New Analysis"):
            st.session_state.current_step = "upload_data"
            st.rerun()
        
        st.markdown("---")
        st.header("Filters")
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            
            start_date = st.date_input("Start Date", min_date)
            end_date = st.date_input("End Date", max_date)
        
        stakeholder_groups = st.multiselect(
            "Stakeholder Groups",
            options=df['stakeholder_group'].unique(),
            default=df['stakeholder_group'].unique()
        )
    
    # Apply filters
    if 'date' in df.columns:
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    
    if stakeholder_groups:
        df = df[df['stakeholder_group'].isin(stakeholder_groups)]
    
    # Main Dashboard
    st.markdown('<h2>📊 Sentiment Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    # Calculate metrics
    metrics = dashboard.calculate_metrics(df)
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Sentiment", f"{metrics['avg_sentiment']:.2f}")
    
    with col2:
        st.metric("Positive", f"{metrics['positive_pct']:.1f}%")
    
    with col3:
        st.metric("Negative", f"{metrics['negative_pct']:.1f}%")
    
    with col4:
        st.metric("Total Feedback", metrics['total_feedback'])
    
    # Sentiment Analysis
    st.subheader("🎯 Sentiment Analysis")
    dashboard.create_sentiment_gauge(metrics['avg_sentiment'])
    
    # Distribution
    st.subheader("📊 Sentiment Distribution")
    dashboard.create_sentiment_distribution(metrics)
    
    # Trend Analysis
    st.subheader("📈 Trend Analysis")
    dashboard.create_trend_chart(df)
    
    # Stakeholder Analysis
    st.subheader("👥 Stakeholder Analysis")
    dashboard.create_stakeholder_analysis(df)
    
    # Top Words Analysis
    st.subheader("🔤 Top Words Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dashboard.create_top_words_chart(df, 'positive')
    
    with col2:
        dashboard.create_top_words_chart(df, 'neutral')
    
    with col3:
        dashboard.create_top_words_chart(df, 'negative')
    
    # Action Plan
    st.subheader("📋 Automated Action Plan")
    actions = dashboard.generate_action_plan(df)
    
    if actions:
        st.markdown('<div class="critical-alert">', unsafe_allow_html=True)
        st.warning(f"🚨 {len(actions)} critical issue(s) requiring attention")
        st.markdown('</div>', unsafe_allow_html=True)
        
        for action in actions:
            with st.expander(f"🔴 {action['stakeholder_group']} - {action['priority']} Priority"):
                st.write(f"**Sentiment Score:** {action['avg_sentiment']:.2f}")
                st.write(f"**Negative Feedback:** {action['negative_feedback_count']}")
                st.write(f"**Common Issues:** {action['common_issues']}")
                
                st.write("**Recommended Actions:**")
                for i, step in enumerate(action['recommended_actions'], 1):
                    st.write(f"{i}. {step}")
                
                st.write(f"**Timeline:** {action['timeline']}")
    else:
        st.success("✅ No critical issues detected!")
    
    # Data Explorer
    st.subheader("📊 Data Explorer")
    with st.expander("View Raw Data"):
        st.dataframe(df)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Analyzed Data",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )
    
    # New Analysis Button
    st.markdown("---")
    if st.button("🔄 Start New Analysis", use_container_width=True):
        st.session_state.current_step = "upload_data"
        st.rerun()

def main():
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "introduction"
    
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    
    if 'use_sample_data' not in st.session_state:
        st.session_state.use_sample_data = False
    
    # Show appropriate section based on current step
    if st.session_state.current_step == "introduction":
        show_introduction()
    elif st.session_state.current_step == "upload_data":
        show_upload_section()
    elif st.session_state.current_step == "analysis":
        show_analysis_dashboard()

if __name__ == "__main__":
    main()
