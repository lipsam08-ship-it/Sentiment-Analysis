import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from collections import Counter

# TextBlob imports
from textblob import TextBlob

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
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
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_thresholds = {
            'positive': 0.1,
            'negative': -0.1
        }
    
    def analyze_text(self, text):
        """Analyze sentiment using TextBlob"""
        if pd.isna(text) or text == '':
            return 0.0, 'neutral'
        
        try:
            analysis = TextBlob(str(text))
            polarity = analysis.sentiment.polarity
            
            if polarity > self.sentiment_thresholds['positive']:
                sentiment = 'positive'
            elif polarity < self.sentiment_thresholds['negative']:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return polarity, sentiment
        except Exception as e:
            # Fallback in case TextBlob fails
            st.warning(f"TextBlob analysis failed: {e}")
            return 0.0, 'neutral'
    
    def analyze_dataframe(self, df, text_column='feedback_text'):
        """Analyze sentiment for entire dataframe"""
        st.info("🔍 Analyzing sentiment with TextBlob...")
        
        # Show progress
        progress_bar = st.progress(0)
        results = []
        
        for i, text in enumerate(df[text_column]):
            results.append(self.analyze_text(text))
            if i % 20 == 0:  # Update progress every 20 rows
                progress_bar.progress((i + 1) / len(df))
        
        progress_bar.empty()
        
        df['sentiment_score'] = [r[0] for r in results]
        df['sentiment_category'] = [r[1] for r in results]
        return df

class SentimentDashboard:
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        
    def load_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        stakeholders = ['Customers', 'Employees', 'Investors', 'Partners', 'Regulators']
        sources = ['Survey', 'Social Media', 'Support Tickets', 'Reviews', 'Interviews']
        
        data = []
        for i in range(200):
            stakeholder = np.random.choice(stakeholders)
            source = np.random.choice(sources)
            sentiment_score = np.random.normal(0.2, 0.5)
            
            # Generate realistic feedback based on sentiment
            if sentiment_score > 0.3:
                feedbacks = [
                    "Great product and excellent support team!",
                    "Very satisfied with the service quality",
                    "Outstanding features and user-friendly interface",
                    "Prompt response and helpful solutions",
                    "Excellent customer experience overall",
                    "Highly recommend this product to others",
                    "Amazing service and quick resolution times",
                    "Professional team with great expertise"
                ]
            elif sentiment_score < -0.3:
                feedbacks = [
                    "Poor customer service experience",
                    "Facing issues with the latest update",
                    "Frustrated with the billing process",
                    "Disappointed with product reliability",
                    "Slow response time from support",
                    "Difficult to use interface",
                    "Unreliable service with frequent downtime",
                    "Poor quality and bad customer support"
                ]
            else:
                feedbacks = [
                    "Average experience, could be better",
                    "The product meets basic requirements",
                    "Satisfactory performance overall",
                    "Adequate for our current needs",
                    "Standard features work as expected",
                    "Reasonable pricing for what you get",
                    "Acceptable but needs improvement",
                    "Moderate experience with some issues"
                ]
            
            data.append({
                'date': datetime.now() - timedelta(days=np.random.randint(1, 90)),
                'stakeholder_group': stakeholder,
                'source': source,
                'feedback_text': np.random.choice(feedbacks),
                'priority': np.random.choice(['High', 'Medium', 'Low']),
                'sentiment_score': max(-1, min(1, sentiment_score))
            })
        
        return pd.DataFrame(data)
    
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
        """Create a sentiment gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Sentiment Score"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightcoral"},
                    {'range': [-0.3, 0.3], 'color': "lightyellow"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        fig.update_layout(height=300)
        return fig
    
    def create_sentiment_distribution(self, metrics):
        """Create sentiment distribution pie chart"""
        labels = ['Positive', 'Neutral', 'Negative']
        values = [metrics['positive_pct'], metrics['neutral_pct'], metrics['negative_pct']]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        fig = px.pie(
            values=values, 
            names=labels, 
            color=labels,
            color_discrete_map=dict(zip(labels, colors)),
            title="Sentiment Distribution"
        )
        return fig
    
    def create_trend_chart(self, df):
        """Create sentiment trend over time"""
        df['date'] = pd.to_datetime(df['date'])
        daily_avg = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        
        fig = px.line(
            daily_avg, 
            x='date', 
            y='sentiment_score',
            title="Sentiment Trend Over Time",
            labels={'sentiment_score': 'Average Sentiment', 'date': 'Date'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        return fig
    
    def create_top_words_chart(self, df, sentiment_type='positive', top_n=10):
        """Create bar chart of top words"""
        if sentiment_type == 'positive':
            text_data = ' '.join(df[df['sentiment_score'] > 0.1]['feedback_text'])
            color = '#2ecc71'
            title = f"Top Words in Positive Feedback"
        elif sentiment_type == 'negative':
            text_data = ' '.join(df[df['sentiment_score'] < -0.1]['feedback_text'])
            color = '#e74c3c'
            title = f"Top Words in Negative Feedback"
        else:
            text_data = ' '.join(df[(df['sentiment_score'] >= -0.1) & (df['sentiment_score'] <= 0.1)]['feedback_text'])
            color = '#f39c12'
            title = f"Top Words in Neutral Feedback"
        
        if not text_data.strip():
            return None
        
        # Extract words and count frequencies
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_data.lower())
        stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'have', 'has', 'was', 
                     'were', 'are', 'is', 'you', 'your', 'our', 'they', 'their', 'been',
                     'from', 'have', 'that', 'this', 'with', 'they', 'what', 'your',
                     'when', 'which', 'were', 'been', 'will', 'would', 'should',
                     'could', 'about', 'into', 'through', 'during', 'before'}
        filtered_words = [word for word in words if word not in stop_words]
        
        word_freq = Counter(filtered_words).most_common(top_n)
        
        if not word_freq:
            return None
            
        words_df = pd.DataFrame(word_freq, columns=['word', 'frequency'])
        
        fig = px.bar(
            words_df,
            x='frequency',
            y='word',
            orientation='h',
            title=title,
            color_discrete_sequence=[color]
        )
        fig.update_layout(showlegend=False, height=300)
        return fig

    def generate_action_plan(self, df):
        """Generate automated action plans based on sentiment analysis"""
        actions = []
        
        # Analyze by stakeholder group
        for group in df['stakeholder_group'].unique():
            group_data = df[df['stakeholder_group'] == group]
            avg_sentiment = group_data['sentiment_score'].mean()
            negative_count = len(group_data[group_data['sentiment_score'] < -0.3])
            total_feedback = len(group_data)
            
            if avg_sentiment < -0.2 or (negative_count / total_feedback > 0.3 and total_feedback > 5):
                priority = "HIGH" if avg_sentiment < -0.4 else "MEDIUM"
                
                # Get top negative themes using TextBlob for more insight
                negative_feedback = group_data[group_data['sentiment_score'] < -0.1]['feedback_text']
                common_issues = []
                
                for feedback in negative_feedback.head(5):  # Analyze top 5 negative feedbacks
                    try:
                        blob = TextBlob(str(feedback))
                        # Extract noun phrases as potential issues
                        noun_phrases = blob.noun_phrases
                        common_issues.extend(noun_phrases)
                    except:
                        pass
                
                # Count most common issues
                issue_counts = Counter(common_issues).most_common(3)
                issues_display = ', '.join([f"{issue} ({count})" for issue, count in issue_counts]) if issue_counts else 'General dissatisfaction'
                
                actions.append({
                    'stakeholder_group': group,
                    'priority': priority,
                    'avg_sentiment': avg_sentiment,
                    'negative_feedback_count': negative_count,
                    'common_issues': issues_display,
                    'recommended_actions': [
                        f"Schedule meeting with {group} representatives",
                        f"Conduct root cause analysis for: {issues_display}",
                        f"Develop improvement action plan with timeline",
                        f"Assign dedicated relationship manager",
                        f"Monitor sentiment weekly for improvements"
                    ],
                    'timeline': "1-2 weeks" if priority == "HIGH" else "2-4 weeks"
                })
        
        return pd.DataFrame(actions)

def main():
    st.markdown('<h1 class="main-header">📊 Stakeholder Sentiment Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = SentimentDashboard()
    
    # Sidebar for filters and uploads
    with st.sidebar:
        st.header("Data Configuration")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Analyze sentiment if not already present
                if 'sentiment_score' not in df.columns and 'feedback_text' in df.columns:
                    df = dashboard.analyzer.analyze_dataframe(df)
                st.success("✅ File uploaded and analyzed with TextBlob!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.info("Using sample data instead")
                df = dashboard.load_sample_data()
        else:
            st.info("Using sample data for demonstration")
            df = dashboard.load_sample_data()
        
        st.header("Filters")
        
        # Date filter
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Stakeholder group filter
        stakeholder_groups = st.multiselect(
            "Stakeholder Groups",
            options=df['stakeholder_group'].unique(),
            default=df['stakeholder_group'].unique()
        )
        
        # Source filter
        if 'source' in df.columns:
            sources = st.multiselect(
                "Data Sources",
                options=df['source'].unique(),
                default=df['source'].unique()
            )
        else:
            sources = []
    
    # Apply filters
    if 'date' in df.columns:
        df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    
    if stakeholder_groups:
        df = df[df['stakeholder_group'].isin(stakeholder_groups)]
    
    if sources and 'source' in df.columns:
        df = df[df['source'].isin(sources)]
    
    # Calculate metrics
    metrics = dashboard.calculate_metrics(df)
    
    # Key Metrics Row
    st.subheader("📈 Key Metrics (Powered by TextBlob NLP)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_color = "normal"
        if metrics['avg_sentiment'] > 0.3:
            sentiment_color = "normal"
        elif metrics['avg_sentiment'] < -0.3:
            sentiment_color = "inverse"
            
        st.metric(
            "Overall Sentiment Score",
            f"{metrics['avg_sentiment']:.2f}",
            delta=f"{metrics['avg_sentiment']:.2f} vs Neutral",
            delta_color=sentiment_color
        )
    
    with col2:
        st.metric(
            "Positive Feedback",
            f"{metrics['positive_pct']:.1f}%",
            delta=f"{metrics['positive_pct'] - 33:.1f}% vs expected"
        )
    
    with col3:
        st.metric(
            "Negative Feedback",
            f"{metrics['negative_pct']:.1f}%",
            delta=f"{metrics['negative_pct'] - 33:.1f}% vs expected",
            delta_color="inverse"
        )
    
    with col4:
        st.metric("Total Feedback", metrics['total_feedback'])
    
    # Main Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(dashboard.create_sentiment_gauge(metrics['avg_sentiment']), 
                       use_container_width=True)
    
    with col2:
        st.plotly_chart(dashboard.create_sentiment_distribution(metrics), 
                       use_container_width=True)
    
    # Trend Analysis
    st.plotly_chart(dashboard.create_trend_chart(df), use_container_width=True)
    
    # Stakeholder Group Analysis
    st.subheader("👥 Stakeholder Group Analysis")
    
    group_sentiment = df.groupby('stakeholder_group')['sentiment_score'].agg(['mean', 'count']).reset_index()
    group_sentiment = group_sentiment.rename(columns={'mean': 'avg_sentiment', 'count': 'feedback_count'})
    
    fig = px.bar(
        group_sentiment,
        x='stakeholder_group',
        y='avg_sentiment',
        color='avg_sentiment',
        color_continuous_scale='RdYlGn',
        title="Average Sentiment by Stakeholder Group",
        text='avg_sentiment',
        hover_data=['feedback_count']
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Words Analysis
    st.subheader("🔤 Top Words Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        words_fig = dashboard.create_top_words_chart(df, 'positive')
        if words_fig:
            st.plotly_chart(words_fig, use_container_width=True)
        else:
            st.info("No positive feedback available")
    
    with col2:
        words_fig = dashboard.create_top_words_chart(df, 'neutral')
        if words_fig:
            st.plotly_chart(words_fig, use_container_width=True)
        else:
            st.info("No neutral feedback available")
    
    with col3:
        words_fig = dashboard.create_top_words_chart(df, 'negative')
        if words_fig:
            st.plotly_chart(words_fig, use_container_width=True)
        else:
            st.info("No negative feedback available")
    
    # Action Plan
    st.subheader("📋 Automated Action Plan (TextBlob Enhanced)")
    
    action_plan = dashboard.generate_action_plan(df)
    
    if not action_plan.empty:
        st.markdown('<div class="critical-alert">', unsafe_allow_html=True)
        st.warning(f"🚨 {len(action_plan)} critical issue(s) requiring attention")
        st.markdown('</div>', unsafe_allow_html=True)
        
        for _, action in action_plan.iterrows():
            with st.expander(f"🔴 {action['stakeholder_group']} - {action['priority']} Priority (Sentiment: {action['avg_sentiment']:.2f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Average Sentiment:** {action['avg_sentiment']:.2f}")
                    st.write(f"**Negative Feedback Count:** {action['negative_feedback_count']}")
                    st.write(f"**Common Issues:** {action['common_issues']}")
                
                with col2:
                    st.write(f"**Timeline:** {action['timeline']}")
                    st.write(f"**Priority:** {action['priority']}")
                
                st.write("**Recommended Actions:**")
                for i, step in enumerate(action['recommended_actions'], 1):
                    st.write(f"{i}. {step}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"Assign Team", key=f"assign_{action['stakeholder_group']}"):
                        st.success(f"Team assigned for {action['stakeholder_group']}")
                with col2:
                    if st.button(f"Schedule Meeting", key=f"meeting_{action['stakeholder_group']}"):
                        st.success(f"Meeting scheduled for {action['stakeholder_group']}")
                with col3:
                    if st.button(f"Mark Complete", key=f"complete_{action['stakeholder_group']}"):
                        st.success(f"Action completed for {action['stakeholder_group']}")
    else:
        st.success("✅ No critical issues detected. Current sentiment levels are within acceptable ranges.")
    
    # Raw Data Explorer
    st.subheader("📊 Data Explorer")
    
    with st.expander("View Raw Data with TextBlob Analysis"):
        st.dataframe(df)
        
        # Download processed data
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Processed Data",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
