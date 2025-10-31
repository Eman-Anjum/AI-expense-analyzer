import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from groq import Groq
import os
from datetime import datetime

st.set_page_config(page_title="AI Expense Analyzer", layout="wide")
st.title("💸 AI Expense Analyzer")
st.markdown("Track, visualize, and get AI-powered recommendations for your expenses.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("⚠️ Please set your GROQ_API_KEY environment variable.")
else:
    client = Groq(api_key=GROQ_API_KEY)

if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Date", "Amount", "Category", "Description"])

st.sidebar.header("Add New Expense")
with st.sidebar.form("expense_form", clear_on_submit=True):
    date = st.date_input("Date", datetime.now())
    amount = st.number_input("Amount ($)", min_value=0.0, step=0.01)
    category = st.selectbox("Category", ["Food", "Transport", "Shopping", "Entertainment", "Bills", "Other"])
    description = st.text_area("Description")
    submit = st.form_submit_button("Add Expense")

if submit:
    new_data = pd.DataFrame(
    [[pd.to_datetime(date), amount, category, description]],
    columns=["Date", "Amount", "Category", "Description"]
)
    st.session_state.expenses = pd.concat([st.session_state.expenses, new_data], ignore_index=True)
    st.success("Expense added successfully!")

expenses = st.session_state.expenses

if not expenses.empty:
    st.subheader("📊 Expense Overview")
    st.dataframe(expenses, use_container_width=True)

    category_summary = expenses.groupby("Category")["Amount"].sum().reset_index()
    fig_pie = px.pie(category_summary, values="Amount", names="Category", title="Expenses by Category",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_pie, use_container_width=True)

    expenses["Date"] = pd.to_datetime(expenses["Date"])
    daily_trend = expenses.groupby("Date")["Amount"].sum().reset_index()
    fig_trend = px.line(daily_trend, x="Date", y="Amount", markers=True, title="Spending Trend Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)

    if GROQ_API_KEY:
        st.subheader("🤖 AI Expense Insights")
        expense_text = "\n".join(
            f"{row['Date'].strftime('%Y-%m-%d')} - ${row['Amount']:.2f} on {row['Category']} ({row['Description']})"
            for _, row in expenses.iterrows()
        )
        prompt = f"""
        You are a financial assistant. Analyze the following user expenses:
        {expense_text}
        1. Identify spending patterns or habits.
        2. Highlight when the user spent the most.
        3. Provide recommendations to optimize their spending.
        Keep the tone friendly and actionable.
        """
        with st.spinner("Analyzing your expenses with Groq AI..."):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                )
                ai_response = chat_completion.choices[0].message.content
                st.markdown(f"**AI Recommendations:**\n\n{ai_response}")
            except Exception as e:
                st.error(f"Error connecting to Groq API: {e}")
else:
    st.info("👈 Add some expenses in the sidebar to get started.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Groq LLM")
