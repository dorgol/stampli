#!/usr/bin/env python3
"""
Streamlit UI for Disney Reviews QA
Run:
    streamlit run app.py
"""

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8001/qa"


st.set_page_config(page_title="Disney Reviews QA", layout="wide")
st.title("🎢 Disney Reviews QA System")

# --- Input ---
question = st.text_area("Ask a question:", "What do visitors from Australia say about Disneyland in HongKong?")
top_k = st.slider("Top-K snippets to retrieve", 4, 20, 8)

if st.button("Ask"):
    with st.spinner("Querying QA server..."):
        resp = requests.post(API_URL, json={"question": question, "top_k": top_k})
        if resp.status_code != 200:
            st.error(f"Error {resp.status_code}: {resp.text}")
        else:
            data = resp.json()

            # Tabs for debugging pipeline
            tab1, tab2, tab3, tab4 = st.tabs(["💡 Answer", "📌 Filters", "📊 Stats", "📄 Snippets"])

            # --- Answer ---
            with tab1:
                st.subheader("Synthesized Answer")
                st.write(data["answer"])
                if data.get("citations"):
                    st.caption(f"Cited Reviews: {data['citations']}")

            # --- Filters ---
            with tab2:
                st.subheader("Parsed Filters")
                st.json(data["filters"])

            # --- Stats ---
            with tab3:
                st.subheader("Aggregate Statistics")
                stats = data["stats"]

                # JSON for reference
                st.json(stats)

                # Chart: monthly ratings
                if stats.get("by_month"):
                    df_month = pd.DataFrame(stats["by_month"])
                    st.write("### 📅 Ratings by Month")
                    st.bar_chart(df_month.set_index("Month")[["avg_rating", "pos_share"]])

                # Chart: seasonal positive share
                if stats.get("by_season"):
                    df_season = pd.DataFrame(stats["by_season"])
                    st.write("### 🍂 Seasonal Positive Share")
                    st.bar_chart(df_season.set_index("Season")[["pos_share"]])

                # Chart: topic counts
                if stats.get("topic_counts"):
                    df_topics = pd.DataFrame(stats["topic_counts"])
                    st.write("### 🔎 Topics Mentioned")
                    st.bar_chart(df_topics.set_index("topic_regex")[["count"]])

            # --- Snippets ---
            with tab4:
                st.subheader("Retrieved Snippets")
                for snip in data["snippets_used"]:
                    st.markdown(f"**ID {snip['Review_ID']}** | {snip.get('Branch')} | {snip.get('Reviewer_Location')} | ⭐ {snip.get('Rating')}")
                    st.write(snip.get("snippet", ""))
                    st.caption(f"Distance: {snip.get('distance'):.3f}")
                    st.markdown("---")
