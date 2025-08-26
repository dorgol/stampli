#!/usr/bin/env python3
"""
Streamlit UI for Disney Reviews QA
Run:
    streamlit run app.py
"""

import requests
import streamlit as st

API_ASK   = "http://localhost:8001/ask"
API_STATS = "http://localhost:8001/stats"

st.set_page_config(page_title="Disney Reviews QA", layout="wide")
st.title("🎢 Disney Reviews QA System")

# --- Input ---
question = st.text_area("Ask a question:", "What do visitors from Australia say about Disneyland in HongKong?")
top_k = st.slider("Top-K snippets to retrieve", 4, 20, 8)

if st.button("Ask"):
    with st.spinner("Querying QA server..."):
        resp = requests.post(API_ASK, json={"question": question, "top_k": top_k})
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
                import pandas as pd

                st.subheader("Aggregate Statistics")

                stats = data.get("stats", {})  # expects server to include stats in /ask response
                if not stats:
                    st.info("No stats available for this query.")
                else:
                    # --- Headline metrics ---
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total matching reviews", f"{stats.get('n') or stats.get('n_matching_reviews') or 0}")
                    avg = stats.get("avg_rating")
                    c2.metric("Avg rating", f"{avg:.2f}" if isinstance(avg, (int, float)) else "—")
                    pos = stats.get("pos_share")
                    c3.metric("Positive share (≥4⭐)", f"{pos * 100:.1f}%" if isinstance(pos, (int, float)) else "—")

                    # Raw JSON (useful for debugging)
                    with st.expander("Show raw stats JSON", expanded=False):
                        st.json(stats)

                    # --- Monthly ratings / positive share ---
                    if stats.get("by_month"):
                        df_month = pd.DataFrame(stats["by_month"])
                        if not df_month.empty and {"Month", "avg_rating", "pos_share"}.issubset(df_month.columns):
                            st.write("### 📅 Ratings by Month")
                            # Normalize types just in case
                            df_month["Month"] = pd.to_numeric(df_month["Month"], errors="coerce")
                            df_month = df_month.dropna(subset=["Month"]).sort_values("Month")
                            st.bar_chart(df_month.set_index("Month")[["avg_rating", "pos_share"]])

                    # --- Seasonal positive share ---
                    if stats.get("by_season"):
                        df_season = pd.DataFrame(stats["by_season"])
                        if not df_season.empty and {"Season", "pos_share"}.issubset(df_season.columns):
                            st.write("### 🍂 Seasonal Positive Share")
                            st.bar_chart(df_season.set_index("Season")[["pos_share"]])

                    # --- Cluster distribution (counts) ---
                    if stats.get("by_cluster_label"):
                        df_cl = pd.DataFrame(stats["by_cluster_label"])
                        if not df_cl.empty and {"cluster_label", "count"}.issubset(df_cl.columns):
                            st.write("### 🧩 Top Clusters (Count)")
                            st.bar_chart(df_cl.set_index("cluster_label")[["count"]])

                    # --- Cluster average rating (optional) ---
                    if stats.get("by_cluster_rating"):
                        df_cr = pd.DataFrame(stats["by_cluster_rating"])
                        if not df_cr.empty and {"cluster_label", "avg_rating"}.issubset(df_cr.columns):
                            st.write("### ⭐ Average Rating by Cluster")
                            st.bar_chart(df_cr.set_index("cluster_label")[["avg_rating"]])

                    # --- Topic counts (regex hits over text) ---
                    if stats.get("topic_counts"):
                        df_topics = pd.DataFrame(stats["topic_counts"])
                        if not df_topics.empty and {"topic_regex", "count"}.issubset(df_topics.columns):
                            st.write("### 🔎 Topics Mentioned")
                            st.bar_chart(df_topics.set_index("topic_regex")[["count"]])

            # --- Snippets ---
            with tab4:
                st.subheader("Retrieved Snippets")
                for snip in data["snippets"]:
                    st.markdown(f"**ID {snip['Review_ID']}** | {snip.get('Branch')} | {snip.get('Reviewer_Location')} | ⭐ {snip.get('Rating')}")
                    st.write(snip.get("snippet", ""))
                    st.caption(f"Distance: {snip.get('distance'):.3f}")
                    st.markdown("---")
