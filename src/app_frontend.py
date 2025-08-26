# src/app_frontend.py
import requests
import streamlit as st
import pandas as pd
from calendar import month_name

st.set_page_config(page_title="Disney Reviews Q&A", layout="wide")
st.title("Disney Reviews Q&A")

api_url = st.sidebar.text_input("API URL", "http://localhost:8000/ask")
top_k = st.sidebar.slider("Top-K snippets", 4, 30, 12, step=2)

q_default = "What do visitors from Australia say about Disneyland in HongKong?"
question = st.text_input("Ask a question", value=q_default)

if st.button("Ask"):
    with st.spinner("Calling API…"):
        resp = requests.post(api_url, json={"question": question, "top_k": top_k})
    if resp.status_code != 200:
        st.error(f"API error: {resp.status_code} {resp.text}")
    else:
        out = resp.json()
        st.subheader("Answer")
        st.write(out["answer"])

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.markdown("**Filters applied**")
            st.json(out.get("filters", {}))
        with c2:
            st.markdown("**Parsed question**")
            st.json(out.get("parsed", {}))
        with c3:
            st.markdown("**Meta**")
            st.json(out.get("_meta", {}))

        st.markdown("### Stats")
        stats = out["stats"]
        n = stats["n"]
        avg = stats["avg_rating"]
        pos = stats["pos_share"]
        st.write(f"**n = {n}**  |  **avg rating = {avg:.2f}**  |  **positive = {pos*100:.1f}%**" if avg is not None else f"**n = {n}**")

        # Month pretty labels
        by_month = pd.DataFrame(stats["by_month"])
        if not by_month.empty and "Month" in by_month.columns:
            by_month["Month"] = by_month["Month"].apply(lambda m: month_name[int(m)] if pd.notna(m) and m>0 else m)
            st.write("**By month**")
            st.dataframe(by_month, use_container_width=True)

        by_season = pd.DataFrame(stats["by_season"])
        if not by_season.empty:
            st.write("**By season**")
            st.dataframe(by_season, use_container_width=True)

        topics = pd.DataFrame(stats["topic_counts"])
        if not topics.empty:
            st.write("**Topic counts**")
            topics["share"] = (topics["share"]*100).round(1)
            st.dataframe(topics, use_container_width=True)

        st.markdown("### Evidence (top snippets)")
        for s in out.get("snippets_used", []):
            header = f"**Review {s.get('Review_ID')}** | {s.get('Branch')} | {s.get('Reviewer_Location')} | "
            ym = f"{int(s['Year']) if s.get('Year') else ''}-{int(s['Month']) if s.get('Month') else ''}"
            header += f"{ym} | Rating {s.get('Rating')}"
            st.markdown(header)
            st.write(s.get("snippet", ""))
