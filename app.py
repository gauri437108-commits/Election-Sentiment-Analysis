import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

st.set_page_config(page_title="Election Sentiment Analysis", page_icon="🗳️")

st.title("🗳️ Election Sentiment Analysis")
st.write("Upload a CSV file containing election-related comments.")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    if "Comment" not in df.columns:
        st.error("CSV file must contain a column named 'Comment'")
    else:

        sentiments = []
        labels = []

        with st.spinner("Analyzing comments..."):

            for comment in df["Comment"]:
                result = model(str(comment))[0]

                if result["label"] == "POSITIVE":
                    sentiment = "Positive"
                else:
                    sentiment = "Negative"

                sentiments.append(sentiment)
                labels.append(result["label"])

        df["Sentiment"] = sentiments

        st.subheader("Analysis Result")
        st.dataframe(df)

        sentiment_count = df["Sentiment"].value_counts()

        fig, ax = plt.subplots()
        sentiment_count.plot(kind="bar", ax=ax)

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Comments")
        ax.set_title("Sentiment Analysis Result")

        st.pyplot(fig)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Result CSV",
            data=csv,
            file_name="sentiment_result.csv",
            mime="text/csv",
        )
