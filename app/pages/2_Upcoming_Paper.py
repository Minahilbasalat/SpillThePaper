import bootstrap  # noqa: F401
import streamlit as st
import theme

theme.setup("Upcoming paper", active="upcoming")
theme.title("Prediction desk", "Predict the", "next paper")
st.write("Built in the next steps.")