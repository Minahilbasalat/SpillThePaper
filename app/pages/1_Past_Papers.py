import bootstrap  # noqa: F401
import streamlit as st
import theme

theme.setup("Past papers", active="past")
theme.title("The archive", "Past", "papers")
st.write("Built in the next steps.")