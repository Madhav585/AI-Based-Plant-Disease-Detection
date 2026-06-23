import streamlit as st

st.set_page_config(
    page_title="AI Based Plant Disease Detection",
    page_icon="🌱"
)

st.title("🌱 AI Based Plant Disease Detection")

st.subheader("Smart Crop Health Monitoring and Disease Diagnosis System")

st.markdown("""
### 👨‍💻 Developer Information

**Developer:** Madhav Kumar  
**Course:** B.Tech CSE (6th Semester)  
**College Project:** Minor Project  
**Organization:** SEA AUTO
""")

uploaded_file = st.file_uploader(
    "Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file)
    st.success("Leaf Image Uploaded Successfully")

st.markdown("---")
st.write("© 2026 Madhav Kumar | AI Based Plant Disease Detection")
