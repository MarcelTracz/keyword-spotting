import streamlit as st


st.set_page_config(
    page_title="Keyword Spotting",
    page_icon="🎙️",
    layout="wide"
)

st.write("# Keyword Spotting")

st.markdown(
    """
    The App uses neural networks to recognize specific spoken keywords in real-time, such as "Ok Google" or "Hey Siri." It ensures reliable detection even with background noise or continuous speech, enabling seamless interaction with voice-controlled systems.
    """
)

st.markdown("### Key Features:")

st.markdown(
    """
    🎧 **Real-time Audio Capture** - continuously processes incoming audio to detect specific keywords effortlessly.\n
    🧠 **Neural Network-based Recognition** - utilizes powerful neural networks to achieve high accuracy in keyword detection.\n
    🔊 **Robust to Background Noise** - effectively identifies keywords even with minimal and manageable background noise.\n
    """
)

st.markdown("### How It Works:")

st.markdown(
    """
    1️⃣ **Audio Capture** - records audio input in real-time from the user.\n
    2️⃣ **Pre-processing** - extracts key audio features for analysis.\n
    3️⃣ **Modeling** - applies a trained neural network model to predict the presence of predefined keywords.\n
    4️⃣ **Post-processing** - responds to recognized keywords with appropriate system actions.\n
    5️⃣ **GUI Feedback** - provides real-time feedback to the user through an interactive interface.\n
    """
)

st.markdown("### Challenges Tackled:")

st.markdown(
    """
    - **Background Noise**: Ensures reliable keyword detection even with environmental noise.  
    - **Speaker Variability**: Adapts to different speakers with varying tones and accents.  
    - **Continuous Speech**: Handles continuous and natural speech patterns effectively.  
    """
)
