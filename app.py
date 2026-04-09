import streamlit as st
from src.engine import get_query_engine

st.set_page_config(page_title="Surrey OCP AI Assistant", layout="centered")

#UI CSS Injection to match The City of Surrey vibes
st.markdown("""
    <style>
    /* Style the suggestion buttons to look like pills */
    div.stButton > button {
        border-radius: 20px;
        border: 2px solid #8CBF3F; /* Surrey Light Green */
        color: #2a2a2a;
        background-color: white;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #8CBF3F;
        color: white;
        border-color: #4F7900; /* Surrey Dark Green */
    }
    
    /* Center text */
    .center-text { text-align: center; }
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: #4F7900;
        font-weight: 700;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Surrey 2050 OCP Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p class='center-text'>Ask questions about the Official Community Plan and Engagement Reports.</p>", unsafe_allow_html=True)

#Initializing the engine with True Caching
@st.cache_resource(show_spinner="Loading the Surrey OCP database...")
def load_engine():
    return get_query_engine()

if "query_engine" not in st.session_state:
    st.session_state.query_engine = load_engine()

#Chat history setup
if "messages" not in st.session_state:
    st.session_state.messages = []

#Suggestion prompts for first time users
if not st.session_state.messages:
    st.write("---")
    st.markdown("<h3 class='center-text'>Let's chat! What's on your mind?</h3>", unsafe_allow_html=True)
    st.markdown("<p class='center-text'>Try these Surrey 2050 OCP prompts:</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What is the vision for Surrey 2050?"):
            st.session_state.messages.append({"role": "user", "content": "What is the vision for Surrey 2050?"})
            st.rerun()
        if st.button("How will public transit improve?"):
            st.session_state.messages.append({"role": "user", "content": "How will public transit improve?"})
            st.rerun()
            
    with col2:
        if st.button("Where are the new housing developments?"):
            st.session_state.messages.append({"role": "user", "content": "Where are the new housing developments?"})
            st.rerun()
        if st.button("What are the environmental goals?"):
            st.session_state.messages.append({"role": "user", "content": "What are the environmental goals?"})
            st.rerun()
    st.write("---")

# 4. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle Chat Input (Bottom Bar)
if prompt := st.chat_input("What would you like to know about Surrey's future?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# 6. Generate Assistant Response (Triggers for BOTH buttons and text input)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        try:
            with st.status("Analyzing Surrey's documents...", expanded=True) as status:
                st.write("Searching through Official Community Plan...")
                
                # Get the prompt from the last user message
                user_prompt = st.session_state.messages[-1]["content"]
                response_stream = st.session_state.query_engine.query(user_prompt)
                
                st.write("Generating answer...")
                status.update(label="Analysis complete!", state="complete", expanded=False)
            
            # Stream the final response
            full_response = st.write_stream(response_stream.response_gen)
            
            # Save the response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error("The AI servers are currently experiencing high traffic. Please wait 60 seconds and try asking again!")
            print(f"Server Error Details: {e}")