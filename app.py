import streamlit as st
from src.engine import get_query_engine

st.set_page_config(page_title="Surrey 2050 AI Assistant", layout="centered")

st.title("🏙️ Surrey OCP 2050 Explorer")
st.markdown("Ask questions about the Official Community Plan and Engagement Reports.")

# 1. Initialize the engine with True Caching
@st.cache_resource(show_spinner="Loading the Surrey 2050 database...")
def load_engine():
    return get_query_engine()

if "query_engine" not in st.session_state:
    st.session_state.query_engine = load_engine()

# 2. Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input & Processing
if prompt := st.chat_input("What would you like to know about Surrey's future?"):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream response
    with st.chat_message("assistant"):
        try:
            # 1. Create a "Thinking" status container
            with st.status("Analyzing Surrey's documents...", expanded=True) as status:
                st.write("Searching through Official Community Plan...")
                
                # Get the streaming response from the engine
                response_stream = st.session_state.query_engine.query(prompt)
                
                st.write("Generating answer...")
                
                # 2. Update status to 'complete' once the engine is ready
                status.update(label="Analysis complete!", state="complete", expanded=False)
            
            # 3. Stream the final response to the user
            full_response = st.write_stream(response_stream.response_gen)
            
            # Save the fully generated response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error("The AI servers are currently experiencing high traffic. Please wait 60 seconds and try asking again!")
            print(f"Server Error Details: {e}")