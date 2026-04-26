import streamlit as st
from src.engine import get_query_engine

st.set_page_config(page_title="Surrey OCP AI Assistant", layout="centered")

#UI CSS Injection to match The City of Surrey vibes anjay
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

with st.sidebar:
    st.header("About This Tool")
    st.markdown("This AI assistant helps residents navigate the Surrey 2050 Official Community Plan.")
    
    st.divider() 
    
    st.markdown("### Official Resources")
    st.markdown("Need to check the exact land-use designation for your specific property?")
    # Adds a clickable button directly to the Surrey map tool
    st.link_button("🗺️ Open Surrey COSMOS Lookup Tool", "https://cosmos.surrey.ca/external/")

    st.divider()

    st.markdown("### Session")
    if st.button("Clear & Refresh chat"):
        st.session_state.messages = []
        st.rerun()

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

prompt = st.chat_input("What would you like to know about Surrey's future?")

#Suggestion prompts for first time users
if not st.session_state.messages and not prompt:
    st.write("---")
    st.markdown("<h3 class='center-text'>Let's chat! What's on your mind?</h3>", unsafe_allow_html=True)
    st.markdown("<p class='center-text'>Try these Surrey 2050 OCP prompts:</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("**Housing & Growth:** How will the OCP help address housing affordability and where is the City focusing growth?"):
            st.session_state.messages.append({"role": "user", "content": "How will the OCP help address housing affordability and where is the City focusing growth?"})
            st.rerun()
        if st.button("**Transit & Infrastructure:** How is the OCP supporting active transportation and transit expansion?"):
            st.session_state.messages.append({"role": "user", "content": "How is the OCP supporting active transportation and transit expansion?"})
            st.rerun()
            
    with col2:
        if st.button("**Parks & Environment:** What are the long-term goals for protecting green spaces, parks, and the tree canopy?"):
            st.session_state.messages.append({"role": "user", "content": "What are the long-term goals for protecting green spaces, parks, and the tree canopy?"})
            st.rerun()
        if st.button("**Economy & Jobs:** How does the 2050 plan support local job creation and the development of commercial Town Centres?"):
            st.session_state.messages.append({"role": "user", "content": "How does the 2050 plan support local job creation and the development of commercial Town Centres?"})
            st.rerun()
    st.write("---")

#Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Handle Chat Input (Bottom Bar)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

#Generate Assistant Response (Triggers for BOTH buttons and text input)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        try:
            with st.status("Analyzing Surrey's documents...", expanded=True) as status:
                st.write("Searching through Official Community Plan...")
                
                #Get the pure prompt from the last user message (what the user sees)
                user_prompt = st.session_state.messages[-1]["content"]
                
                #Prompt Padding for certain user prompts to dirrect to more specific knowledge sources
                invisible_rules = (
                    "\n\n---\nCRITICAL INSTRUCTIONS TO FOLLOW RIGHT NOW:\n"
                    "1. If the user's question asks about land use, density, or building heights, YOU MUST append this exact text at the very bottom of your answer: '*Disclaimer: The OCP is a high-level guiding document. For specific zoning regulations, legal allowances, or building permits for your property, please consult official City of Surrey staff.*'\n"
                    "2. If the user's question contains a specific street address, DO NOT answer it. Refuse and direct them to the City of Surrey COSMOS mapping system."
                )
                enforced_prompt = user_prompt + invisible_rules
                
                #Send the ENFORCED prompt to the LLM, not the pure user_prompt
                response_stream = st.session_state.query_engine.query(enforced_prompt)
                
                st.write("Generating answer...")
                status.update(label="Analysis complete!", state="complete", expanded=False)
            
            #Stream the final response
            full_response = st.write_stream(response_stream.response_gen)
            
            #Save the response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error("The AI servers are currently experiencing high traffic. Please wait 60 seconds and try asking again!")
            print(f"Server Error Details: {e}")