import streamlit as st
from src.engine import get_chat_engine 

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


#We create two columns. [1, 1] means they are equal width.
col_action1, col_action2 = st.columns([1, 1])

with col_action1:
    # use_container_width=True makes the button stretch to fill its half of the screen
    st.link_button("🗺️ Open Surrey COSMOS Map", "https://cosmos.surrey.ca/external/", use_container_width=True)

with col_action2:
    if st.button("Clear & Refresh Chat", use_container_width=True):
        st.session_state.messages = []
        # Clear the chat memory in LlamaIndex when the user clicks refresh
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset()
        st.rerun()


#Initializing the engine with True Caching
@st.cache_resource(show_spinner="Loading the Surrey OCP database...")
def load_engine():
    return get_chat_engine()

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = load_engine()

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

# ==========================================
# Display chat history WITH NotebookLM Sources
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If this message has saved sources, render the expander
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Official Source Documents"):
                for source_str in message["sources"]:
                    st.markdown(f"{source_str}")

#Handle Chat Input (Bottom Bar)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

#Generate Assistant Response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        try:
            with st.status("Analyzing Surrey's documents...", expanded=True) as status:
                st.write("Searching through Official Community Plan...")
                
                user_prompt = st.session_state.messages[-1]["content"]
                
                # ==========================================
                # INVISIBLE RULES (Added Rule 6 for Citations)
                # ==========================================
                invisible_rules = (
                    "\n\n---\nCRITICAL INSTRUCTIONS TO FOLLOW RIGHT NOW:\n"
                    "1. JURISDICTION (Transit/Infrastructure): If the user asks about public transit operations, bus schedules, or SkyTrain construction, explain that TransLink manages transit delivery, while the City's OCP only plans land use. You MUST provide this exact clickable link: [TransLink Canada](https://www.translink.ca/).\n"
                    "2. SSMUH / BILL 44: If the user asks about Small-Scale Multi-Unit Housing, Bill 44, or building multiplexes/secondary suites on single-family lots, explain that BC has new legislation for this and provide this exact clickable link: [BC Small-Scale Multi-Unit Housing](https://www2.gov.bc.ca/gov/content/housing-tenancy/local-governments-and-housing/housing-initiatives/smale-scale-multi-unit-housing).\n"
                    "3. OCP vs. DEVELOPMENT: If the user asks about specific new construction projects or active development applications, explain that the OCP is a high-level 'vision' document. Direct them to the City's Development Inquiry portal for active construction information.\n"
                    "4. SPECIFIC ADDRESSES: If the user's question contains a specific street address, DO NOT guess or answer. Refuse politely and direct them to the official City of Surrey COSMOS mapping system (https://cosmos.surrey.ca/external/) for exact property designations.\n"
                    "5. LIABILITY DISCLAIMER: If the user's question asks about land use, density, or building heights generally, YOU MUST append this exact text at the very bottom of your answer: '*Disclaimer: The OCP is a high-level guiding document. For specific zoning regulations, legal allowances, or building permits for your property, please consult official City of Surrey staff.*'\n"
                    "6. INLINE CITATIONS: You MUST cite your claims using inline brackets like [1], [2], etc., corresponding to the numbered context chunks provided to you. Do not list the sources at the end of your text, just put the bracketed numbers immediately after the fact you are stating."
                )
                enforced_prompt = user_prompt + invisible_rules
                
                response_stream = st.session_state.chat_engine.stream_chat(enforced_prompt)
                
                st.write("Generating answer...")
                status.update(label="Analysis complete!", state="complete", expanded=False)
            
            #Stream the final response text (Watch the AI type the [1], [2] live!)
            full_response = st.write_stream(response_stream.response_gen)
            
            # ==========================================
            # NEW: NOTEBOOK-LM STYLE SOURCE REFERENCING
            # (With Clickable URL Mapping)
            # ==========================================
            
            # 1. Map your document filenames to their real URLs
            # Add all the exact file names you ingested into ChromaDB here!
            URL_MAP = {
                "Surrey_OCP_2050.pdf": "https://www.surrey.ca/sites/default/files/media/documents/OfficialCommunityPlanDraft_Surrey2050.pdf",
                "Policy Chapters.pdf": "https://engage.surrey.ca/41242/widgets/185164/documents/163363",
                "Land Use Summary.pdf": "https://engage.surrey.ca/41242/widgets/185164/documents/163359",
                "Phase 1.pdf": "https://engage.surrey.ca/41242/widgets/185164/documents/133988",
                "Phase 2.pdf": "https://engage.surrey.ca/41242/widgets/185164/documents/141839",
                "Phase 3.pdf": "https://engage.surrey.ca/41242/widgets/185164/documents/157693",
                "Phase 4.pdf": "https://engage.surrey.ca/41242/widgets/185164/documents/163625"
            }

            ordered_sources = []
            
            # Check if the response contains source nodes from ChromaDB
            if hasattr(response_stream, 'source_nodes') and response_stream.source_nodes:
                with st.expander("📄 View Official Source Documents"):
                    
                    # enumerate(..., start=1) creates the 1, 2, 3 counter
                    for index, source in enumerate(response_stream.source_nodes, start=1):
                        metadata = source.node.metadata
                        
                        file_name = metadata.get('file_name', 'Unknown Document')
                        page_num = metadata.get('page_label') or metadata.get('page_num') or metadata.get('page')
                        
                        # 2. Match the URL (fallback to general plans page if filename isn't in dictionary)
                        url = URL_MAP.get(file_name, "https://www.surrey.ca/your-government/plans-reports")
                        
                        # 3. Create the markdown format: "**[1]** [Surrey_OCP.pdf (Page 12)](https://...)"
                        if page_num:
                            link_text = f"{file_name} (Page {page_num})"
                            reference_string = f"**[{index}]** [{link_text}]({url})"
                        else:
                            reference_string = f"**[{index}]** [{file_name}]({url})"
                            
                        ordered_sources.append(reference_string)
                        st.markdown(reference_string)
            
            #Save the response AND the ordered sources to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": ordered_sources 
            })
            
        except Exception as e:
            st.error("The AI servers are currently experiencing high traffic. Please wait 60 seconds and try asking again!")
            print(f"Server Error Details: {e}")