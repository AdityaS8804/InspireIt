import streamlit as st
from snowflake.core import Root
from snowflake.cortex import Complete
from snowflake.snowpark.context import get_active_session
from typing import List, Dict, Any
import json

# Constants
MODELS = [
    "mistral-large2",
    "llama3.1-70b",
    "llama3.1-8b",
]

def init_session_state():
    """Initialize all session state variables"""
    if "specifications" not in st.session_state:
        st.session_state.specifications = ""
    if "generate_new" not in st.session_state:
        st.session_state.generate_new = False
    # Change this to only set if not present
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if 'domain_inputs' not in st.session_state:
        st.session_state.domain_inputs = ['']
    if 'ideas' not in st.session_state:
        st.session_state.ideas = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'previous_prompt' not in st.session_state:
        st.session_state.previous_prompt = ""
    if 'selected_idea' not in st.session_state:
        st.session_state.selected_idea = None
    if 'final_idea' not in st.session_state:
        st.session_state.final_idea = None
    if "navigating_to_final" not in st.session_state:
        st.session_state.navigating_to_final = False

def init_service_metadata():
    """Initialize Cortex search service metadata"""
    if "service_metadata" not in st.session_state:
        services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
        service_metadata = []
        if services:
            for s in services:
                svc_name = s["name"]
                svc_search_col = session.sql(
                    f"DESC CORTEX SEARCH SERVICE {svc_name};"
                ).collect()[0]["search_column"]
                service_metadata.append(
                    {"name": svc_name, "search_column": svc_search_col}
                )
        st.session_state.service_metadata = service_metadata

def init_config_options():
    """Initialize sidebar configuration options"""
    st.sidebar.selectbox(
        "Select cortex search service:",
        [s["name"] for s in st.session_state.service_metadata],
        key="selected_cortex_search_service",
    )

    if st.sidebar.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.ideas = []
        st.session_state.domain_inputs = ['']
        st.session_state.chat_history = []
        st.session_state.page = "home"
        st.rerun()

    st.sidebar.toggle("Debug", key="debug", value=False)
    st.sidebar.toggle("Use chat history", key="use_chat_history", value=True)

    with st.sidebar.expander("Advanced options"):
        st.selectbox("Select model:", MODELS, key="model_name")
        st.number_input(
            "Select number of context chunks",
            value=5,
            key="num_retrieved_chunks",
            min_value=1,
            max_value=10,
        )
        st.number_input(
            "Select number of messages to use in chat history",
            value=5,
            key="num_chat_messages",
            min_value=1,
            max_value=10,
        )

def query_cortex_search_service(query, columns=[], filter={}):
    """Query the Cortex search service"""
    db, schema = session.get_current_database(), session.get_current_schema()

    cortex_search_service = (
        root.databases[db]
        .schemas[schema]
        .cortex_search_services[st.session_state.selected_cortex_search_service]
    )

    context_documents = cortex_search_service.search(
        query, columns=columns, filter=filter, limit=st.session_state.num_retrieved_chunks,
    )
    results = context_documents.results

    service_metadata = st.session_state.service_metadata
    search_col = [s["search_column"] for s in service_metadata
                    if s["name"] == st.session_state.selected_cortex_search_service][0].lower()

    context_str = ""
    for i, r in enumerate(results):
        context_str += f"Context document {i+1}: {r[search_col]} \n" + "\n"

    if st.session_state.debug:
        st.sidebar.text_area("Context documents", context_str, height=500)

    return context_str, results

def complete(model, prompt):
    """Generate completion using Cortex"""
    return Complete(model, prompt).replace("$", "\$")

def generate_idea_prompt(domains: List[str], specifications: str) -> str:
    """Create RAG-enhanced prompt for idea generation"""
    context_str, results = query_cortex_search_service(
        " ".join(domains + [specifications]),
        columns=["chunk", "file_url", "relative_path"],
        filter={"@and": [{"@eq": {"language": "English"}}]},
    )
    
    return f"""
    [INST]
    As an AI research consultant, generate creative business ideas based on the following:
    
    Domains: {', '.join(domains)}
    User Specifications: {specifications}
    
    Consider this relevant context:
    {context_str}
    
    Provide your response in JSON format with the following structure:
    {{
        "ideas": [
            {{
                "title": "Idea title",
                "description": "Brief description",
                "opportunities": ["opp1", "opp2", ...],
                "drawbacks": ["drawback1", "drawback2", ...],
                "references": ["ref1", "ref2", ...],
                "paper_url": "URL of the reference paper"
            }}
        ]
    }}
    
    Generate 3 innovative ideas that combine elements from the specified domains.
    Include URLs for reference papers where available from the context.
    [/INST]
    """

def generate_final_paper_prompt(idea: str, topics: str) -> str:
    """Create prompt for final paper generation"""
    context_str, _ = query_cortex_search_service(
        f"{idea} {topics}",
        columns=["chunk", "file_url", "relative_path"],
        filter={"@and": [{"@eq": {"language": "English"}}]},
    )
    
    return f"""
    [INST]
    Generate a complete research paper outline based on the following idea and topics:
    
    Idea: {idea}
    Topics: {topics}
    
    Consider this relevant context:
    {context_str}
    
    Provide your response in JSON format with the following structure:
    {{
        "abstract": "Detailed abstract of the proposed research",
        "references": ["ref1", "ref2", ...],
        "opportunities": ["innovation1", "innovation2", ...]
    }}
    [/INST]
    """

def home_button():
    """Add home button to top-right corner"""
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()

def home_page():
    """Render home page with updated navigation"""
    st.markdown("""
        <div class="header">
            <h1>Inspire-It</h1>
            <p>Transform your ideas into innovation with AI-powered insights and analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="card">
                <h3>Generate Ideas</h3>
                <p>Let AI help you brainstorm innovative business ideas across multiple domains</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Get Started", key="get_idea_btn"):
            st.session_state.page = "get_idea"
            st.rerun()
    
    with col2:
        st.markdown("""
            <div class="card">
                <h3>Review & Analyze</h3>
                <p>Get detailed analysis and feedback on your business ideas</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Now", key="review_btn"):
            st.session_state.selected_idea = None  # Clear any previous selection
            st.session_state.page = "review_idea"
            st.rerun()
    
    with col3:
        st.markdown("""
            <div class="card">
                <h3>Explore More</h3>
                <p>Discover additional tools and resources for innovation</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Explore", key="explore_btn"):
            st.session_state.page = "explore"
            st.rerun()

def get_idea_page():
    """Enhanced idea generation page"""
    home_button()
    st.markdown('<div class="header"><h2>Generate New Idea</h2></div>', unsafe_allow_html=True)
    
    cols = st.columns([3, 1])
    
    with cols[0]:
        for i, domain in enumerate(st.session_state.domain_inputs):
            new_value = st.text_input(f"Domain {i+1}", value=domain, key=f"domain_{i}")
            st.session_state.domain_inputs[i] = new_value
    
    with cols[1]:
        if st.button("➕", key="add_domain"):
            st.session_state.domain_inputs.append('')
            st.rerun()
    
    specifications = st.text_area(
        "User Specifications", 
        value=st.session_state.previous_prompt,
        height=100, 
        key="specifications"
    )
        
    if st.button("Generate Ideas") or not st.session_state.get('generate_new', False):
        if any(st.session_state.domain_inputs) and specifications:
            generate_and_display_ideas(st.session_state.domain_inputs, specifications)
        else:
            st.warning("Please fill in at least one domain and specifications")

def generate_and_display_ideas(domains: List[str], specifications: str):
    """Generate and display ideas with enhanced formatting"""
    domains = [d for d in domains if d]
    prompt = generate_idea_prompt(domains, specifications)
    
    if st.session_state.debug:
        st.code(prompt, language="text")
    
    response = complete(st.session_state.model_name, prompt)
    
    try:
        cleaned_response = response.strip().replace("```json", "").replace("```", "")
        ideas_data = json.loads(cleaned_response)
        
        if "ideas" not in ideas_data:
            st.error("Invalid response format")
            return
        
        st.session_state.ideas = ideas_data["ideas"]
        
        for i, idea in enumerate(st.session_state.ideas):
            with st.container():
                # Reference Paper
                st.markdown('<h3 class="idea-section-header">Reference Paper</h3>', 
                          unsafe_allow_html=True)
                st.markdown(f'<div class="idea-text">{idea.get("title", "")}</div>', 
                          unsafe_allow_html=True)
                
                # Summary
                st.markdown('<h3 class="idea-section-header">Summary</h3>', 
                          unsafe_allow_html=True)
                st.markdown(f'<div class="idea-text">{idea.get("description", "")}</div>', 
                          unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<h3 class="idea-section-header">Drawbacks</h3>', 
                              unsafe_allow_html=True)
                    for drawback in idea.get("drawbacks", []):
                        st.markdown(f"- <span class='red-text'>{drawback}</span>", 
                                  unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<h3 class="idea-section-header">Opportunities</h3>', 
                              unsafe_allow_html=True)
                    for opportunity in idea.get("opportunities", []):
                        st.markdown(f"- <span class='green-text'>{opportunity}</span>", 
                                  unsafe_allow_html=True)
                
                # Store the current idea in session state before handling button clicks
                suggestion = st.text_area("Your Suggestion (If you have no suggestions, please click on Develop Idea)", 
                                      key=f"suggest_{i}", 
                                      height=100)
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("Submit Suggestion", key=f"submit_{i}"):
                        if suggestion.strip():
                            # Update session state
                            st.session_state.previous_prompt = specifications
                            st.session_state.specifications = f"{specifications}\nAdditional context: {suggestion}"
                            st.session_state.generate_new = True
                            # Important: Store suggestion in session state
                            st.session_state[f"suggestion_{i}"] = suggestion
                            st.rerun()
                
                with col2:
                    # IMPORTANT: Let's modify this section
                    if st.button("Develop Idea", key=f"develop_{i}"):
                        # First, set all the necessary state
                        st.session_state.selected_idea = idea
                        st.session_state.final_idea = {
                            "idea": idea.get("description", ""),
                            "topics": ", ".join(idea.get("references", []))
                        }
                        # Set generate_new to False to prevent rerun loop
                        st.session_state.generate_new = False
                        # Set the page LAST, right before rerun
                        st.session_state.page = "final_paper"
                        # Add a flag to ensure we're really trying to go to final paper
                        st.session_state.navigating_to_final = True
                        # Force the rerun
                        st.rerun()

                st.markdown("<hr style='margin: 2rem 0; border-color: #e2e8f0;'>", 
                          unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error generating ideas: {str(e)}")
        
def review_idea_page():
    """Enhanced review idea page"""
    home_button()
    st.markdown('<div class="header"><h2>Review Idea</h2></div>', unsafe_allow_html=True)
    
    # Pre-fill if coming from Develop Idea
    initial_idea = ""
    initial_topics = ""
    
    if st.session_state.selected_idea:
        idea = st.session_state.selected_idea
        st.markdown("### Previous Analysis")
        st.markdown(f"**Title:** {idea.get('title', '')}")
        st.markdown(f"**Summary:** {idea.get('description', '')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Drawbacks")
            for drawback in idea.get("drawbacks", []):
                st.markdown(f"- <span class='red-text'>{drawback}</span>", 
                          unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Opportunities")
            for opportunity in idea.get("opportunities", []):
                st.markdown(f"- <span class='green-text'>{opportunity}</span>", 
                          unsafe_allow_html=True)
        
        st.markdown("### References")
        for ref in idea.get('references', []):
            st.markdown(f"- {ref}")
        
        initial_idea = idea.get('description', '')
        initial_topics = ", ".join(idea.get('references', []))
    
    idea_text = st.text_area("Refine Your Idea", value=initial_idea, height=150)
    topics = st.text_area("Topics Covered", value=initial_topics, height=100)
    
    if st.button("Generate Complete Paper"):
        if idea_text and topics:
            st.session_state.final_idea = {
                "idea": idea_text,
                "topics": topics
            }
            st.session_state.page = "final_paper"
            st.rerun()
        else:
            st.warning("Please provide both idea details and topics")

def final_paper_page():
    """Final paper details page"""
    home_button()
    st.markdown('<div class="header"><h2>Complete Paper Details</h2></div>', 
                unsafe_allow_html=True)
    
    if st.session_state.final_idea:
        idea = st.session_state.final_idea
        prompt = generate_final_paper_prompt(idea['idea'], idea['topics'])
        response = complete(st.session_state.model_name, prompt)
        
        try:
            cleaned_response = response.strip().replace("```json", "").replace("```", "")
            paper_details = json.loads(cleaned_response)
            
            # Updated heading with new style
            st.markdown('<h3 class="paper-section-header">Abstract</h3>', 
                       unsafe_allow_html=True)
            st.markdown(f'<div class="paper-abstract">{paper_details.get("abstract", "")}</div>', 
                       unsafe_allow_html=True)
            
            # Updated heading with new style
            st.markdown('<h3 class="paper-section-header">References</h3>', 
                       unsafe_allow_html=True)
            refs_html = '<div class="paper-references"><ul>'
            for ref in paper_details.get('references', []):
                refs_html += f'<li>{ref}</li>'
            refs_html += '</ul></div>'
            st.markdown(refs_html, unsafe_allow_html=True)
            
            # Updated heading with new style
            st.markdown('<h3 class="paper-section-header">Innovation & Opportunities</h3>', 
                       unsafe_allow_html=True)
            opps_html = '<div class="paper-innovation"><ul>'
            for opp in paper_details.get('opportunities', []):
                opps_html += f'<li>{opp}</li>'
            opps_html += '</ul></div>'
            st.markdown(opps_html, unsafe_allow_html=True)
        except json.JSONDecodeError:
            st.error("Failed to generate paper details")
    else:
        st.warning("No idea selected for paper generation")

def explore_page():
    """Chatbot-style explore page with enhanced styling"""
    home_button()
    st.markdown('<div class="header"><h2>Explore Ideas</h2></div>', 
                unsafe_allow_html=True)
    
    # Display chat history with custom styling
    for message in st.session_state.chat_history:
        role_style = "chat-message-user" if message["role"] == "user" else "chat-message-assistant"
        st.markdown(
            f'<div class="chat-message {role_style}">{message["content"]}</div>',
            unsafe_allow_html=True
        )
    
    # Chat input
    if prompt := st.chat_input("What would you like to explore?"):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate response
        context_str, _ = query_cortex_search_service(
            prompt,
            columns=["chunk", "file_url", "relative_path"],
            filter={"@and": [{"@eq": {"language": "English"}}]},
        )
        
        full_prompt = f"""
        [INST]
        Consider this context:
        {context_str}
        
        User question: {prompt}
        
        Provide a helpful and informative response.
        [/INST]
        """
        
        response = complete(st.session_state.model_name, full_prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun()

def apply_custom_styles():
    """Apply custom CSS styles"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;500;600;700&display=swap');
        
        /* Reset Streamlit's default background */
        .stApp {
            background: #E5F3FD;
            margin: 0;
            padding: 0;
        }
        
        .main { 
            padding: 2rem;
            background-color: #E5F3FD;
        }
        
        .stButton button {
            width: 100%;
            border-radius: 8px;
            height: 3.5em;
            background: linear-gradient(to top, #60a5fa, #93c5fd);
            color: white;
            font-family: 'Source Sans Pro', sans-serif;
            font-weight: 500;
            font-color: 2563eb;
            border: none;
            transition: transform 0.2s ease;
            box-shadow: 0 2px 4px rgba(96, 165, 250, 0.2);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(96, 165, 250, 0.25);
            background: linear-gradient(to top, #3b82f6, #60a5fa);
        }
        
        .header {
            text-align: center;
            padding: 3rem 0;
            background: linear-gradient(to bottom, #f0f4ff 0%, #AECCE4 100%);
            margin-bottom: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        }
        
        .header h1 {
            font-family: 'Source Sans Pro', sans-serif;
            font-weight: 700;
            font-size: 3.5rem;
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.01em;
        }
        
        .header h2 {
            font-family: 'Source Sans Pro', sans-serif;
            font-weight: 600;
            font-size: 2.5rem;
            color: #1e3a8a;
            margin-bottom: 1rem;
        }
        
        .header p {
            font-family: 'Source Sans Pro', sans-serif;
            color: #475569;
            font-size: 1.2rem;
            line-height: 1.6;
            font-weight: 400;
        }
        
        .card {
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
            margin-bottom: 1.5rem;
            border: 1px solid #f1f5f9;
            background: linear-gradient(to bottom, #f0f4ff 0%, #AECCE4 100%);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .card h3 {
            font-family: 'Source Sans Pro', sans-serif;
            color: #1e3a8a;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .card p {
            font-family: 'Source Sans Pro', sans-serif;
            color: #475569;
            line-height: 1.6;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        .red-text { 
            font-family: 'Source Sans Pro', sans-serif;
            color: #dc2626;
            font-weight: 600;
        }
        
        .green-text { 
            font-family: 'Source Sans Pro', sans-serif;
            color: #059669;
            font-weight: 600;
        }
        
        .stTextInput input, .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            padding: 0.75rem;
            transition: all 0.2s ease;
            background: #ffffff;
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 1rem;
            color: #1e293b;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        /* Chat message styling */
        .stChatMessage {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .stChatMessage.user {
            background-color: #f0f4ff;
        }

        .stChatInput {
            border-radius: 8px;
            border: 2px solid #e2e8f0;
            padding: 0.75rem;
            margin-top: 1rem;
            width: 100%;
            font-family: 'Source Sans Pro', sans-serif;
        }

        
        [data-testid="stTextInput"] label,
        [data-testid="stTextArea"] label {
            color: #1e293b !important;  /* Dark slate blue color */
            font-weight: 600 !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Keep existing input/textarea styling */
        [data-testid="stTextInput"] input {
            color: #1e3a8a;  /* Dark blue color */
            font-weight: 500;
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
        }
        
        [data-testid="stTextArea"] textarea {
            color: #1e3a8a;  /* Dark blue color */
            font-weight: 500;
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
        }
        
        /* Add Domain button styling */
        .add-domain-btn {
            background: linear-gradient(to top, #60a5fa, #93c5fd);
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            margin-top: 0.5rem;
            width: auto !important;
        }
        
        .add-domain-btn:hover {
            background: linear-gradient(to top, #3b82f6, #60a5fa);
        }

         /* Updated heading styles for final paper page */
        .paper-section-header {
            color: #1e3a8a;  /* Deep blue */
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3b82f6;  /* Blue underline */
        }
        
        /* Updated text colors for final paper page */
        .paper-abstract {
            color: #1e293b;  /* Dark slate blue */
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 1.1rem;
            line-height: 1.8;
            margin-bottom: 2rem;
        }
        
        .paper-references {
            color: #334155;  /* Slate gray */
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 1rem;
            line-height: 1.6;
        }
        
        .paper-references li {
            margin-bottom: 0.5rem;
        }
        
        .paper-innovation {
            color: #047857;  /* Darker green for better visibility */
            font-family: 'Source Sans Pro', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            line-height: 1.6;
        }

        /* Idea section headers */
        .idea-section-header {
            color: #1e3a8a;  /* Deep blue */
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3b82f6;
        }

        /* Idea text styling */
        .idea-text {
            color: #1e293b;  /* Dark slate blue for better visibility */
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 1.1rem;
            line-height: 1.8;
            margin-bottom: 1rem;
        }

        /* Button styling */
        .stButton > button {
            height: 40px;
            margin: 0;
            padding: 0.5rem 1rem;
        }

        /* Textarea styling */
        .stTextArea textarea {
            margin-bottom: 1rem;
        }

        /* Button container */
        .button-container {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            align-items: center;
        }

        .button-container .stButton {
            flex: 1;
            margin: 0;
        }

        /* Reference link styling */
        .reference-link {
            color: #2563eb;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s ease;
        }

        .reference-link:hover {
            color: #1d4ed8;
            text-decoration: underline;
        }

        /* Input container */
        .input-container {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
            margin-bottom: 1rem;
        }

        .input-container .stTextInput {
            flex: 1;
        }

        /* Chat container */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 1rem;
        }
        
        /* Chat message common styles */
        .chat-message {
            padding: 1.25rem;
            margin-bottom: 1rem;
            border-radius: 12px;
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 1.1rem;
            line-height: 1.6;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            animation: fadeIn 0.3s ease-in-out;
        }
        
        /* User message styling */
        .chat-message-user {
            background: linear-gradient(to right, #bfdbfe, #93c5fd);
            color: #1e3a8a;
            margin-left: 2rem;
            margin-right: 0;
            border-top-right-radius: 4px;
        }
        
        /* Assistant message styling */
        .chat-message-assistant {
            background: #ffffff;
            color: #1e293b;
            margin-right: 2rem;
            margin-left: 0;
            border-top-left-radius: 4px;
            border: 1px solid #e2e8f0;
        }
        
        /* Chat input container */
        .stChatInputContainer {
            padding: 1rem;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-top: 2rem;
        }
        
        /* Chat input field */
        .stChatInput input {
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 1.1rem;
            padding: 1rem;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            width: 100%;
            transition: all 0.2s ease;
        }
        
        .stChatInput input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            outline: none;
        }
        
        /* Message animation */
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Placeholder text styling */
        .stChatInput input::placeholder {
            color: #94a3b8;
            font-family: 'Source Sans Pro', sans-serif;
        }
        
        /* Send button styling */
        button[data-testid="chatInputSubmitButton"] {
            background: linear-gradient(to top, #60a5fa, #93c5fd);
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            color: white;
            font-family: 'Source Sans Pro', sans-serif;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        button[data-testid="chatInputSubmitButton"]:hover {
            background: linear-gradient(to top, #3b82f6, #60a5fa);
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
        }
        
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    global session, root
    session = get_active_session()
    root = Root(session)
    
    st.set_page_config(page_title="InspireIt", page_icon="💡", layout="wide")
    apply_custom_styles()
    
    init_session_state()
    init_service_metadata()
    init_config_options()
    
    # Add navigation flag check
    if st.session_state.get("navigating_to_final", False):
        st.session_state.page = "final_paper"
        st.session_state.navigating_to_final = False  # Reset the flag
    
    pages = {
        "home": home_page,
        "get_idea": get_idea_page,
        "review_idea": review_idea_page,
        "final_paper": final_paper_page,
        "explore": explore_page
    }
    
    current_page = pages.get(st.session_state.page, home_page)
    current_page()

if __name__ == "__main__":
    main()
