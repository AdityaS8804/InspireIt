import streamlit as st
from typing import List, Dict, Any
import json
import snowflake.connector

# Constants
MODELS = [
    "mistral-large2",
    "llama3.1-70b",
    "llama3.1-8b",
]
def init_snowflake():
    """Initialize Snowflake connection"""
    return snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
        role=st.secrets["snowflake"]["role"]
    )

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
    """Initialize Cortex search service metadata using Snowflake connection"""
    if "service_metadata" not in st.session_state:
        cursor = conn.cursor()
        try:
            services = cursor.execute("SHOW CORTEX SEARCH SERVICES;").fetchall()
            service_metadata = []
            if services:
                for s in services:
                    svc_name = s[1]  # Assuming name is in second column
                    svc_details = cursor.execute(
                        f"DESC CORTEX SEARCH SERVICE {svc_name};"
                    ).fetchone()
                    svc_search_col = svc_details[1]  # Assuming search_column is in second column
                    service_metadata.append(
                        {"name": svc_name, "search_column": svc_search_col}
                    )
            st.session_state.service_metadata = service_metadata
        finally:
            cursor.close()

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
    """Query the Cortex search service using Snowflake stored procedure"""
    cursor = conn.cursor()
    try:
        params = json.dumps({
            "query": query,
            "service_name": st.session_state.selected_cortex_search_service,
            "num_chunks": st.session_state.num_retrieved_chunks
        })
        result = cursor.execute(
            "CALL CORTEX_SEARCH_PROC(%s)",
            (params,)
        ).fetchone()[0]
        
        results = json.loads(result)
        
        # Format context string similar to original function
        context_str = ""
        for i, r in enumerate(results):
            context_str += f"Context document {i+1}: {r['chunk']} \n\n"
            
        if st.session_state.debug:
            st.sidebar.text_area("Context documents", context_str, height=500)
            
        return context_str, results
    finally:
        cursor.close()


def complete(model, prompt):
    """Generate completion using Snowflake stored procedure"""
    cursor = conn.cursor()
    try:
        params = json.dumps({
            "prompt": prompt,
            "model_name": model
        })
        result = cursor.execute(
            "CALL CORTEX_COMPLETE_PROC(%s)",
            (params,)
        ).fetchone()[0]
        return result.replace("$", "\$")
    finally:
        cursor.close()

def generate_idea_prompt(domains: List[str], specifications: str) -> str:
    """Create RAG-enhanced prompt for idea generation"""
    context_str, results = query_cortex_search_service(
        " ".join(domains + [specifications]),
        columns=["chunk", "file_url", "relative_path"],
        filter={"@and": [{"@eq": {"language": "English"}}]},
    )
    
    return f"""
    [INST]
    As an AI research consultant, generate creative research ideas based on the following:
    
    Domains: {', '.join(domains)}
    User Specifications: {specifications}
    
    Consider this relevant context:
    {context_str}
    
    Provide your response in JSON format with the following structure:
    {{
        "ideas": [
            {{
                "title": "Idea title",
                "description": "Very lengthy description",
                "opportunities": ["opp1", "opp2", ...],
                "drawbacks": ["drawback1", "drawback2", ...],
                "references": ["ref1", "ref2", ...],
                ""
            }}
        ]
    }}
    
    Generate 3 innovative ideas that combine elements from the specified domains.
    Include URLs for reference papers where available from the context.
    [/INST]
    """

def generate_summaries_paper(references: List[str]) -> str:
    """Generate summaries for referenced papers using Cortex"""
    if not references:
        return []
    
    # Join references into a search query
    search_query = " ".join(references)
    
    # Query Cortex search service for paper content
    context_str, results = query_cortex_search_service(
        search_query,
        columns=["chunk", "file_url", "relative_path"],
        filter={"@and": [{"@eq": {"language": "English"}}]},
    )
    
    # Create prompt for generating summaries
    prompt = f"""
    [INST]
    Based on the following context about research papers:
    {context_str}
    
    Generate brief summaries for the referenced papers. Include key findings and methodologies where available.
    
    Provide your response in JSON format with the following structure:
    {{
        "paper_summaries": [
            {{
                "title": "Paper title",
                "summary": "Brief summary of key findings and methodology"
            }}
        ]
    }}
    [/INST]
    """
    
    # Generate completion using Cortex
    response = complete(st.session_state.model_name, prompt)
    
    try:
        cleaned_response = response.strip().replace("```json", "").replace("```", "")
        summaries_data = json.loads(cleaned_response)
        return summaries_data.get("paper_summaries", [])
    except json.JSONDecodeError:
        return []


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

def develop_idea_prompt(idea: str) -> str:
    """
    Generate a prompt for developing a comprehensive analysis of a research idea.
    
    Parameters:
    idea (str): The research idea to be developed
    
    Returns:
    str: A formatted prompt for the LLM to analyze the idea
    """
    return f"""
    [INST]
    Analyze and develop the following research idea in detail:
    
    {idea}
    
    Provide your response in JSON format with the following structure:
    {{
        "brief_description": "A concise 2-3 sentence overview of the core idea",
        "methodology": [
            {{
                "phase": "name of research phase",
                "description": "brief description of what will be done",
                "key_techniques": ["technique1", "technique2", ...]
            }}
        ],
        "key_points": [
            {{
                "point": "main research point or finding",
                "significance": "why this point is important",
                "potential_impact": "expected impact in the field"
            }}
        ],
        "technical_requirements": [
            {{
                "requirement": "specific technical need",
                "justification": "why this is necessary",
                "alternatives": ["alternative1", "alternative2"]
            }}
        ],
        "innovation_areas": [
            {{
                "area": "specific area of innovation",
                "novelty_factor": "what makes this innovative",
                "existing_gaps": "gaps in current research this addresses"
            }}
        ]
    }}
    
    Focus on making each section:
    1. Specific and actionable
    2. Grounded in research feasibility
    3. Clear in its innovative aspects
    4. Technically sound
    
    Keep descriptions concise but informative.
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
    
    # Change to single container for better domain input layout
    for i, domain in enumerate(st.session_state.domain_inputs):
        # Use columns with different ratios for better spacing
        col1, col2, col3 = st.columns([3.5, 0.5, 1])  # Adjust the ratios as needed
        with col1:
            new_value = st.text_input(f"Domain {i+1}", value=domain, key=f"domain_{i}")
            st.session_state.domain_inputs[i] = new_value
            if i == len(st.session_state.domain_inputs) - 1:  # Only show + button for last input
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
                # Paper Title with updated styling
                st.markdown('<h3 class="idea-section-header">Research Paper</h3>', 
                          unsafe_allow_html=True)
                st.markdown(f'<div class="research-title">{idea.get("title", "")}</div>', 
                          unsafe_allow_html=True)
                
                # Idea Description
                st.markdown('<h3 class="idea-section-header">Idea Description</h3>', 
                          unsafe_allow_html=True)
                st.markdown(f'<div class="idea-description">{idea.get("description", "")}</div>', 
                          unsafe_allow_html=True)
                
                # Generate and display paper summaries
                references = idea.get("references", [])
                if references:
                    st.markdown('<h3 class="idea-section-header">Related Paper Summaries</h3>', 
                              unsafe_allow_html=True)
                    
                    summaries = generate_summaries_paper(references)
                    for summary in summaries:
                        with st.expander(f"📄 {summary.get('title', 'Paper Summary')}"):
                            st.markdown(f'<div class="summary-text"><b>Summary:</b> {summary.get("summary", "")}</div>', 
                                      unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<h3 class="idea-section-header">Drawbacks</h3>', 
                              unsafe_allow_html=True)
                    drawbacks_html = '<div class="drawbacks-list">'
                    for drawback in idea.get("drawbacks", []):
                        drawbacks_html += f'<div class="drawback-item">• {drawback}</div>'
                    drawbacks_html += '</div>'
                    st.markdown(drawbacks_html, unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<h3 class="idea-section-header">Opportunities</h3>', 
                              unsafe_allow_html=True)
                    opportunities_html = '<div class="opportunities-list">'
                    for opportunity in idea.get("opportunities", []):
                        opportunities_html += f'<div class="opportunity-item">• {opportunity}</div>'
                    opportunities_html += '</div>'
                    st.markdown(opportunities_html, unsafe_allow_html=True)
                
                if st.button("Develop Idea", key=f"develop_{i}"):
                    st.session_state.selected_idea = idea
                    st.session_state.final_idea = {
                        "idea": idea.get("description", ""),
                        "topics": ", ".join(idea.get("references", []))
                    }
                    st.session_state.generate_new = False
                    st.session_state.page = "final_paper"
                    st.session_state.navigating_to_final = True
                    st.rerun()

                st.markdown("<hr class='idea-separator'>", unsafe_allow_html=True)

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
    """Chatbot-style explore page"""
    home_button()
    st.markdown('<div class="header"><h2>Explore Ideas</h2></div>', 
                unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Enter your message"):
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
            background: #9ABDDC;
            margin: 0;
            padding: 0;
        }
        
        .main { 
            padding: 2rem;
            background-color: #9ABDDC;
        }
        
        /* Button Styles */
        .stButton button {
            width: 100%;
            border-radius: 8px;
            height: 3.5em;
            background: linear-gradient(to top, #3b82f6, #60a5fa);
            color: white;
            font-family: 'Source Sans Pro', sans-serif;
            font-weight: 500;
            font-color: #2563eb;
            border: none;
            transition: transform 0.2s ease;
            box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
            background: linear-gradient(to top, #2563eb, #3b82f6);
        }
        
        /* Header Styles */
        .header {
            text-align: center;
            padding: 3rem 0;
            background: linear-gradient(to bottom, #f8fafc 0%, #bfdbfe 100%);
            margin-bottom: 1.5rem;
            border-radius: 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        }
        
        .header h1 {
            font-family: 'Source Sans Pro', sans-serif !important;
            font-weight: 700 !important;
            font-size: 3.5rem !important;
            background: linear-gradient(135deg, #1e40af, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.01em;
            color: #1e40af !important;
            opacity: 1 !important;
        }
        
        .header h2 {
            font-family: 'Source Sans Pro', sans-serif !important;
            font-weight: 600 !important;
            font-size: 2.5rem !important;
            color: #1e40af !important;
            margin-bottom: 1rem !important;
            opacity: 1 !important;
        }
        
        .header h3 {
            color: #1e40af !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            opacity: 1 !important;
        }
        
        .header p {
            font-family: 'Source Sans Pro', sans-serif;
            color: #475569;
            font-size: 1.2rem;
            line-height: 1.6;
            font-weight: 400;
        }
        
        /* Card Styles */
        .card {
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
            margin-bottom: 1.5rem;
            border: 1px solid #f1f5f9;
            background: linear-gradient(to bottom, #f8fafc 0%, #bfdbfe 100%);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .card h3 {
            font-family: 'Source Sans Pro', sans-serif !important;
            color: #1e40af !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            opacity: 1 !important;
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
        
        /* Text Colors */
        .red-text { 
            font-family: 'Source Sans Pro', sans-serif !important;
            color: #991b1b !important;  /* Deeper, more visible red */
            font-weight: 600 !important;
            opacity: 1 !important;
        }
        
        .green-text { 
            font-family: 'Source Sans Pro', sans-serif !important;
            color: #065f46 !important;  /* Deeper, more visible green */
            font-weight: 600 !important;
            opacity: 1 !important;
        }
        
        /* Specifically target Domain and User Specifications labels */
        div[data-testid="stTextInput"] div[data-baseweb="label"] label,
        div[data-testid="stTextArea"] div[data-baseweb="label"] label {
            font-size: 2rem !important;
            color: #1e40af !important;
            font-weight: 700 !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            margin-bottom: 1rem !important;
            display: block !important;
            opacity: 1 !important;
            line-height: 2.5rem !important;
        }
        
        /* Override any conflicting styles */
        div[data-testid="stTextInput"] div[data-baseweb="label"],
        div[data-testid="stTextArea"] div[data-baseweb="label"] {
            font-size: 2rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* Updated Input Text Styles */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            color: #1e40af !important;
            font-weight: 500 !important;
            font-size: 1.1rem !important;  /* Increased from 1.1rem */
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 0.75rem !important;
            line-height: 1.5 !important;
        }
                
        /* Target Streamlit label elements specifically */
        .st-emotion-cache-16idsys p,
        .st-emotion-cache-16idsys label {
            font-size: 2rem !important;
            color: #1e40af !important;
            font-weight: 700 !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            margin-bottom: 1rem !important;
            display: block !important;
            opacity: 1 !important;
            line-height: 2.5rem !important;
        }
        
        /* Ensure input fields stay the same size */
        .st-emotion-cache-16idsys input,
        .st-emotion-cache-16idsys textarea {
            font-size: 1.2rem !important;
            line-height: 1.5 !important;
        }
        
        /* Specifically target Domain and Specifications labels */
        [data-testid="stFormSubmitButton"] ~ div [data-baseweb="label"] p,
        [data-testid="stTextArea"] ~ div [data-baseweb="label"] p {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #1e40af !important;
        }
                
        /* Update Paper Summary Text Size */
        .stExpander {
            font-size: 1.2rem !important;  /* Increased from base size */
        }
        
        .stTextInput input:focus, 
        .stTextArea textarea:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        /* Section Headers */
        .idea-section-header,
        .paper-section-header,
        div[data-testid="stMarkdownContainer"] h3 {
            color: #1e40af !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            margin: 2rem 0 1rem 0 !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 2px solid #3b82f6 !important;
            opacity: 1 !important;
        }
        
        /* Paper Styles */
        .paper-abstract {
            color: #1e293b !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1.1rem !important;
            line-height: 1.8 !important;
            margin-bottom: 2rem !important;
            opacity: 1 !important;
        }
        
        .paper-references {
            color: #1e293b !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            opacity: 1 !important;
        }
        
        .paper-references li {
            margin-bottom: 0.5rem;
        }
        
        .paper-innovation {
            color: #065f46 !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            opacity: 1 !important;
        }
        
        /* Idea Styles */
        .research-title {
            color: #1e293b !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
            padding: 0.5rem !important;
            background-color: #f8fafc !important;
            border-radius: 6px !important;
            opacity: 1 !important;
        }
        
        .idea-description {
            color: #1e293b !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1.1rem !important;
            line-height: 1.8 !important;
            margin-bottom: 1.5rem !important;
            opacity: 1 !important;
        }
        
        /* Summary Styles */
        .summary-text {
            color: #1e293b !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1.2rem !important;  /* Increased from 1.1rem */
            line-height: 1.6 !important;
            padding: 0.5rem !important;
            opacity: 1 !important;
        }
        
        /* List Styles */
        .drawbacks-list, 
        .opportunities-list {
            margin-top: 0.5rem;
        }
        
        .drawback-item {
            color: #991b1b !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1.1rem !important;
            margin-bottom: 0.5rem !important;
            padding: 0.5rem !important;
            background-color: #fef2f2 !important;
            border-radius: 4px !important;
            opacity: 1 !important;
        }
        
        .opportunity-item {
            color: #065f46 !important;
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 1.1rem !important;
            margin-bottom: 0.5rem !important;
            padding: 0.5rem !important;
            background-color: #f0fdf4 !important;
            border-radius: 4px !important;
            opacity: 1 !important;
        }
        
        /* Separator */
        .idea-separator {
            margin: 2rem 0;
            border: 0;
            border-top: 1px solid #e2e8f0;
        }
        
        /* Fix domain input and add button layout */
        [data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="stButton"] {
            margin-top: 25px !important;
        }
                
        [data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
            gap: 0.5rem !important;
        }
                
        [data-testid="stTextInput"] {
            margin-bottom: 0 !important;
        }
        
        /* Plus Button Alignment and Styling */
        [data-testid="stButton"] button:has(div:contains("➕")) {
            width: 40px !important;
            height: 40px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin-top: 1.5rem !important;  /* Align with input field */
            min-height: unset !important;
            background: linear-gradient(to top, #3b82f6, #60a5fa) !important;
            border-radius: 8px !important;
        }
        
        [data-testid="stButton"] button:has(div:contains("➕")):hover {
            background: linear-gradient(to top, #2563eb, #3b82f6) !important;
            transform: translateY(-2px);
        }
                
        /* Better spacing for the domain inputs */
        [data-testid="column"] {
            padding: 0 0.5rem !important;  /* Add minimal padding */
            display: flex !important;
            align-items: flex-start !important;
        }
                
                /* Input Label Adjustment */
        [data-testid="stTextInput"] label {
            margin-bottom: 0.5rem !important;
            display: block !important;
        }

        /* Input Field Height */
        [data-testid="stTextInput"] input {
            height: 40px !important;  /* Match button height */
            margin-top: 0 !important;
        }
        
        /* Increase font size for paper titles in expanders */
        .stExpander button p {
            font-size: 1.2rem !important;
            color: #1e293b !important;
            font-weight: 500 !important;
        }
        
        /* Force dark mode prevention */
        [data-testid="stAppViewContainer"] {
            color-scheme: light !important;
        }
        
        /* Chat Styles */
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
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    global conn  # Add global connection variable
    
    st.set_page_config(page_title="InspireIt", page_icon="💡", layout="wide")
    apply_custom_styles()
    
    # Initialize Snowflake connection instead of session and root
    conn = init_snowflake()
    
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