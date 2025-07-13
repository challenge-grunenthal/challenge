import os
import sys

import streamlit as st

# Add the parent directory to the path to import the agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import reset_agent, run_agent_with_streaming

# Page configuration
st.set_page_config(
    page_title="Pharmaceutical AI Assistant", page_icon="💊", layout="wide"
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .step-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        padding: 0.5rem;
    }
    .step-header {
        font-weight: bold;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .step-content {
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 0.25rem;
        padding: 0.75rem;
        font-family: monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
    }
    .config-section {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .config-title {
        font-weight: bold;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for chat history and configuration
if "messages" not in st.session_state:
    st.session_state.messages = []
if "steps_history" not in st.session_state:
    st.session_state.steps_history = {}
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "neo4j_uri" not in st.session_state:
    st.session_state.neo4j_uri = ""
if "neo4j_username" not in st.session_state:
    st.session_state.neo4j_username = ""
if "neo4j_password" not in st.session_state:
    st.session_state.neo4j_password = ""

# Header
st.markdown(
    '<h1 class="main-header">💊 Pharmaceutical AI Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Ask me about FDA adverse events, pharmaceutical knowledge graphs, and company reports</p>',
    unsafe_allow_html=True,
)

# Sidebar with configuration and information
with st.sidebar:
    st.header("⚙️ Configuration")

    # Configuration section
    with st.container():
        st.markdown(
            '<div class="config-title">🔑 OpenAI API Key</div>', unsafe_allow_html=True
        )
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key",
        )
        if openai_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_key

        st.markdown(
            '<div class="config-title">🗄️ Neo4j Database</div>', unsafe_allow_html=True
        )
        neo4j_uri = st.text_input(
            "Database URI",
            value=st.session_state.neo4j_uri,
            placeholder="bolt://localhost:7687",
            help="Enter your Neo4j database URI",
        )
        if neo4j_uri != st.session_state.neo4j_uri:
            st.session_state.neo4j_uri = neo4j_uri

        neo4j_username = st.text_input(
            "Username",
            value=st.session_state.neo4j_username,
            placeholder="neo4j",
            help="Enter your Neo4j username",
        )
        if neo4j_username != st.session_state.neo4j_username:
            st.session_state.neo4j_username = neo4j_username

        neo4j_password = st.text_input(
            "Password",
            value=st.session_state.neo4j_password,
            type="password",
            placeholder="password",
            help="Enter your Neo4j password",
        )
        if neo4j_password != st.session_state.neo4j_password:
            st.session_state.neo4j_password = neo4j_password
        st.markdown("</div>", unsafe_allow_html=True)

    # Check if all configuration is complete
    config_complete = all(
        [
            st.session_state.openai_api_key.strip(),
            st.session_state.neo4j_uri.strip(),
            st.session_state.neo4j_username.strip(),
            st.session_state.neo4j_password.strip(),
        ]
    )

    if not config_complete:
        st.markdown(
            """
        <div class="warning-box">
            ⚠️ <strong>Configuration Required</strong><br>
            Please fill in all configuration fields above before using the assistant.
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.header("🤖 Available Tools")
    st.markdown(
        """
    This AI assistant can help you with:
    
    **🔍 FDA Adverse Events**
    - Search for drug safety data
    - Get adverse event reports
    
    **🧠 Knowledge Graph**
    - Query pharmaceutical relationships
    - Find drug-manufacturer connections
    
    **📄 Document Search**
    - Search through company reports
    - Get detailed information from PDFs
    """
    )
    
    st.markdown("---")
    st.header("💡 Top 9 Development Examples")
    st.markdown(
        """
    **🔍 FDA Examples:**
    1. "What adverse events are reported for TRAMADOL?"
    2. "Show me safety data for OXYCODONE including serious adverse events"
    3. "Compare adverse events between ASPIRIN and IBUPROFEN"
    
    **🧠 Knowledge Graph Examples:**
    4. "Which manufacturers are connected to drugs containing REVLIMID?"
    5. "Find all drugs manufactured by PFIZER in the knowledge graph"
    
    **📄 Document Examples:**
    6. "What information can you find about Grünenthal's revenue in 2023?"
    7. "Summarize Grünenthal's research and development activities from the annual report"
    8. "What are the key strategic initiatives mentioned in the company report?"

    **🔍 Combined Examples:**
    9. "Compare the safety profile of TRAMADOL with its market performance and manufacturer information"
    """
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.steps_history = {}
        st.rerun()

    # Reset agent button
    if st.button("🔄 Reset Agent Configuration"):
        reset_agent()
        st.success("Agent configuration reset successfully!")
        st.rerun()

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show steps for assistant messages if available
        if message["role"] == "assistant" and i in st.session_state.steps_history:
            steps = st.session_state.steps_history[i]
            if steps and len(steps) > 1:
                st.markdown("---")
                st.markdown("**🔍 Processing Steps:**")

                for j, step in enumerate(steps):
                    if step["step_type"] == "final_answer":
                        continue  # Skip final answer as it's already displayed

                    step_type_emoji = {
                        "tool_decision": "🤔",
                        "tool_execution": "🔧",
                        "model_call": "🧠",
                    }.get(step["step_type"], "📝")

                    # Create collapsible section for each step
                    with st.expander(
                        f"{step_type_emoji} Step {j + 1}: {step['task_name'].replace('_', ' ').title()}",
                        expanded=False,
                    ):
                        st.markdown(
                            f"**Type:** {step['step_type'].replace('_', ' ').title()}"
                        )
                        st.markdown("**Content:**")
                        st.code(step["content"], language="text")

# Chat input - only show if configuration is complete
if config_complete:
    if prompt := st.chat_input("Ask me anything about pharmaceuticals..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            steps_placeholder = st.empty()

            try:
                # Initialize variables
                final_answer = ""
                steps = []
                step_count = 0

                # Stream the agent execution with configuration
                for step_data in run_agent_with_streaming(
                    prompt,
                    openai_api_key=st.session_state.openai_api_key,
                    neo4j_uri=st.session_state.neo4j_uri,
                    neo4j_username=st.session_state.neo4j_username,
                    neo4j_password=st.session_state.neo4j_password,
                ):
                    steps.append(step_data)

                    if step_data["is_final"]:
                        # Display final answer
                        final_answer = step_data["content"]
                        message_placeholder.markdown(final_answer)
                    else:
                        # Display only the current step in real-time using Streamlit expander
                        steps_placeholder.markdown("**🔍 Processing Steps:**")
                        if step_data["step_type"] != "final_answer":
                            step_type_emoji = {
                                "tool_decision": "🤔",
                                "tool_execution": "🔧",
                                "model_call": "🧠",
                            }.get(step_data["step_type"], "📝")
                            with st.expander(
                                f"{step_type_emoji} Step: {step_data['step_type'].replace('_', ' ').title()}",
                                expanded=True,
                            ):
                                st.markdown(
                                    f"**Type:** {step_data['step_type'].replace('_', ' ').title()}"
                                )
                                st.markdown("**Content:**")
                                st.code(step_data["content"], language="text")

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )

                # Store steps in session state for later viewing
                message_id = len(st.session_state.messages) - 1
                st.session_state.steps_history[message_id] = steps

            except Exception as e:
                error_message = f"❌ Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
else:
    # Show disabled chat input when configuration is incomplete
    st.markdown(
        """
    <div style='text-align: center; padding: 2rem; color: #666;'>
        🔒 Chat is disabled until all configuration fields are filled in the sidebar.
    </div>
    """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Powered by LangChain, OpenAI, and Streamlit | 
        <a href='https://github.com/your-repo' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
