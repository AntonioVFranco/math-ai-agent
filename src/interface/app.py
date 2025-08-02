"""
Math AI Agent - Gradio Interface

A minimal viable interface using Gradio for the math-ai-agent project.
This UI allows users to securely provide their OpenAI API key and submit
mathematical problems for solving.

Author: Math AI Agent Team
Task ID: UI-001
User Story: US-003
"""

import gradio as gr
import re
import os
from typing import Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
APP_TITLE = "Math AI Agent"
APP_DESCRIPTION = """
# 🧮 Math AI Agent

Specialized AI agent for advanced mathematical problem solving.
Supports symbolic computation, equation solving, calculus, and more.

**Features:**
- Symbolic mathematics with SymPy
- LaTeX rendering support
- Step-by-step solutions
- Verification of results
"""

# Custom CSS for responsive design and styling
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* Responsive design for mobile devices */
@media (max-width: 768px) {
    .gradio-container {
        max-width: 100% !important;
        padding: 10px !important;
    }
    
    /* Make textboxes more mobile-friendly */
    .gr-textbox textarea {
        font-size: 16px !important; /* Prevents zoom on iOS */
    }
    
    /* Adjust button sizing for mobile */
    .gr-button {
        min-height: 44px !important; /* Touch-friendly size */
        font-size: 16px !important;
    }
}

/* Custom styling for the problem input */
.problem-input {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
}

/* Custom styling for the output */
.output-display {
    border: 1px solid #e1e5e9;
    border-radius: 8px;
    padding: 16px;
    background-color: #f8f9fa;
    min-height: 200px;
}

/* Accordion styling */
.accordion-header {
    background-color: #f1f3f4 !important;
    border-radius: 6px !important;
}

/* Success and error message styling */
.success-message {
    color: #28a745 !important;
    font-weight: 500;
}

.error-message {
    color: #dc3545 !important;
    font-weight: 500;
}

/* LaTeX rendering improvements */
.katex {
    font-size: 1.1em !important;
}
"""


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate the OpenAI API key format.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key or not api_key.strip():
        return False, "API key is required. Please enter your OpenAI API key."
    
    api_key = api_key.strip()
    
    # Basic format validation for OpenAI API keys
    if not api_key.startswith('sk-'):
        return False, "Invalid API key format. OpenAI API keys should start with 'sk-'."
    
    if len(api_key) < 20:
        return False, "API key appears to be too short. Please check your key."
    
    # Check for valid characters (alphanumeric and specific symbols)
    if not re.match(r'^sk-[A-Za-z0-9\-_]+$', api_key):
        return False, "API key contains invalid characters."
    
    return True, "API key format is valid."


def validate_problem_input(problem_text: str) -> Tuple[bool, str]:
    """
    Validate the mathematical problem input.
    
    Args:
        problem_text: The problem text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not problem_text or not problem_text.strip():
        return False, "Please enter a mathematical problem to solve."
    
    problem_text = problem_text.strip()
    
    if len(problem_text) < 3:
        return False, "Problem description seems too short. Please provide more details."
    
    if len(problem_text) > 5000:
        return False, "Problem description is too long. Please limit to 5000 characters."
    
    return True, "Problem input is valid."


def process_mathematical_problem(api_key: str, problem_text: str) -> Tuple[str, Any]:
    """
    Process mathematical problems using the complete Math AI Agent pipeline.
    
    This function integrates the problem parser, OpenAI reasoning, SymPy calculations,
    and final synthesis to deliver comprehensive mathematical solutions.
    
    Args:
        api_key: OpenAI API key
        problem_text: Mathematical problem to solve
        
    Returns:
        Tuple of (formatted response string, plot object or None)
    """
    logger.info(f"Processing problem: {problem_text[:100]}...")
    
    # Validate API key
    api_valid, api_message = validate_api_key(api_key)
    if not api_valid:
        error_msg = f"""
## ❌ Error: Invalid API Key

{api_message}

**Please check your API key and try again.**

---
*Note: Your API key should start with 'sk-' and be obtained from [OpenAI's platform](https://platform.openai.com/api-keys).*
"""
        return error_msg, None
    
    # Validate problem input
    problem_valid, problem_message = validate_problem_input(problem_text)
    if not problem_valid:
        error_msg = f"""
## ❌ Error: Invalid Problem Input

{problem_message}

**Please provide a valid mathematical problem and try again.**
"""
        return error_msg, None
    
    try:
        # Import the engine here to avoid circular imports
        from ..core.engine import execute_solution_pipeline
        
        # Execute the complete solution pipeline
        logger.info("Executing Math AI Agent solution pipeline...")
        result = execute_solution_pipeline(problem_text, api_key)
        
        if result.success:
            # Add execution metadata to the response
            metadata_parts = []
            
            # Add processing time and statistics
            if result.total_execution_time_ms:
                execution_time_sec = result.total_execution_time_ms / 1000
                metadata_parts.append(f"⏱️ **Processing Time:** {execution_time_sec:.2f} seconds")
            
            if result.openai_calls > 0:
                metadata_parts.append(f"🤖 **OpenAI API Calls:** {result.openai_calls}")
            
            if result.sympy_calls > 0:
                metadata_parts.append(f"🧮 **SymPy Calculations:** {result.sympy_calls}")
            
            # Add parsing confidence if available
            if result.parsed_problem and result.parsed_problem.confidence.overall > 0:
                confidence_pct = result.parsed_problem.confidence.overall * 100
                metadata_parts.append(f"🎯 **Parsing Confidence:** {confidence_pct:.1f}%")
            
            # Combine the final answer with metadata
            final_response_parts = [result.final_answer]
            
            if metadata_parts:
                final_response_parts.extend([
                    "",
                    "---",
                    "### 📊 Processing Information",
                    ""
                ])
                final_response_parts.extend(metadata_parts)
                final_response_parts.extend([
                    "",
                    "*Math AI Agent v2.0 - Complete integration of parsing, reasoning, and symbolic computation*"
                ])
            
            final_text = "\n".join(final_response_parts)
            return final_text, result.plot_object
        
        else:
            # Handle pipeline failure
            error_details = []
            
            if result.parsed_problem:
                error_details.append(f"**Problem Domain:** {result.parsed_problem.domain.value}")
                error_details.append(f"**Problem Type:** {result.parsed_problem.problem_type.value}")
            
            if result.total_execution_time_ms:
                execution_time_sec = result.total_execution_time_ms / 1000
                error_details.append(f"**Processing Time:** {execution_time_sec:.2f} seconds")
            
            error_response = f"""
## ❌ Solution Pipeline Failed

{result.error_message}

### 🔍 Diagnostic Information
{chr(10).join(error_details) if error_details else '*No additional diagnostic information available.*'}

### 💡 Troubleshooting Tips
- **Check your API key**: Ensure it's valid and has sufficient credits
- **Verify the problem**: Make sure your mathematical problem is clearly stated
- **Try a simpler problem**: Start with basic problems to test the system
- **Check connectivity**: Ensure you have a stable internet connection

**If the problem persists, please try again or contact support.**

---
*Error logged for debugging purposes. Problem parsing and analysis may have been successful even if the complete solution failed.*
"""
            return error_response, None
    
    except ImportError as e:
        logger.error(f"Failed to import engine module: {str(e)}")
        error_msg = f"""
## ❌ System Configuration Error

The Math AI Agent engine is not properly configured. This may be due to missing dependencies or incorrect installation.

**Error Details:**
```
{str(e)}
```

**Please check:**
- All required dependencies are installed
- The engine module is properly configured
- The system has been set up correctly

---
*Contact support if this error persists.*
"""
        return error_msg, None
    
    except Exception as e:
        logger.error(f"Unexpected error in problem processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"""
## ❌ Unexpected Processing Error

An unexpected error occurred while processing your mathematical problem.

**Error Details:**
```
{str(e)}
```

**What to try:**
1. Check that your API key is valid and active
2. Ensure your problem is clearly stated
3. Try refreshing the page and submitting again
4. Try a simpler mathematical problem to test the system

**If this error continues to occur, please report it with:**
- The exact problem you submitted
- Your browser and system information
- The time when the error occurred

---
*Error logged for debugging purposes.*
"""
        return error_msg, None


def create_interface() -> gr.Blocks:
    """
    Create and configure the Gradio interface.
    
    Returns:
        Configured Gradio Blocks interface
    """
    
    with gr.Blocks(
        title=APP_TITLE,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate"
        )
    ) as interface:
        
        # Header
        gr.Markdown(APP_DESCRIPTION)
        
        # Settings accordion (API key input)
        with gr.Accordion("Settings / Configurações", open=False) as settings_accordion:
            gr.Markdown("### 🔑 API Configuration")
            gr.Markdown("Enter your OpenAI API key to enable mathematical problem solving.")
            
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-...",
                type="password",
                info="Your API key will be encrypted and used only for this session.",
                container=True,
                scale=1
            )
            
            gr.Markdown("""
            **How to get your API key:**
            1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Sign in to your account
            3. Create a new API key
            4. Copy and paste it above
            
            *Your API key is never stored and is only used for the current session.*
            """)
        
        # Main interface
        gr.Markdown("### 📝 Mathematical Problem Input")
        
        with gr.Row():
            with gr.Column(scale=1):
                problem_input = gr.Textbox(
                    label="Insira seu problema matemático (suporta LaTeX)",
                    placeholder="""Exemplo:
Resolva a equação: x² - 5x + 6 = 0

Ou em LaTeX:
$\\int_{0}^{\\pi} \\sin(x) dx$

Ou descreva o problema:
Encontre o valor máximo da função f(x) = -x² + 4x + 1""",
                    lines=15,
                    info="Você pode usar notação matemática, LaTeX, ou descrever o problema em linguagem natural.",
                    container=True,
                    elem_classes=["problem-input"]
                )
        
        # Submit button
        with gr.Row():
            submit_button = gr.Button(
                "Resolver",
                variant="primary",
                size="lg",
                scale=1
            )
        
        # Output display with tabs
        gr.Markdown("### 📊 Results / Resultados")
        
        with gr.Tabs():
            with gr.Tab("📄 Solution / Solução"):
                output_display = gr.Markdown(
                    value="*Insira um problema matemático e clique em 'Resolver' para ver os resultados aqui.*\n\n*Enter a mathematical problem and click 'Resolver' to see the results here.*",
                    elem_classes=["output-display"],
                    container=True
                )
            
            with gr.Tab("📊 Visualização"):
                plot_display = gr.Plot(
                    visible=False,
                    container=True,
                    elem_classes=["plot-display"]
                )
                plot_info = gr.Markdown(
                    value="*Visualizações aparecerão aqui quando relevantes para o problema.*\n\n*Visualizations will appear here when relevant to the problem.*",
                    visible=True
                )
        
        # Event handlers
        def handle_submit(api_key: str, problem_text: str) -> Tuple[str, Any, str]:
            """Handle the submit button click."""
            try:
                text_result, plot_result = process_mathematical_problem(api_key, problem_text)
                
                # Handle plot visibility and content
                if plot_result is not None:
                    # Show plot and update info
                    plot_update = gr.update(visible=True, value=plot_result)
                    plot_info_update = "*Visualização gerada para o problema atual.*\n\n*Visualization generated for the current problem.*"
                else:
                    # Hide plot and show info message
                    plot_update = gr.update(visible=False)
                    plot_info_update = "*Nenhuma visualização disponível para este problema.*\n\n*No visualization available for this problem.*"
                
                return text_result, plot_update, plot_info_update
                
            except Exception as e:
                logger.error(f"Submit handler error: {str(e)}")
                error_msg = f"""
## ❌ System Error

An unexpected system error occurred:

```
{str(e)}
```

Please try again or refresh the page.
"""
                return error_msg, gr.update(visible=False), "*Erro no sistema.*\n\n*System error.*"
        
        # Connect the submit button to the handler
        submit_button.click(
            fn=handle_submit,
            inputs=[api_key_input, problem_input],
            outputs=[output_display, plot_display, plot_info],
            api_name="solve_problem"
        )
        
        # Also allow Enter key submission (when focused on problem input)
        problem_input.submit(
            fn=handle_submit,
            inputs=[api_key_input, problem_input],
            outputs=[output_display, plot_display, plot_info]
        )
        
        # Footer
        gr.Markdown("""
        ---
        
        ### 📚 About Math AI Agent
        
        This is a specialized AI agent designed to excel at mathematical problem solving.
        It combines symbolic computation, advanced reasoning, and step-by-step explanations
        to provide accurate solutions to complex mathematical problems.
        
        **Features in development:**
        - SymPy integration for symbolic mathematics
        - OpenAI GPT integration for natural language understanding
        - LaTeX rendering for beautiful mathematical expressions
        - Step-by-step solution explanations
        - Result verification and validation
        
        **Status:** Minimal Viable Interface (UI-001) ✅
        """)
    
    return interface


def launch_app(
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    debug: bool = False
) -> None:
    """
    Launch the Gradio application.
    
    Args:
        share: Whether to create a public shareable link
        server_name: Server hostname
        server_port: Server port
        debug: Whether to enable debug mode
    """
    logger.info("Starting Math AI Agent Gradio Interface...")
    
    # Create the interface
    interface = create_interface()
    
    # Launch the interface
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        debug=debug,
        show_error=True,
        quiet=False,
        favicon_path=None,  # Could add a custom favicon later
        ssl_verify=False,
        app_kwargs={
            "docs_url": "/docs",
            "redoc_url": "/redoc"
        }
    )


def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Math AI Agent Gradio Interface")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link")
    parser.add_argument("--host", default="0.0.0.0", help="Server hostname")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    launch_app(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()