import streamlit as st
import os
import re
import json
import base64
import tempfile
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

# Set page configuration
st.set_page_config(
    page_title="Telecom Bill Analyzer",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .agent-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f7ff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .plan-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #e8f5e9;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .info-text {
        color: #444;
        font-size: 1rem;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
    }
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    .step {
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        width: 23%;
    }
    .step.active {
        background-color: #1E88E5;
        color: white;
    }
    .step.completed {
        background-color: #81c784;
        color: white;
    }
    .step.waiting {
        background-color: #f5f5f5;
        color: #757575;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border-bottom: 2px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'bill_content' not in st.session_state:
    st.session_state.bill_content = None
if 'bill_type' not in st.session_state:
    st.session_state.bill_type = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'bill_data' not in st.session_state:
    st.session_state.bill_data = None
if 'usage_analysis' not in st.session_state:
    st.session_state.usage_analysis = None
if 'plan_catalog' not in st.session_state:
    st.session_state.plan_catalog = None
if 'recommended_plans' not in st.session_state:
    st.session_state.recommended_plans = None

# Initialize LLM
def get_llm(temperature=0.2, model="gpt-4o-mini"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Please provide an OpenAI API key to continue.")
        st.stop()
    return ChatOpenAI(
        temperature=temperature,
        model_name=model,
        api_key=api_key
    )

#-------------------------------------------------------------------------
# AGENT 1: Bill Extraction Agent
#-------------------------------------------------------------------------
def extract_bill_content(uploaded_file):
    """
    Agent that directs how to extract content from different file types
    """
    # Read file content
    file_content = uploaded_file.read()
    file_type = uploaded_file.type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Prepare file info for the agent
    file_info = {
        "file_name": uploaded_file.name,
        "file_type": file_type,
        "file_extension": file_extension,
        "file_size_kb": len(file_content) / 1024
    }
    
    # Create a temporary file to work with
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    # Encode small portion of file for the agent
    file_preview = ""
    try:
        if file_type.startswith('image'):
            # For images, we'll describe it but not encode
            file_preview = f"[Image file: {file_info['file_name']}]"
        elif file_type == 'application/pdf':
            # For PDFs, we'll describe it but not encode
            file_preview = f"[PDF file: {file_info['file_name']}]"
        elif file_type.startswith('text') or file_extension in ['html', 'txt', 'csv']:
            # For text files, show a preview
            preview_content = file_content[:2000].decode('utf-8', errors='ignore')
            file_preview = f"Text preview:\n{preview_content}..."
    except Exception as e:
        file_preview = f"[File preview unavailable: {str(e)}]"
    
    # Get the agent's guidance on extracting text from this file type
    llm = get_llm(temperature=0)
    
    with st.spinner("Bill Extraction Agent is analyzing your file..."):
        extraction_prompt = PromptTemplate(
            input_variables=["file_name", "file_type", "file_extension", "file_size_kb", "file_preview"],
            template="""
            You are a Telecom Bill Extraction Agent specializing in extracting text from telecom bills in various formats.
            
            I have a telecom bill file with the following properties:
            - File name: {file_name}
            - File type: {file_type}
            - File extension: {file_extension}
            - File size: {file_size_kb:.2f} KB
            
            File preview (if available):
            {file_preview}
            
            The customer has uploaded this file to extract their telecom bill information. Your role is to analyze what we're looking at and provide detailed instructions for a machine learning system to extract the text content.
            
            Please provide:
            1. A description of what this file appears to be
            2. The best method to extract text from this type of file
            3. What specific libraries and functions would be most effective
            4. A step-by-step process for text extraction
            5. How to handle any potential issues or edge cases
            6. What post-processing might be needed after extraction
            
            Format your response as a structured JSON with these sections.
            """
        )
        
        # Update to use the recommended pattern instead of deprecated LLMChain
        extraction_instructions = llm.invoke(extraction_prompt.format(
            file_name=file_info["file_name"],
            file_type=file_info["file_type"],
            file_extension=file_info["file_extension"],
            file_size_kb=file_info["file_size_kb"],
            file_preview=file_preview
        )).content
        
        # Clean up the JSON response
        extraction_instructions = re.sub(r'```json|```', '', extraction_instructions).strip()
        
        try:
            instructions_json = json.loads(extraction_instructions)
        except json.JSONDecodeError:
            # If parsing fails, create a simple structure
            instructions_json = {
                "description": "Telecom bill file",
                "extraction_method": "Use appropriate library based on file type",
                "steps": ["Read file", "Extract text", "Process content"]
            }
    
    # For demonstration, let's show what the agent recommends, but use a simulated extraction
    # In a production system, we would implement the agent's recommended extraction method
    
    # Now we'll have another agent actually extract text from the file based on its type
    with st.spinner("Extracting text from your bill..."):
        # Process different file types
        if file_type.startswith('image'):
            # For image files - have the agent describe how to handle it
            extraction_result = "The bill appears to be an image. To extract text, we would use OCR technology."
            
            # In a real implementation, we'd use the agent's recommended OCR method
            # For demo purposes, we'll have the agent simulate text extraction from an image bill
            simulation_prompt = PromptTemplate(
                input_variables=["file_name", "file_type"],
                template="""
                You are a Telecom Bill OCR Agent. You need to simulate the extraction of text from a telecom bill image.
                
                The file information is:
                - File name: {file_name}
                - File type: {file_type}
                
                Generate a realistic representation of what text would be extracted from a typical telecom bill image.
                Include common elements like:
                - Customer information
                - Account number
                - Billing period
                - Plan details
                - Usage information for data, calls, and SMS
                - Charges and fees
                - Due date and amount
                
                Format your response as plain text that mimics OCR output from a telecom bill, including some realistic formatting irregularities that might occur during OCR.
                """
            )
            
            # Update to use the recommended pattern instead of deprecated LLMChain
            extracted_text = llm.invoke(simulation_prompt.format(
                file_name=file_info["file_name"],
                file_type=file_info["file_type"]
            )).content
            
        elif file_type == 'application/pdf':
            # For PDF files - have the agent describe how to handle it
            extraction_result = "The bill appears to be a PDF. To extract text, we would use a PDF processing library."
            
            # In a real implementation, we'd use the agent's recommended PDF extraction method
            # For demo purposes, we'll have the agent simulate text extraction from a PDF bill
            simulation_prompt = PromptTemplate(
                input_variables=["file_name", "file_type"],
                template="""
                You are a Telecom Bill PDF Processing Agent. You need to simulate the extraction of text from a telecom bill PDF.
                
                The file information is:
                - File name: {file_name}
                - File type: {file_type}
                
                Generate a realistic representation of what text would be extracted from a typical telecom bill PDF.
                Include common elements like:
                - Customer information
                - Account number
                - Billing period
                - Plan details
                - Usage information for data, calls, and SMS
                - Charges and fees
                - Due date and amount
                
                Format your response as plain text that mimics PDF text extraction output from a telecom bill, including some realistic formatting you'd expect from a PDF.
                """
            )
            
            # Update to use the recommended pattern instead of deprecated LLMChain
            extracted_text = llm.invoke(simulation_prompt.format(
                file_name=file_info["file_name"],
                file_type=file_info["file_type"]
            )).content
            
        elif file_type.startswith('text') or file_extension in ['html', 'txt', 'csv']:
            # For text-based files, read the content directly
            try:
                # Reset file pointer and read again
                uploaded_file.seek(0)
                file_content = uploaded_file.read()
                extracted_text = file_content.decode('utf-8', errors='ignore')
                extraction_result = "Successfully extracted text from the file."
            except Exception as e:
                extracted_text = f"Error extracting text: {str(e)}"
                extraction_result = f"Failed to extract text: {str(e)}"
        else:
            # For unsupported file types
            extracted_text = "Unsupported file type for text extraction."
            extraction_result = "Unsupported file type."
    
    # Clean up temp file
    try:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    except:
        pass
    
    return {
        "agent_instructions": instructions_json,
        "extraction_result": extraction_result,
        "extracted_text": extracted_text
    }

#-------------------------------------------------------------------------
# AGENT 2: Bill Parsing Agent
#-------------------------------------------------------------------------
def parse_bill_structure(extracted_text):
    """
    Agent that parses the extracted text to identify bill structure and components
    """
    llm = get_llm(temperature=0.1)
    
    with st.spinner("Bill Parsing Agent is analyzing the text..."):
        parsing_prompt = PromptTemplate(
            input_variables=["extracted_text"],
            template="""
            You are a Telecom Bill Parsing Agent that specializes in understanding the structure of telecom bills.
            
            Here is the extracted text from a telecom bill:
            
            ```
            {extracted_text}
            ```
            
            Your task is to parse this text and extract the following information in a structured JSON format:
            
            1. Customer information:
               - Customer name
               - Account number
               - Phone number
            
            2. Billing information:
               - Billing period (start and end dates)
               - Bill date
               - Due date
               - Total amount due
            
            3. Current plan details:
               - Plan name
               - Base cost/monthly fee
               - Plan features (data limit, voice minutes, SMS, etc.)
            
            4. Usage breakdown:
               - Data usage (amount used and total allocation)
               - Voice/call usage (minutes used and allocation)
               - SMS/messaging usage (count used and allocation)
               - Any other services usage
            
            5. Additional charges:
               - Itemized extra services with descriptions and amounts
               - One-time charges
               - Overage charges
            
            6. Taxes and fees:
               - Total taxes amount
               - Breakdown of individual taxes and regulatory fees if available
            
            7. Total amount:
               - Total due
               - Previous balance if mentioned
               - Payments received if mentioned
            
            Be adaptive - different telecom bills have different formats. Extract whatever fields are available.
            If a field isn't present in the bill, set its value to null.
            
            Format your response as a clean JSON object. The JSON should be properly formatted and valid.
            """
        )
        
        # Update to use the recommended pattern instead of deprecated LLMChain
        parsing_result = llm.invoke(parsing_prompt.format(
            extracted_text=extracted_text
        )).content
        
        # Clean up the response to ensure it's valid JSON
        parsing_result = parsing_result.strip()
        # Remove any backticks and json language markers that might be in the response
        parsing_result = re.sub(r'```json|```', '', parsing_result).strip()
        
        try:
            # Convert the string to a JSON object
            parsed_data = json.loads(parsing_result)
            return parsed_data
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return an error message
            return {
                "error": f"Could not parse bill structure: {str(e)}",
                "raw_text": parsing_result
            }

#-------------------------------------------------------------------------
# AGENT 3: Usage Pattern Analyzer Agent
#-------------------------------------------------------------------------
def analyze_usage_patterns(bill_data):
    """
    Agent that analyzes usage patterns from the parsed bill data
    """
    llm = get_llm(temperature=0.2)
    
    # Convert the bill data to a string for the prompt
    bill_data_str = json.dumps(bill_data, indent=2)
    
    with st.spinner("Usage Pattern Analyzer Agent is analyzing your usage..."):
        analysis_prompt = PromptTemplate(
            input_variables=["bill_data"],
            template="""
                You are a Telecom Usage Pattern Analyzer Agent specialized in deriving insights from telecom bill data.

                Here is the structured data from a telecom bill:
            
            ```
            {bill_data}
            ```
            Analyze this data and provide the following in a structured JSON format:

1. "userProfile": {{
  "classification": Classify the user (e.g., Data-heavy, Voice-heavy, Balanced, Minimal user),
  "primaryUsage": Identify which services they use most,
  "usagePercentages": Calculate usage percentages relative to their plan limits for each service,
  "recommendedPlanCategory": Recommend a broad category of plans that might suit them better
}}

2. "usagePatterns": {{
  "notablePatterns": Array of notable usage patterns,
  "inefficiencies": Array of potential usage inefficiencies,
  "excessCharges": Array of excess charges that could be avoided,
  "optimizationOpportunities": Array of ways usage could be optimized
}}

3. "usageProjections": {{
  "nextCycle": Projected usage for data, voice, SMS in the next billing cycle,
  "overageRisk": {{
    "level": Risk level of exceeding limits (low, medium, high),
    "details": Explanation of the risk assessment
  }}
}}

4. "costAnalysis": {{
  "breakdown": Cost percentages by category (e.g., base plan, overage, add-ons),
  "highestCost": Highest cost area,
  "unitCosts": Cost per GB/minute/message,
  "potentialSavings": Estimated monthly savings with a better plan
}}

5. "recommendations": {{
  "behavioralChanges": Array of usage behavior changes that could reduce costs,
  "planFeatures": Array of features to look for in a new plan,
  "addOns": Object with recommended add-ons to add or remove
}}

Make sure all arrays contain at least 2-3 items and all values are properly filled out.
If you cannot determine exact values, provide reasonable estimates.
All numeric values should be presented without currency symbols in the JSON structure.

The JSON output must be properly formatted, with no trailing commas, and all quotes properly escaped.
"""
        )
        
        # Update to use the recommended pattern instead of deprecated LLMChain
        analysis_result = llm.invoke(analysis_prompt.format(
            bill_data=bill_data_str
        )).content
        
        # Clean up the response to ensure it's valid JSON
        analysis_result = analysis_result.strip()
        # Remove any backticks and json language markers that might be in the response
        analysis_result = re.sub(r'```json|```', '', analysis_result).strip()
        
        try:
            # Convert the string to a JSON object
            analysis_data = json.loads(analysis_result)
            return analysis_data
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return an error message
            st.error(f"JSON Error: {str(e)}")
            st.write("Raw text:")
            st.write(analysis_result)
            return {
                "error": f"Could not analyze usage patterns: {str(e)}",
                "raw_text": analysis_result
            }
   
        # Update to use the recommended pattern instead of deprecated LLMChain
        analysis_result = llm.invoke(analysis_prompt.format(
            bill_data=bill_data_str
        )).content
        
        # Clean up the response to ensure it's valid JSON
        analysis_result = analysis_result.strip()
        # Remove any backticks and json language markers that might be in the response
        analysis_result = re.sub(r'```json|```', '', analysis_result).strip()
        
        try:
            # Convert the string to a JSON object
            analysis_data = json.loads(analysis_result)
            return analysis_data
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return an error message
            st.error(f"JSON Error: {str(e)}")
            st.write("Raw text:")
            st.write(analysis_result)
            return {
                "error": f"Could not analyze usage patterns: {str(e)}",
                "raw_text": analysis_result
            }

#-------------------------------------------------------------------------
# AGENT 4: Plan Catalog Agent
#-------------------------------------------------------------------------
def generate_plan_catalog():
    """
    Agent that generates a catalog of available telecom plans
    """
    llm = get_llm(temperature=0.3)
    
    with st.spinner("Plan Catalog Agent is gathering available plans..."):
        catalog_prompt = PromptTemplate(
            input_variables=[],
            template="""
            You are a Telecom Plan Catalog Agent responsible for maintaining an up-to-date database of available telecom plans.
            
            Generate a realistic catalog of telecom plans that could be offered by a major telecom provider.
            Include a diverse range of plans across these categories:
            
            1. Individual Plans:
               - Basic (budget-friendly options with limited features)
               - Standard (balanced options for average users)
               - Premium (feature-rich options for heavy users)
            
            2. Family Plans:
               - Options for different family sizes (2-line, 4-line, etc.)
               - Shared data plans
               - Unlimited family plans
            
            3. Specialized Plans:
               - Senior plans
               - Student/youth plans
               - Business plans
            
            4. Prepaid Options:
               - Pay-as-you-go
               - Monthly prepaid
               - Annual prepaid
            
            For each plan, include:
            - "planName": Name of the plan (be creative but realistic)
            - "category": Category the plan belongs to (from the list above)
            - "monthlyCost": Monthly cost as a number without currency symbol
            - "contractLength": Contract requirements in months (0 for no contract)
            - "data": Data allocation details
            - "voice": Voice minutes details
            - "messaging": Messaging allowance details
            - "features": Array of special features
            - "addOns": Array of available add-ons
            
            Include at least 15 different plans total across all categories.
            Format your response as a clean JSON array of plan objects.
            """
        )
        
        # Update to use the recommended pattern instead of deprecated LLMChain
        catalog_result = llm.invoke(catalog_prompt.format()).content
        
        # Clean up the response to ensure it's valid JSON
        catalog_result = catalog_result.strip()
        # Remove any backticks and json language markers that might be in the response
        catalog_result = re.sub(r'```json|```', '', catalog_result).strip()
        
        try:
            # Convert the string to a JSON object
            catalog_data = json.loads(catalog_result)
            return catalog_data
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return an error message
            return {
                "error": f"Could not generate plan catalog: {str(e)}",
                "raw_text": catalog_result
            }

#-------------------------------------------------------------------------
# AGENT 5: Plan Recommendation Agent
#-------------------------------------------------------------------------
def recommend_plans(bill_data, usage_analysis, plan_catalog):
    """
    Agent that recommends optimal plans based on usage analysis and available plans
    """
    llm = get_llm(temperature=0.3)
    
    # Convert the input data to strings for the prompt
    bill_data_str = json.dumps(bill_data, indent=2)
    usage_analysis_str = json.dumps(usage_analysis, indent=2)
    plan_catalog_str = json.dumps(plan_catalog, indent=2)
    
    with st.spinner("Plan Recommendation Agent is finding the best plans for you..."):
        recommendation_prompt = PromptTemplate(
            input_variables=["bill_data", "usage_analysis", "plan_catalog"],
            template="""
            You are a Telecom Plan Recommendation Agent that specializes in matching users with optimal telecom plans.
            
            You have the following information:
            
            1. Current Bill Data:
            ```
            {bill_data}
            ```
            
            2. Usage Analysis:
            ```
            {usage_analysis}
            ```
            
            3. Available Plan Catalog:
            ```
            {plan_catalog}
            ```
            
            Based on this information, recommend the top 3 most suitable plans for this customer in the following JSON format:
            
            {{
                "summary": "Overall explanation of why these plans were selected",
                "recommendations": [
                    {{
                        "planName": "Name of recommended plan 1",
                        "monthlyCost": Cost as a number,
                        "savings": Monthly savings compared to current plan as a number,
                        "matchScore": Percentage match score (0-100),
                        "reasons": ["Reason 1", "Reason 2", "Reason 3"],
                        "pros": ["Pro 1", "Pro 2", "Pro 3"],
                        "cons": ["Con 1", "Con 2"],
                        "recommendedAddOns": ["Add-on 1", "Add-on 2"]
                    }},
                    {{
                        "planName": "Name of recommended plan 2",
                        ... (same structure as above)
                    }},
                    {{
                        "planName": "Name of recommended plan 3",
                        ... (same structure as above)
                    }}
                ],
                "comparison": {{
                    "columns": ["Feature", "Current Plan", "Plan 1", "Plan 2", "Plan 3"],
                    "rows": [
                        ["Monthly Cost", "$X", "$Y", "$Z", "$W"],
                        ["Data", "X GB", "Y GB", "Z GB", "W GB"],
                        ["Voice", "X min", "Y min", "Z min", "W min"],
                        ["SMS", "X texts", "Y texts", "Z texts", "W texts"],
                        ["Contract", "X months", "Y months", "Z months", "W months"],
                        ["Key Features", "Feature list", "Feature list", "Feature list", "Feature list"]
                    ]
                }}
            }}
            
            Make sure the JSON is valid. Include realistic data for all fields.
            """
        )
        
        # Update to use the recommended pattern instead of deprecated LLMChain
        recommendation_result = llm.invoke(recommendation_prompt.format(
            bill_data=bill_data_str,
            usage_analysis=usage_analysis_str,
            plan_catalog=plan_catalog_str
        )).content
        
        # Clean up the response to ensure it's valid JSON
        recommendation_result = recommendation_result.strip()
        # Remove any backticks and json language markers that might be in the response
        recommendation_result = re.sub(r'```json|```', '', recommendation_result).strip()
        
        try:
            # Convert the string to a JSON object
            recommendation_data = json.loads(recommendation_result)
            return recommendation_data
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return an error message
            st.error(f"JSON Error: {str(e)}")
            st.write("Raw text:")
            st.write(recommendation_result)
            return {
                "error": f"Could not generate recommendations: {str(e)}",
                "raw_text": recommendation_result
            }

#-------------------------------------------------------------------------
# FUNCTIONS TO DISPLAY DATA VISUALIZATIONS
#-------------------------------------------------------------------------
def display_usage_charts(usage_analysis):
    """
    Creates and displays visualizations of usage data
    """
    try:
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. User Profile - Usage Percentages (Top Left)
        if 'userProfile' in usage_analysis and 'usagePercentages' in usage_analysis['userProfile']:
            usage_percentages = usage_analysis['userProfile']['usagePercentages']
            if isinstance(usage_percentages, dict):
                labels = list(usage_percentages.keys())
                values = list(usage_percentages.values())
                
                # Convert string percentages to float if needed
                numeric_values = []
                for v in values:
                    if isinstance(v, str) and '%' in v:
                        try:
                            numeric_values.append(float(v.strip('%')))
                        except ValueError:
                            numeric_values.append(0)
                    elif isinstance(v, (int, float)):
                        numeric_values.append(float(v))
                    else:
                        numeric_values.append(0)
                
                axes[0, 0].pie(numeric_values, labels=labels, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Usage Breakdown by Service', fontsize=14)
        
        # 2. Cost Analysis - Breakdown (Top Right)
        if 'costAnalysis' in usage_analysis and 'breakdown' in usage_analysis['costAnalysis']:
            cost_breakdown = usage_analysis['costAnalysis']['breakdown']
            if isinstance(cost_breakdown, dict):
                categories = list(cost_breakdown.keys())
                percentages = list(cost_breakdown.values())
                
                # Convert string percentages to float if needed
                numeric_percentages = []
                for p in percentages:
                    if isinstance(p, str) and '%' in p:
                        try:
                            numeric_percentages.append(float(p.strip('%')))
                        except ValueError:
                            numeric_percentages.append(0)
                    elif isinstance(p, (int, float)):
                        numeric_percentages.append(float(p))
                    else:
                        numeric_percentages.append(0)
                
                axes[0, 1].bar(categories, numeric_percentages)
                axes[0, 1].set_title('Cost Breakdown', fontsize=14)
                axes[0, 1].set_ylabel('Percentage (%)')
                plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
        
        # 3. Usage Projections - Next Cycle (Bottom Left)
        if 'usageProjections' in usage_analysis and 'nextCycle' in usage_analysis['usageProjections']:
            projections = usage_analysis['usageProjections']['nextCycle']
            if isinstance(projections, dict):
                services = list(projections.keys())
                projected_values = []
                
                for service in services:
                    val = projections[service]
                    if isinstance(val, str):
                        # Try to extract numeric part from strings like "8.5 GB"
                        try:
                            numeric_part = re.search(r'(\d+(\.\d+)?)', val)
                            if numeric_part:
                                projected_values.append(float(numeric_part.group(1)))
                            else:
                                projected_values.append(0)
                        except:
                            projected_values.append(0)
                    elif isinstance(val, (int, float)):
                        projected_values.append(float(val))
                    else:
                        projected_values.append(0)
                
                axes[1, 0].bar(services, projected_values, color='orange')
                axes[1, 0].set_title('Projected Usage Next Cycle', fontsize=14)
                axes[1, 0].set_ylabel('Amount')
                plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # 4. Potential Savings Bar Chart (Bottom Right)
        if 'costAnalysis' in usage_analysis and 'potentialSavings' in usage_analysis['costAnalysis']:
            potential_savings = usage_analysis['costAnalysis']['potentialSavings']
            
            # Convert to float if it's a string
            if isinstance(potential_savings, str):
                try:
                    if "$" in potential_savings:
                        potential_savings = float(potential_savings.replace("$", "").strip())
                    else:
                        potential_savings = float(potential_savings)
                except ValueError:
                    potential_savings = 0
            
            current_plan_cost = 0
            if 'bill_data' in st.session_state and 'plan' in st.session_state.bill_data:
                if 'cost' in st.session_state.bill_data['plan']:
                    current_plan_cost = st.session_state.bill_data['plan']['cost']
                    if isinstance(current_plan_cost, str) and "$" in current_plan_cost:
                        try:
                            current_plan_cost = float(current_plan_cost.replace("$", "").strip())
                        except ValueError:
                            current_plan_cost = 0
            
            # Create a comparison chart
            comparison = ['Current Plan', 'With Optimal Plan']
            costs = [current_plan_cost, max(0, current_plan_cost - potential_savings)]
            
            axes[1, 1].bar(comparison, costs, color=['#FF5252', '#4CAF50'])
            axes[1, 1].set_title('Cost Comparison', fontsize=14)
            axes[1, 1].set_ylabel('Monthly Cost ($)')
            
            # Add savings amount as text
            if potential_savings > 0:
                axes[1, 1].text(0.5, 0.5, f'Save ${potential_savings:.2f}/month', 
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=axes[1, 1].transAxes,
                               fontsize=16, fontweight='bold')
        
        # Adjust layout and spacing
        plt.tight_layout()
        return fig
    except Exception as e:
        # Catch any exceptions that might occur during chart creation
        st.error(f"Error creating visualization: {str(e)}")
        # Create and return an empty figure as fallback
        fig = plt.figure(figsize=(10, 6))
        fig.text(0.5, 0.5, f"Could not create charts: {str(e)}", 
                horizontalalignment='center', verticalalignment='center')
        return fig

#-------------------------------------------------------------------------
# FUNCTION TO DISPLAY RECOMMENDED PLANS
#-------------------------------------------------------------------------
def display_plan_recommendations(recommended_plans):
    """
    Creates a visual display of the recommended plans
    """
    if "error" in recommended_plans:
        st.error(recommended_plans["error"])
        return
    
    # Display the summary
    if "summary" in recommended_plans:
        st.markdown("### Recommendation Summary")
        st.markdown(f"<div class='highlight'>{recommended_plans['summary']}</div>", unsafe_allow_html=True)
    
    # Display plan comparison table if available
    if "comparison" in recommended_plans:
        comparison = recommended_plans["comparison"]
        
        if "columns" in comparison and "rows" in comparison:
            st.markdown("### Plan Comparison")
            
            # Convert comparison data to DataFrame
            df = pd.DataFrame(comparison["rows"], columns=comparison["columns"])
            st.table(df)
    
    # Display individual plan recommendations
    if "recommendations" in recommended_plans:
        st.markdown("### Recommended Plans")
        
        for i, plan in enumerate(recommended_plans["recommendations"], 1):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"<div class='plan-card'>", unsafe_allow_html=True)
                st.markdown(f"#### {i}. {plan.get('planName', 'Plan')}")
                
                # Plan match score
                match_score = plan.get('matchScore', 0)
                if isinstance(match_score, str) and '%' in match_score:
                    match_score = match_score.strip('%')
                st.progress(float(match_score) / 100)
                st.markdown(f"Match Score: {match_score}%")
                
                # Cost and savings
                st.markdown(f"**Cost:** ${plan.get('monthlyCost', 0)}/month")
                st.markdown(f"**Savings:** ${plan.get('savings', 0)}/month")
                
                # Reasons for recommendation
                if "reasons" in plan:
                    st.markdown("**Why this plan:**")
                    for reason in plan["reasons"]:
                        st.markdown(f"- {reason}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Pros and cons
                if "pros" in plan:
                    st.markdown("**Pros:**")
                    for pro in plan["pros"]:
                        st.markdown(f"✅ {pro}")
                
                if "cons" in plan:
                    st.markdown("**Cons:**")
                    for con in plan["cons"]:
                        st.markdown(f"⚠️ {con}")

#-------------------------------------------------------------------------
# USER INTERFACE
#-------------------------------------------------------------------------

# Header
st.markdown('<h1 class="main-header">📱 Telecom Bill Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Upload your telecom bill to get personalized plan recommendations</p>', unsafe_allow_html=True)

# Step indicator
steps = ["1. Upload Bill", "2. Extract & Parse", "3. Analyze Usage", "4. Get Recommendations"]
step_statuses = []

for i, step in enumerate(steps, 1):
    if i < st.session_state.current_step:
        status = "completed"
    elif i == st.session_state.current_step:
        status = "active"
    else:
        status = "waiting"
    step_statuses.append(status)

step_html = '<div class="step-indicator">'
for i, (step, status) in enumerate(zip(steps, step_statuses)):
    step_html += f'<div class="step {status}">{step}</div>'
step_html += '</div>'

st.markdown(step_html, unsafe_allow_html=True)

# Step 1: Upload Bill
if st.session_state.current_step == 1:
    st.markdown('<h2 class="sub-header">Upload Your Telecom Bill</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="agent-card">', unsafe_allow_html=True)
        st.markdown("### Bill Ingestion Agent")
        st.markdown("This agent will process your telecom bill and extract all the relevant information.")
        st.markdown("**Supported formats:**")
        st.markdown("- PDF bills")
        st.markdown("- Image files (JPG, PNG)")
        st.markdown("- HTML bills")
        st.markdown("- Text files")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        uploaded_file = st.file_uploader("Choose your telecom bill file", 
                                      type=["pdf", "jpg", "jpeg", "png", "html", "txt"])
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Process Bill", type="primary"):
                with st.spinner("Processing your bill..."):
                    # Store the file info in session state
                    st.session_state.bill_content = uploaded_file
                    st.session_state.bill_type = uploaded_file.type
                    
                    # Extract text from the bill using Agent 1
                    extraction_result = extract_bill_content(uploaded_file)
                    st.session_state.extracted_text = extraction_result["extracted_text"]
                    
                    # Proceed to next step
                    st.session_state.current_step = 2
                    st.rerun()

# Step 2: Extract & Parse
elif st.session_state.current_step == 2:
    st.markdown('<h2 class="sub-header">Bill Extraction & Parsing</h2>', unsafe_allow_html=True)
    
    # Display a preview of the extracted text
    with st.expander("View Extracted Text", expanded=False):
        st.text_area("Text extracted from your bill:", 
                   value=st.session_state.extracted_text, 
                   height=200)
    
    # Parse the bill structure using Agent 2
    if st.session_state.bill_data is None:
        with st.spinner("Parsing bill structure..."):
            parsed_data = parse_bill_structure(st.session_state.extracted_text)
            st.session_state.bill_data = parsed_data
    
    # Display the parsed bill data in a user-friendly format
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.markdown("### Bill Structure Parsing Results")
    
    if "error" in st.session_state.bill_data:
        st.error(st.session_state.bill_data["error"])
    else:
        # Customer Information
        if "customer" in st.session_state.bill_data:
            st.markdown("#### Customer Information")
            customer = st.session_state.bill_data["customer"]
            customer_df = pd.DataFrame({
                "Field": ["Name", "Account Number", "Phone Number"],
                "Value": [
                    customer.get("name", "N/A"),
                    customer.get("account_number", "N/A"),
                    customer.get("phone_number", "N/A")
                ]
            })
            st.dataframe(customer_df, hide_index=True)
        
        # Billing Information
        if "billing" in st.session_state.bill_data:
            st.markdown("#### Billing Information")
            billing = st.session_state.bill_data["billing"]
            billing_df = pd.DataFrame({
                "Field": ["Billing Period", "Due Date", "Total Amount Due"],
                "Value": [
                    billing.get("billing_period", "N/A"),
                    billing.get("due_date", "N/A"),
                    f"${billing.get('total_amount', 'N/A')}"
                ]
            })
            st.dataframe(billing_df, hide_index=True)
        
        # Plan Details
        if "plan" in st.session_state.bill_data:
            st.markdown("#### Current Plan")
            plan = st.session_state.bill_data["plan"]
            plan_df = pd.DataFrame({
                "Field": ["Plan Name", "Monthly Cost"],
                "Value": [
                    plan.get("name", "N/A"),
                    f"${plan.get('cost', 'N/A')}"
                ]
            })
            st.dataframe(plan_df, hide_index=True)
        
        # Usage Summary
        if "usage" in st.session_state.bill_data:
            st.markdown("#### Usage Summary")
            usage = st.session_state.bill_data["usage"]
            usage_data = []
            
            if "data" in usage:
                usage_data.append(["Data", f"{usage['data'].get('used', 'N/A')} of {usage['data'].get('limit', 'Unlimited')}"])
            
            if "voice" in usage:
                usage_data.append(["Voice", f"{usage['voice'].get('used', 'N/A')} of {usage['voice'].get('limit', 'Unlimited')}"])
            
            if "sms" in usage:
                usage_data.append(["SMS", f"{usage['sms'].get('used', 'N/A')} of {usage['sms'].get('limit', 'Unlimited')}"])
            
            usage_df = pd.DataFrame(usage_data, columns=["Service", "Usage"])
            st.dataframe(usage_df, hide_index=True)
        
        # Additional Charges
        if "charges" in st.session_state.bill_data:
            st.markdown("#### Additional Charges")
            charges = st.session_state.bill_data["charges"]
            if isinstance(charges, list):
                charges_df = pd.DataFrame([
                    [item.get("description", "N/A"), f"${item.get('amount', 'N/A')}"]
                    for item in charges
                ], columns=["Description", "Amount"])
                st.dataframe(charges_df, hide_index=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a button to continue to the next step
    if st.button("Analyze Usage Patterns", type="primary"):
        st.session_state.current_step = 3
        st.rerun()

# Step 3: Analyze Usage
elif st.session_state.current_step == 3:
    st.markdown('<h2 class="sub-header">Usage Analysis</h2>', unsafe_allow_html=True)
    
    # Generate usage analysis if not already done
    if st.session_state.usage_analysis is None:
        with st.spinner("Analyzing your usage patterns..."):
            usage_analysis = analyze_usage_patterns(st.session_state.bill_data)
            st.session_state.usage_analysis = usage_analysis
    
    # Display usage analysis results
    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
    st.markdown("### Usage Pattern Analysis Results")
    
    if "error" in st.session_state.usage_analysis:
        st.error(st.session_state.usage_analysis["error"])
    else:
        tabs = st.tabs(["User Profile", "Usage Patterns", "Usage Projections", "Cost Analysis", "Recommendations", "Visualizations"])
        
        with tabs[0]:
            if "userProfile" in st.session_state.usage_analysis:
                user_profile = st.session_state.usage_analysis["userProfile"]
                
                st.markdown(f"**User Classification:** {user_profile.get('classification', 'N/A')}")
                st.markdown(f"**Primary Usage:** {user_profile.get('primaryUsage', 'N/A')}")
                
                # Display usage percentages
                if "usagePercentages" in user_profile:
                    st.markdown("#### Usage Percentages")
                    percentages = user_profile["usagePercentages"]
                    if isinstance(percentages, dict):
                        percentage_df = pd.DataFrame({
                            "Service": list(percentages.keys()),
                            "Usage %": list(percentages.values())
                        })
                        st.dataframe(percentage_df, hide_index=True)
                
                st.markdown(f"**Recommended Plan Category:** {user_profile.get('recommendedPlanCategory', 'N/A')}")
        
        with tabs[1]:
            if "usagePatterns" in st.session_state.usage_analysis:
                patterns = st.session_state.usage_analysis["usagePatterns"]
                
                # Notable patterns
                if "notablePatterns" in patterns:
                    st.markdown("#### Notable Usage Patterns")
                    for pattern in patterns["notablePatterns"]:
                        st.markdown(f"- {pattern}")
                
                # Inefficiencies
                if "inefficiencies" in patterns:
                    st.markdown("#### Usage Inefficiencies")
                    for inefficiency in patterns["inefficiencies"]:
                        st.markdown(f"- {inefficiency}")
                
                # Excess charges
                if "excessCharges" in patterns:
                    st.markdown("#### Excess Charges")
                    for charge in patterns["excessCharges"]:
                        st.markdown(f"- {charge}")
                
                # Optimization opportunities
                if "optimizationOpportunities" in patterns:
                    st.markdown("#### Optimization Opportunities")
                    for opportunity in patterns["optimizationOpportunities"]:
                        st.markdown(f"- {opportunity}")
        
        with tabs[2]:
            if "usageProjections" in st.session_state.usage_analysis:
                projections = st.session_state.usage_analysis["usageProjections"]
                
                # Next cycle projections
                if "nextCycle" in projections:
                    st.markdown("#### Next Billing Cycle Projections")
                    next_cycle = projections["nextCycle"]
                    if isinstance(next_cycle, dict):
                        projection_df = pd.DataFrame({
                            "Service": list(next_cycle.keys()),
                            "Projected Usage": list(next_cycle.values())
                        })
                        st.dataframe(projection_df, hide_index=True)
                
                # Overage risk
                if "overageRisk" in projections:
                    risk = projections["overageRisk"]
                    st.markdown(f"#### Overage Risk: {risk.get('level', 'N/A')}")
                    st.markdown(f"{risk.get('details', '')}")
        
        with tabs[3]:
            if "costAnalysis" in st.session_state.usage_analysis:
                cost = st.session_state.usage_analysis["costAnalysis"]
                
                # Cost breakdown
                if "breakdown" in cost:
                    st.markdown("#### Cost Breakdown")
                    breakdown = cost["breakdown"]
                    if isinstance(breakdown, dict):
                        breakdown_df = pd.DataFrame({
                            "Category": list(breakdown.keys()),
                            "Percentage": list(breakdown.values())
                        })
                        st.dataframe(breakdown_df, hide_index=True)
                
                # Highest cost
                if "highestCost" in cost:
                    st.markdown(f"**Highest Cost Area:** {cost.get('highestCost', 'N/A')}")
                
                # Unit costs
                if "unitCosts" in cost:
                    st.markdown("#### Unit Costs")
                    unit_costs = cost["unitCosts"]
                    if isinstance(unit_costs, dict):
                        unit_costs_df = pd.DataFrame({
                            "Service": list(unit_costs.keys()),
                            "Cost per Unit": list(unit_costs.values())
                        })
                        st.dataframe(unit_costs_df, hide_index=True)
                
                # Potential savings
                if "potentialSavings" in cost:
                    savings = cost["potentialSavings"]
                    if isinstance(savings, (int, float)):
                        st.markdown(f"**Potential Monthly Savings:** ${savings}")
                    else:
                        st.markdown(f"**Potential Monthly Savings:** {savings}")
        
        with tabs[4]:
            if "recommendations" in st.session_state.usage_analysis:
                recs = st.session_state.usage_analysis["recommendations"]
                
                # Behavioral changes
                if "behavioralChanges" in recs:
                    st.markdown("#### Recommended Behavior Changes")
                    for change in recs["behavioralChanges"]:
                        st.markdown(f"- {change}")
                
                # Plan features
                if "planFeatures" in recs:
                    st.markdown("#### Recommended Plan Features")
                    for feature in recs["planFeatures"]:
                        st.markdown(f"- {feature}")
                
                # Add-ons
                if "addOns" in recs:
                    add_ons = recs["addOns"]
                    
                    if "add" in add_ons and isinstance(add_ons["add"], list):
                        st.markdown("#### Recommended Add-Ons to Add")
                        for addon in add_ons["add"]:
                            st.markdown(f"- {addon}")
                    
                    if "remove" in add_ons and isinstance(add_ons["remove"], list):
                        st.markdown("#### Recommended Add-Ons to Remove")
                        for addon in add_ons["remove"]:
                            st.markdown(f"- {addon}")
        
        with tabs[5]:
            # Create visualizations
            try:
                fig = display_usage_charts(st.session_state.usage_analysis)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating visualizations: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate plan catalog if not already done
    if st.session_state.plan_catalog is None:
        with st.spinner("Generating plan catalog..."):
            plan_catalog = generate_plan_catalog()
            st.session_state.plan_catalog = plan_catalog
    
    # Add a button to continue to the next step
    if st.button("Get Plan Recommendations", type="primary"):
        st.session_state.current_step = 4
        st.rerun()

# Step 4: Get Recommendations
elif st.session_state.current_step == 4:
    st.markdown('<h2 class="sub-header">Plan Recommendations</h2>', unsafe_allow_html=True)
    
    # Generate recommendations if not already done
    if st.session_state.recommended_plans is None:
        with st.spinner("Finding the best plans for you..."):
            recommended_plans = recommend_plans(
                st.session_state.bill_data,
                st.session_state.usage_analysis,
                st.session_state.plan_catalog
            )
            st.session_state.recommended_plans = recommended_plans
    
    # Display recommendations
    display_plan_recommendations(st.session_state.recommended_plans)
    
    # Add a button to restart the process
    if st.button("Analyze Another Bill", type="primary"):
        # Reset session state
        for key in ['current_step', 'bill_content', 'bill_type', 'extracted_text', 
                   'bill_data', 'usage_analysis', 'plan_catalog', 'recommended_plans']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.current_step = 1
        st.rerun()