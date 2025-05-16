import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import os
import requests
import re
from collections import Counter, defaultdict
from pathlib import Path

from collections import defaultdict
issue_summary = defaultdict(lambda: {'count': 0, 'examples': []})
MAX_EXAMPLES = 5

# Function to fetch JSON from website
def load_transcript(json_str):
    """Parse JSON transcript data into a pandas DataFrame"""
    try:
        # Parsing the JSON string
        transcript_data = json.loads(json_str)
        
        # Converting to Pandas DataFrame
        df = pd.DataFrame(transcript_data)
        
        # Adding message length
        df['message_length'] = df['content'].apply(len)
        
        # Adding word count
        df['word_count'] = df['content'].apply(lambda x: len(x.split()))
        
        # Adding sequence number
        df['sequence'] = range(1, len(df) + 1)
        
        return df
    except Exception as e:
        print(f"Error processing transcript data: {e}")
        return None
    
# Functions to analyze call transcript
def analyze_script_adherence(df):
    """Analyze how closely the agent followed the script based on the call script provided"""
    # Defining script steps and their keywords based on the detailed script provided
    script_steps = {
        'step1_warm_welcome': ['Hi', 'hello', 'Amelia', 'Solar', 'Is that', 'speaking with'],
        'step3_purpose_availability': ['federal government', 'solar rebate program', 'rebate', 'homeowners', 'install solar panels', 'qualify', 'minutes'],
        'step4_qualify_customer': ['check if you qualify', 'own', 'address', 'solar panels', 'installed', 'roof', 'free-standing house', 'unit', 'apartment'],
        'step5_free_assessment': ['free home assessment', 'solar expert', 'visit your place', 'no-obligation quote', 'analyze your bill', 'usage patterns', 'inspect your roof'],
        'step6_closing_remarks': ['thank you for your time', 'keep you posted', 'future rebate', 'have a nice day', 'goodbye'],
        'step7_thanks_close': ['thank you', 'have a nice day', 'goodbye'],
        'communication_check': ['Are you still there', 'Hello', 'checking in'],
        'objection_handling': ['apologize', 'sorry', 'understand', 'concern', 'misunderstanding', 'budget', 'not selling', 'no obligation']
    }
    
    # Objection types from the script
    objection_types = {
        'wrong_information': ['wrong name', 'wrong information', 'incorrect details'],
        'call_source': ['where are you calling from', 'call center', 'Green Energy Solar'],
        'data_source': ['how did you get my', 'contact details', 'information'],
        'call_frequency': ['how many times', 'keep getting calls', 'fed up'],
        'not_interested': ['not interested', 'don\'t want this'],
        'rebate_reason': ['why is the government giving', 'reason for this rebate'],
        'government_call': ['calling from the government', 'government call'],
        'affordability': ['cannot afford', 'don\'t have the budget', '0% interest'],
        'rebate_amount': ['how much rebate', 'rebate amount', '30% of the total'],
        'total_cost': ['how much will it cost', 'total cost', '6.6 kW system'],
        'feed_in_tariff': ['feed-in tariff', 'cents per kWh'],
        'company_name': ['name of your company', 'which company', 'Green Energy Solar'],
        'privacy_concern': ['not give my first name', 'scam', 'need my address', 'confidential information'],
        'renting': ['renting', 'don\'t own the house'],
        'ai_question': ['are you an AI', 'robot'],
        'availability': ['not sure if I\'ll be available', 'confirmation call', 'reschedule']
    }
    
    # Check agent messages for script elements
    adherence = {}
    agent_msgs = df[df['role'] == 'agent']
    user_msgs = df[df['role'] == 'user']
    
    for step, keywords in script_steps.items():
        matches = []
        for _, row in agent_msgs.iterrows():
            content = row['content'].lower()
            if any(keyword.lower() in content for keyword in keywords):
                matches.append({
                    'sequence': row['sequence'],
                    'content': row['content']
                })
        
        adherence[step] = matches
    
    # Check for objection handling
    objection_handling = {}
    for obj_type, keywords in objection_types.items():
        matches = []
        # Check for objections in user messages
        for _, user_row in user_msgs.iterrows():
            user_content = user_row['content'].lower()
            if any(keyword.lower() in user_content for keyword in keywords):
                # Look for agent response in next messages
                next_agent_msgs = agent_msgs[agent_msgs['sequence'] > user_row['sequence']].head(2)
                if not next_agent_msgs.empty:
                    for _, agent_row in next_agent_msgs.iterrows():
                        matches.append({
                            'objection_sequence': user_row['sequence'],
                            'objection_content': user_row['content'],
                            'response_sequence': agent_row['sequence'],
                            'response_content': agent_row['content']
                        })
                        break
        
        objection_handling[obj_type] = matches
    
    adherence['objection_handling_detailed'] = objection_handling
    
    # Check for interruptions
    interruptions = df[df['event'] == 'agent_stopped_speaking']
    if not interruptions.empty:
        adherence['interruptions'] = interruptions[['sequence', 'content']].to_dict('records')
    else:
        adherence['interruptions'] = []
    
    # Check for communication issues
    communication_issues = []
    for _, row in user_msgs.iterrows():
        content = row['content'].lower()
        if content == "hello?" or "hello" in content:
            # Find preceding agent message
            preceding_msgs = agent_msgs[agent_msgs['sequence'] < row['sequence']].tail(1)
            if not preceding_msgs.empty:
                communication_issues.append({
                    'user_sequence': row['sequence'],
                    'user_content': row['content'],
                    'agent_sequence': preceding_msgs.iloc[0]['sequence'],
                    'agent_content': preceding_msgs.iloc[0]['content']
                })
    
    adherence['communication_issues'] = communication_issues
    
    return adherence

def analyze_script_flow(df, adherence):
    """Analyze if the script flow followed the correct sequence"""
    # Check if script steps happened in the correct order
    expected_order = [
        'step1_warm_welcome',
        'step3_purpose_availability', 
        'step4_qualify_customer',
        'step5_free_assessment',
        'step6_closing_remarks',
        'step7_thanks_close'
    ]
    
    actual_order = []
    for step in expected_order:
        if step in adherence and adherence[step]:
            min_seq = min([m['sequence'] for m in adherence[step]])
            actual_order.append((step, min_seq))
    
    # Sort by sequence number
    actual_order.sort(key=lambda x: x[1])
    
    # Check if any steps were skipped
    skipped_steps = [step for step in expected_order if step not in [a[0] for a in actual_order]]
    
    # Check if steps happened out of order
    actual_step_order = [a[0] for a in actual_order]
    expected_indices = {step: i for i, step in enumerate(expected_order)}
    
    out_of_order = False
    for i in range(len(actual_step_order) - 1):
        if expected_indices[actual_step_order[i]] > expected_indices[actual_step_order[i+1]]:
            out_of_order = True
            break
    
    # Check if proper customer qualification flow was followed
    qualification_flow_correct = False
    
    # Check if Step 4 was followed by either Step 5 (if qualified) or Step 6 (if not qualified)
    if 'step4_qualify_customer' in [a[0] for a in actual_order]:
        step4_idx = [i for i, a in enumerate(actual_order) if a[0] == 'step4_qualify_customer'][0]
        
        if step4_idx < len(actual_order) - 1:
            next_step = actual_order[step4_idx + 1][0]
            if next_step in ['step5_free_assessment', 'step6_closing_remarks']:
                qualification_flow_correct = True
    
    return {
        'actual_order': actual_order,
        'skipped_steps': skipped_steps,
        'out_of_order': out_of_order,
        'qualification_flow_correct': qualification_flow_correct
    }
    
def identify_script_deviations(df, adherence):
    """Identify deviations from the script based on call content"""
    deviations = []
    
    # Check if proper name verification was done
    if 'step1_warm_welcome' in adherence and adherence['step1_warm_welcome']:
        name_verification = False
        for item in adherence['step1_warm_welcome']:
            if "is that" in item['content'].lower() or "speaking with" in item['content'].lower():
                name_verification = True
                break
        
        if not name_verification:
            deviations.append({
                'type': 'missing_name_verification',
                'description': 'Agent did not properly verify customer name as per script'
            })
    
    # Check repeated "Are you still there?" messages
    still_there_count = 0
    for _, row in df[df['role'] == 'agent'].iterrows():
        if "are you still there" in row['content'].lower():
            still_there_count += 1
    
    if still_there_count > 2:
        deviations.append({
            'type': 'excessive_checking',
            'description': f'Agent asked "Are you still there?" {still_there_count} times, indicating communication issues'
        })
    
    # Check if address verification was done as required in Step 4
    address_verification = False
    for _, row in df[df['role'] == 'agent'].iterrows():
        content = row['content'].lower()
        if any(term in content for term in ["address", "street", "suburb", "state", "pincode"]):
            address_verification = True
            break
    
    if not address_verification and 'step4_qualify_customer' in adherence and adherence['step4_qualify_customer']:
        deviations.append({
            'type': 'incomplete_qualification',
            'description': 'Agent did not verify address as required in Step 4'
        })
    
    # Check if roof verification was done as required in Step 4
    roof_check = False
    solar_panel_check = False
    freestanding_check = False
    
    for _, row in df[df['role'] == 'agent'].iterrows():
        content = row['content'].lower()
        if "roof" in content:
            roof_check = True
        if "solar panels" in content:
            solar_panel_check = True
        if any(term in content for term in ["free-standing", "unit", "apartment"]):
            freestanding_check = True
    
    if 'step4_qualify_customer' in adherence and adherence['step4_qualify_customer']:
        if not roof_check:
            deviations.append({
                'type': 'incomplete_qualification',
                'description': 'Agent did not verify if customer has a suitable roof for solar panels'
            })
        if not solar_panel_check:
            deviations.append({
                'type': 'incomplete_qualification',
                'description': 'Agent did not ask if customer already has solar panels installed'
            })
        if not freestanding_check:
            deviations.append({
                'type': 'incomplete_qualification',
                'description': 'Agent did not verify if property is a free-standing house'
            })
    
    # Check if free home assessment was offered to qualified customers (Step 5)
    if 'step4_qualify_customer' in adherence and adherence['step4_qualify_customer'] and 'step5_free_assessment' not in adherence:
        deviations.append({
            'type': 'missing_free_assessment',
            'description': 'Agent qualified the customer but did not offer the free home assessment'
        })
    
    # Check for proper closing remarks (Step 6 or 7)
    if ('step6_closing_remarks' not in adherence or not adherence['step6_closing_remarks']) and \
       ('step7_thanks_close' not in adherence or not adherence['step7_thanks_close']):
        deviations.append({
            'type': 'improper_call_closure',
            'description': 'Call ended without proper closing remarks as per script Steps 6 or 7'
        })
    
    # Check for objection handling
    objection_found = False
    objection_handled = False
    
    for obj_type, matches in adherence.get('objection_handling_detailed', {}).items():
        if matches:
            objection_found = True
            # Check if agent response contains appropriate handling language
            for match in matches:
                response = match['response_content'].lower()
                if any(keyword in response for keyword in ['understand', 'apologize', 'sorry', 'no obligation', 'not selling']):
                    objection_handled = True
                    break
    
    if objection_found and not objection_handled:
        deviations.append({
            'type': 'poor_objection_handling',
            'description': 'Agent did not handle customer objections according to script guidelines'
        })
    
    return deviations

def analyze_objection_handling(df, adherence):
    """Analyze how well objections were handled according to script guidelines"""
    objection_analysis = {
        'objections_identified': 0,
        'properly_handled': 0,
        'poorly_handled': 0,
        'details': []
    }
    
    # Examine each objection type
    for obj_type, matches in adherence.get('objection_handling_detailed', {}).items():
        for match in matches:
            objection_analysis['objections_identified'] += 1
            
            # Check if response follows script guidelines
            response = match['response_content'].lower()
            objection = match['objection_content'].lower()
            
            # Define expected responses for common objections
            expected_responses = {
                'wrong_information': ['apologize', 'technical glitch', 'correct number'],
                'call_source': ['melbourne', 'green energy solar'],
                'not_interested': ['misunderstanding', 'not selling', 'no-obligation', 'assessment'],
                'affordability': ['30%', 'government', 'rebate', '0% interest', 'finance'],
                'privacy_concern': ['understand', 'concern', 'confidential', 'inform']
            }
            
            # Check if response contains expected elements
            handled_properly = False
            
            # Check for generic good response patterns
            if any(term in response for term in ['understand', 'apologize', 'sorry']):
                handled_properly = True
            
            # Check for specific response patterns based on objection type
            for obj_key, expected_terms in expected_responses.items():
                if obj_key in obj_type and any(term in response for term in expected_terms):
                    handled_properly = True
                    break
            
            if handled_properly:
                objection_analysis['properly_handled'] += 1
            else:
                objection_analysis['poorly_handled'] += 1
            
            objection_analysis['details'].append({
                'objection_type': obj_type,
                'objection': objection,
                'response': response,
                'handled_properly': handled_properly,
                'sequence': match['objection_sequence']
            })
    
    return objection_analysis

def check_qualification_process(df, adherence):
    """Check if the qualification process was followed correctly"""
    qualification_analysis = {
        'address_verification': False,
        'solar_panels_check': False,
        'property_type_check': False,
        'under_75_age_check': False,
        'bill_amount_check': False,
        'sunlight_check': False,
        'second_contact_check': False,
        'complete': False,
        'details': []
    }
    
    # Check for address verification
    for _, row in df[df['role'] == 'agent'].iterrows():
        content = row['content'].lower()
        
        # Check for address verification
        if any(term in content for term in ["address", "street", "suburb", "state", "pincode"]):
            qualification_analysis['address_verification'] = True
            qualification_analysis['details'].append({
                'check_type': 'address_verification',
                'sequence': row['sequence'],
                'content': row['content']
            })
        
        # Check for solar panels question
        if "solar panels" in content and ("roof" in content or "installed" in content):
            qualification_analysis['solar_panels_check'] = True
            qualification_analysis['details'].append({
                'check_type': 'solar_panels_check',
                'sequence': row['sequence'],
                'content': row['content']
            })
        
        # Check for property type question
        if "free-standing" in content or "unit" in content or "apartment" in content:
            qualification_analysis['property_type_check'] = True
            qualification_analysis['details'].append({
                'check_type': 'property_type_check',
                'sequence': row['sequence'],
                'content': row['content']
            })
        
        # Check for age verification
        if "75" in content or "age" in content:
            qualification_analysis['under_75_age_check'] = True
            qualification_analysis['details'].append({
                'check_type': 'under_75_age_check',
                'sequence': row['sequence'],
                'content': row['content']
            })
        
        # Check for bill amount
        if "bill" in content and ("quarter" in content or "$" in content):
            qualification_analysis['bill_amount_check'] = True
            qualification_analysis['details'].append({
                'check_type': 'bill_amount_check',
                'sequence': row['sequence'],
                'content': row['content']
            })
        
        # Check for sunlight question
        if "sunlight" in content and "roof" in content:
            qualification_analysis['sunlight_check'] = True
            qualification_analysis['details'].append({
                'check_type': 'sunlight_check',
                'sequence': row['sequence'],
                'content': row['content']
            })
        
        # Check for second contact
        if "mobile" in content or "second point of contact" in content:
            qualification_analysis['second_contact_check'] = True
            qualification_analysis['details'].append({
                'check_type': 'second_contact_check',
                'sequence': row['sequence'],
                'content': row['content']
            })
    
    # Determine if qualification process was complete
    basic_checks = [
        qualification_analysis['address_verification'],
        qualification_analysis['solar_panels_check'],
        qualification_analysis['property_type_check']
    ]
    
    additional_checks = [
        qualification_analysis['under_75_age_check'],
        qualification_analysis['bill_amount_check'],
        qualification_analysis['sunlight_check'],
        qualification_analysis['second_contact_check']
    ]
    
    # Basic qualification requires all basic checks
    basic_qualification = all(basic_checks)
    
    # Full qualification requires basic qualification and at least some of the additional checks
    # (Not all may be necessary depending on the call flow)
    full_qualification = basic_qualification and sum(additional_checks) >= 2
    
    qualification_analysis['complete'] = full_qualification
    
    return qualification_analysis

def analyze_communication_problems(df):
    """Analyze communication problems in the call"""
    problems = []
    
    # Check for repeated user messages indicating they can't hear
    hello_count = 0
    for _, row in df[df['role'] == 'user'].iterrows():
        if "hello" in row['content'].lower():
            hello_count += 1
    
    if hello_count > 2:
        problems.append({
            'type': 'customer_hearing_issues',
            'description': f'Customer said "Hello" {hello_count} times, indicating they could not hear the agent clearly'
        })
    
    # Check for repeated agent checking messages
    checking_count = 0
    for _, row in df[df['role'] == 'agent'].iterrows():
        if "are you still there" in row['content'].lower() or "checking in" in row['content'].lower():
            checking_count += 1
    
    if checking_count > 2:
        problems.append({
            'type': 'agent_connection_issues',
            'description': f'Agent checked if customer was there {checking_count} times, indicating connection problems'
        })
    
    # Check if agent was interrupted
    interruptions = df[df['event'] == 'agent_stopped_speaking']
    if not interruptions.empty:
        problems.append({
            'type': 'agent_interrupted',
            'description': f'Agent was interrupted {len(interruptions)} time(s), indicating potential call quality issues'
        })
    
    # Check for abrupt call ending
    last_message = df.iloc[-1]
    if last_message['role'] == 'agent' and "goodbye" in last_message['content'].lower():
        # Check if there was proper closure before goodbye
        proper_closure = False
        for i in range(len(df) - 2, max(0, len(df) - 5), -1):
            if df.iloc[i]['role'] == 'agent' and any(keyword in df.iloc[i]['content'].lower() for keyword in ['thank you', 'thanks for your time']):
                proper_closure = True
                break
        
        if not proper_closure:
            problems.append({
                'type': 'abrupt_call_ending',
                'description': 'Call ended abruptly without proper closure as per script'
            })
    
    return problems

def visualize_analysis(df, adherence, flow_analysis, deviations, problems):
    """Create visualizations for the call analysis"""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Message distribution grid
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    role_counts = df['role'].value_counts()
    ax1.bar(role_counts.index, role_counts.values, color=['#3498db', '#e74c3c'])
    ax1.set_title('Message Count by Role')
    ax1.set_ylabel('Number of Messages')
    
    # 2. Message length by sequence
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    for i, row in df.iterrows():
        color = '#3498db' if row['role'] == 'agent' else '#e74c3c'
        ax2.scatter(row['sequence'], row['word_count'], color=color, s=100)
    
    # Add regression line
    sns.regplot(x='sequence', y='word_count', data=df, ax=ax2, scatter=False, color='gray')
    
    ax2.set_xlabel('Message Sequence')
    ax2.set_ylabel('Word Count')
    ax2.set_title('Message Length Throughout Conversation')
    ax2.legend(['Agent', 'Customer'], loc='upper right')
    
    # 3. Script adherence visualization
    ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    steps = list(adherence.keys())
    step_counts = [len(adherence[step]) for step in steps]
    
    # Only plot actual script steps (not issues)
    script_steps = [step for step in steps if step not in ['interruptions', 'communication_issues']]
    script_counts = [len(adherence[step]) for step in script_steps]
    
    ax3.barh(script_steps, script_counts, color='#9b59b6')
    ax3.set_title('Script Steps Adherence')
    ax3.set_xlabel('Number of Occurrences')
    
    # 4. Communication issues timeline
    ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    
    # Plot user responses
    user_seq = df[df['role'] == 'user']['sequence'].tolist()
    user_y = [0.3] * len(user_seq)
    ax4.scatter(user_seq, user_y, color='#e74c3c', s=80, marker='o', label='User')
    
    # Plot agent messages
    agent_seq = df[df['role'] == 'agent']['sequence'].tolist()
    agent_y = [0.7] * len(agent_seq)
    ax4.scatter(agent_seq, agent_y, color='#3498db', s=80, marker='o', label='Agent')
    
    # Highlight communication issues
    if 'communication_issues' in adherence:
        for issue in adherence['communication_issues']:
            ax4.axvspan(issue['agent_sequence'], issue['user_sequence'], alpha=0.2, color='yellow')
    
    # Highlight interruptions
    if 'interruptions' in adherence and adherence['interruptions']:
        for interruption in adherence['interruptions']:
            ax4.axvline(x=interruption['sequence'], color='red', linestyle='--', alpha=0.7)
    
    ax4.set_title('Communication Issues Timeline')
    ax4.set_xlabel('Message Sequence')
    ax4.set_yticks([0.3, 0.7])
    ax4.set_yticklabels(['User', 'Agent'])
    ax4.set_ylim(0, 1)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('call_analysis.png')
    print("Visualization saved as 'call_analysis.png'")
    plt.close()
    
def generate_report(df, adherence, flow_analysis, deviations, problems, objection_analysis, qualification_analysis):
    """Generate a comprehensive analysis report with detailed script adherence metrics"""
    # Basic statistics
    total_messages = len(df)
    agent_messages = len(df[df['role'] == 'agent'])
    user_messages = len(df[df['role'] == 'user'])
    
    # Calculate average message lengths
    avg_agent_words = df[df['role'] == 'agent']['word_count'].mean()
    avg_user_words = df[df['role'] == 'user']['word_count'].mean()
    
    # Script adherence summary
    script_steps = {
        'step1_warm_welcome': 'Step 1 - Warm Welcome',
        'step3_purpose_availability': 'Step 3 - Purpose and Availability',
        'step4_qualify_customer': 'Step 4 - Customer Qualification',
        'step5_free_assessment': 'Step 5 - Free Home Assessment',
        'step6_closing_remarks': 'Step 6 - Closing Remarks',
        'step7_thanks_close': 'Step 7 - Thanks and Close'
    }
    
    adherence_summary = {script_steps[step]: len(matches) for step, matches in adherence.items() 
                        if step in script_steps}
    
    # Generate the report
    report = f"""
# Solar Call Analysis Report

## Call Summary
- Total messages: {total_messages}
- Agent messages: {agent_messages}
- Customer messages: {user_messages}
- Average agent message length: {avg_agent_words:.1f} words
- Average customer message length: {avg_user_words:.1f} words
- Call date: {df['time'].iloc[0].date() if 'time' in df.columns else 'N/A'}

## Script Adherence Summary
"""
    
    # Add script steps summary
    for step_name, count in adherence_summary.items():
        report += f"- {step_name}: {'✓' if count > 0 else '✗'} ({count} instances)\n"
    
    # Add script flow analysis
    report += """
## Script Flow Analysis
"""
    
    if flow_analysis['skipped_steps']:
        skipped = ', '.join([script_steps.get(step, step) for step in flow_analysis['skipped_steps']])
        report += f"- **Skipped Steps**: {skipped}\n"
    else:
        report += "- No steps were skipped\n"
    
    if flow_analysis['out_of_order']:
        report += "- **Issue**: Script steps were executed out of order\n"
    else:
        report += "- Steps were executed in the correct order\n"
    
    if flow_analysis.get('qualification_flow_correct', False):
        report += "- Customer qualification flow was followed correctly\n"
    else:
        report += "- **Issue**: Customer qualification flow was not followed correctly\n"
    
    # Add qualification process analysis
    report += """
## Qualification Process Analysis
"""
    
    report += f"- Address verification: {'✓' if qualification_analysis['address_verification'] else '✗'}\n"
    report += f"- Solar panels check: {'✓' if qualification_analysis['solar_panels_check'] else '✗'}\n"
    report += f"- Property type check: {'✓' if qualification_analysis['property_type_check'] else '✗'}\n"
    
    if 'step5_free_assessment' in adherence and adherence['step5_free_assessment']:
        report += f"- Age verification (under 75): {'✓' if qualification_analysis['under_75_age_check'] else '✗'}\n"
        report += f"- Bill amount check: {'✓' if qualification_analysis['bill_amount_check'] else '✗'}\n"
        report += f"- Sunlight on roof check: {'✓' if qualification_analysis['sunlight_check'] else '✗'}\n"
        report += f"- Second contact point: {'✓' if qualification_analysis['second_contact_check'] else '✗'}\n"
    
    report += f"- **Overall qualification process**: {'Complete' if qualification_analysis['complete'] else 'Incomplete'}\n"
    
    report += """
## Objection Handling Analysis
"""
    
    if objection_analysis['objections_identified'] > 0:
        report += f"- Total objections identified: {objection_analysis['objections_identified']}\n"
        report += f"- Properly handled: {objection_analysis['properly_handled']} ({100 * objection_analysis['properly_handled'] / objection_analysis['objections_identified']:.1f}%)\n"
        report += f"- Poorly handled: {objection_analysis['poorly_handled']}\n"
        
        if objection_analysis['details']:
            report += "\n### Objection Details\n"
            for detail in objection_analysis['details']:
                report += f"- **{detail['objection_type'].replace('_', ' ').title()}**: {'✓' if detail['handled_properly'] else '✗'}\n"
                report += f"  - Customer: \"{detail['objection']}\"\n"
                report += f"  - Agent: \"{detail['response']}\"\n"
    else:
        report += "- No customer objections identified in this call\n"
    
    report += """
## Script Deviations
"""
    
    if deviations:
        for deviation in deviations:
            report += f"- **{deviation['type'].replace('_', ' ').title()}**: {deviation['description']}\n"
    else:
        report += "- No specific script deviations identified\n"
    
    report += """
## Communication Problems
"""
    
    if problems:
        for problem in problems:
            report += f"- **{problem['type'].replace('_', ' ').title()}**: {problem['description']}\n"
    else:
        report += "- No specific communication problems identified\n"
    
    if 'interruptions' in adherence and adherence['interruptions']:
        report += f"\n- **Agent Interruptions**: {len(adherence['interruptions'])} instances where the agent was interrupted\n"
    
    if 'communication_issues' in adherence and adherence['communication_issues']:
        report += f"\n- **Communication Issues**: {len(adherence['communication_issues'])} instances where the customer appeared to have trouble hearing the agent\n"
    
    report += """
## Recommendations
"""
    
    recommendations = []
    
    if flow_analysis['skipped_steps']:
        recommendations.append("**Follow Complete Script**: Agent should follow all steps in the script without skipping any sections")
    
    if not qualification_analysis['complete']:
        recommendations.append("**Complete Qualification Process**: Ensure all required qualification questions are asked")
    
    if objection_analysis['objections_identified'] > 0 and objection_analysis['poorly_handled'] > 0:
        recommendations.append("**Improve Objection Handling**: Review script responses for common objections")
    
    if 'communication_issues' in adherence and adherence['communication_issues']:
        recommendations.append("**Address Connection Issues**: When customer repeatedly says \"Hello?\", recognize and address connection problems")
    
    if not ('step6_closing_remarks' in adherence and adherence['step6_closing_remarks']) and not ('step7_thanks_close' in adherence and adherence['step7_thanks_close']):
        recommendations.append("**Proper Call Closure**: Always end calls with proper closing remarks")
    
    if len(recommendations) == 0:
        recommendations = [
            "**Maintain Performance**: Continue following the script closely",
            "**Regular Training**: Conduct refresher training on script requirements"
        ]
    
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"
    
    report += """
## Script Comparison Summary
The script requires agents to:
- Start with a warm welcome and verify customer name (Step 1)
- Explain the purpose and check availability (Step 3)
- Qualify the customer with specific qualifying questions (Step 4)
- Offer free home assessment for qualified customers (Step 5)
- Handle objections appropriately
- Close the call properly (Step 6/7)

This agent:
"""
    
    performance_summary = []
    
    if 'step1_warm_welcome' in adherence and adherence['step1_warm_welcome']:
        if any("is that" in item['content'].lower() for item in adherence['step1_warm_welcome']):
            performance_summary.append("✓ Properly greeted and attempted to verify the customer name")
        else:
            performance_summary.append("⚠️ Greeted the customer but didn't properly verify the name")
    else:
        performance_summary.append("✗ Failed to properly greet the customer")
    
    if 'step3_purpose_availability' in adherence and adherence['step3_purpose_availability']:
        performance_summary.append("✓ Explained the purpose of the call (solar rebate program)")
    else:
        performance_summary.append("✗ Failed to properly explain the call purpose")
    
    qualification_status = "✓ " if qualification_analysis['complete'] else "⚠️ " if sum([
        qualification_analysis['address_verification'],
        qualification_analysis['solar_panels_check'],
        qualification_analysis['property_type_check']
    ]) > 0 else "✗ "
    
    qualification_text = "Completely qualified the customer with all required questions" if qualification_analysis['complete'] else \
                         "Partially qualified the customer but missed some required questions" if sum([
                             qualification_analysis['address_verification'],
                             qualification_analysis['solar_panels_check'],
                             qualification_analysis['property_type_check']
                         ]) > 0 else "Failed to properly qualify the customer"
    
    performance_summary.append(f"{qualification_status}{qualification_text}")
    
    if 'step5_free_assessment' in adherence and adherence['step5_free_assessment']:
        performance_summary.append("✓ Offered the free home assessment to the customer")
    elif 'step4_qualify_customer' in adherence and adherence['step4_qualify_customer']:
        performance_summary.append("✗ Qualified the customer but didn't offer the free home assessment")
    
    if objection_analysis['objections_identified'] > 0:
        if objection_analysis['properly_handled'] == objection_analysis['objections_identified']:
            performance_summary.append("✓ Properly handled all customer objections")
        elif objection_analysis['properly_handled'] > 0:
            performance_summary.append("⚠️ Handled some but not all customer objections appropriately")
        else:
            performance_summary.append("✗ Failed to properly handle customer objections")
    
    if ('step6_closing_remarks' in adherence and adherence['step6_closing_remarks']) or \
       ('step7_thanks_close' in adherence and adherence['step7_thanks_close']):
        performance_summary.append("✓ Properly closed the call according to script guidelines")
    else:
        performance_summary.append("✗ Did not properly close the call according to script guidelines")
    
    for point in performance_summary:
        report += f"- {point}\n"
    
    return report

def analyze_call_transcript(json_str):
    """Main function to analyze the call transcript"""
    print("Analyzing call transcript...")
    
    df = load_transcript(json_str)
    
    if df is None:
        print("Failed to process transcript data")
        return
    
    if 'time' in df.columns:
        try:
            df['time'] = pd.to_datetime(df['time'])
        except:
            pass 
    
    adherence = analyze_script_adherence(df)
    flow_analysis = analyze_script_flow(df, adherence)
    deviations = identify_script_deviations(df, adherence)
    problems = analyze_communication_problems(df)
    
    objection_analysis = analyze_objection_handling(df, adherence)
    qualification_analysis = check_qualification_process(df, adherence)
    
    try:
        print("Creating visualizations...")
        visualize_analysis(df, adherence, flow_analysis, deviations, problems)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    print("Generating analysis report...")
    report = generate_report(
        df, 
        adherence, 
        flow_analysis, 
        deviations, 
        problems, 
        objection_analysis, 
        qualification_analysis
    )
    
    print("Analysis report generated")
    
    print("\nAnalysis Complete! Key findings:")
    print(f"- Found {len(df)} total messages ({len(df[df['role'] == 'agent'])} from agent, {len(df[df['role'] == 'user'])} from customer)")
    print(f"- Identified {len(deviations)} script deviations and {len(problems)} communication problems")
    print(f"- Customer objections: {objection_analysis['objections_identified']} (properly handled: {objection_analysis['properly_handled']})")
    print(f"- Qualification process: {'Complete' if qualification_analysis['complete'] else 'Incomplete'}")
    
    return report

def fetch_transcript_from_url(url):
    """
    Fetches transcript JSON from a URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching transcript from {url}: {e}")
        return None

class SolarCallAnalyzer:
    def __init__(self, report_content=None):
        """Initialize the analyzer with the directory containing report files."""
        self.report_content = report_content
        self.objections_data = []
        self.all_objections = []
        self.objection_categories = {
            "data_privacy": ["database", "information", "private", "privacy", "data", "consent", "permission", "share"],
            "not_interested": ["not interested", "no interest", "don't want", "don't need"],
            "already_have_solar": ["already have", "existing", "installed"],
            "cost_concerns": ["expensive", "cost", "afford", "price", "money"],
            "property_unsuitable": ["rent", "renting", "apartment", "condo", "shade", "unsuitable", "not suitable"],
            "timing_issues": ["bad time", "busy", "call back", "later", "not now"],
            "trust_issues": ["scam", "legitimate", "trust", "proof"],
            "need_to_consult": ["spouse", "husband", "wife", "partner", "think about", "consider"],
            "age_related": ["too old", "elderly", "senior", "retirement"],
            "moving_soon": ["moving", "sell", "selling", "relocate"]
        }
        self.report_summaries = []
        self.failed_files = []

    def categorize_objection(self, objection_text):
        """Categorize an objection based on keywords."""
        objection_text = objection_text.lower()
        
        for category, keywords in self.objection_categories.items():
            for keyword in keywords:
                if keyword.lower() in objection_text:
                    return category
        
        return "other"

    def extract_objections_from_content(self, content, file_name="report.md"):
        """Extract objections from report content."""
        try:
            report_info = {
                "file_name": file_name,
                "total_objections": 0,
                "properly_handled": 0,
                "poorly_handled": 0
            }
            
            objection_section_match = re.search(r'## Objection Handling Analysis\n(.*?)(?=\n##|\Z)', 
                                               content, re.DOTALL)
            
            if not objection_section_match:
                self.report_summaries.append(report_info)
                return
            
            objection_section = objection_section_match.group(1)
            
            total_match = re.search(r'Total objections identified: (\d+)', objection_section)
            properly_match = re.search(r'Properly handled: (\d+)', objection_section)
            poorly_match = re.search(r'Poorly handled: (\d+)', objection_section)
            
            if total_match:
                report_info["total_objections"] = int(total_match.group(1))
            if properly_match:
                report_info["properly_handled"] = int(properly_match.group(1).split()[0])
            if poorly_match:
                report_info["poorly_handled"] = int(poorly_match.group(1).split()[0])
            
            self.report_summaries.append(report_info)
            
            objection_details_match = re.search(r'### Objection Details\n(.*?)(?=\n##|\Z)', 
                                               content, re.DOTALL)
                                               
            if not objection_details_match:
                return
                
            objection_details = objection_details_match.group(1)
            
            objection_entries = re.findall(r'- \*\*(.*?)\*\*: (.)\s*\n\s+- Customer: "(.*?)"\s*\n\s+- Agent: "(.*?)"', 
                                          objection_details, re.DOTALL)
            
            for entry in objection_entries:
                objection_type = entry[0]
                handled_marker = entry[1]  # ✓ or ✗
                customer_text = entry[2]
                agent_text = entry[3]
                
                properly_handled = handled_marker == "✓"
                category = self.categorize_objection(customer_text)
                
                objection_data = {
                    "file_name": file_name,
                    "objection_type": objection_type,
                    "customer_text": customer_text,
                    "agent_text": agent_text,
                    "properly_handled": properly_handled,
                    "category": category
                }
                
                self.objections_data.append(objection_data)
                self.all_objections.append(customer_text)
        
        except Exception as e:
            self.failed_files.append({
                "file_name": file_name,
                "error": str(e)
            })
            
    def analyze_report_content(self):
        """Process the provided report content."""
        if not self.report_content:
            print("No report content provided.")
            return
        
        print("Analyzing report content...")
        self.extract_objections_from_content(self.report_content)
        print("Analysis complete.")
        
    def analyze_all_reports(self):
        """Process all report files in the directory."""
        if self.report_content:
            self.analyze_report_content()
            return
            
        print("No report content provided and no directory specified.")
    
    def generate_statistics(self):
        """Generate statistics from the analyzed objections."""
        if not self.objections_data:
            print("No objections data available.")
            return {}

        df = pd.DataFrame(self.objections_data)
        
        # Get counts by category
        category_counts = df['category'].value_counts().to_dict()
        
        # Calculate handling effectiveness by category
        category_effectiveness = {}
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            total_in_category = len(category_df)
            properly_handled = sum(category_df['properly_handled'])
            effectiveness = properly_handled / total_in_category if total_in_category > 0 else 0
            category_effectiveness[category] = {
                'total': total_in_category,
                'properly_handled': properly_handled,
                'effectiveness': effectiveness
            }
        
        # Get most common customer objection texts
        objection_counter = Counter(self.all_objections)
        common_objections = objection_counter.most_common(20)
        
        # Common objection examples for each category
        category_examples = {}
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            examples = category_df['customer_text'].value_counts().head(3).index.tolist()
            category_examples[category] = examples
        
        # Get poorly handled objections for each category
        poorly_handled_examples = {}
        for category in df['category'].unique():
            category_df = df[(df['category'] == category) & (~df['properly_handled'])]
            if not category_df.empty:
                examples = category_df['customer_text'].value_counts().head(3).index.tolist()
                poorly_handled_examples[category] = examples

        # Store category counts and effectiveness for use in other methods
        self.category_counts = category_counts
        self.category_effectiveness = category_effectiveness
        
        return {
            'total_objections': len(self.objections_data),
            'total_reports_with_objections': len(set(obj['file_name'] for obj in self.objections_data)),
            'total_reports_analyzed': len(self.report_summaries),
            'category_counts': category_counts,
            'category_effectiveness': category_effectiveness,
            'common_objections': common_objections,
            'category_examples': category_examples,
            'poorly_handled_examples': poorly_handled_examples,
            'properly_handled_total': sum(1 for obj in self.objections_data if obj['properly_handled']),
            'poorly_handled_total': sum(1 for obj in self.objections_data if not obj['properly_handled']),
        }
        
    def create_visualizations(self, stats, output_dir="objection_analysis_output"):
        """Create visualizations of the objection analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set modern aesthetics
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#d35400', '#c0392b', '#16a085', '#27ae60', '#2980b9']
        
        # 1. Objection categories pie chart
        plt.figure(figsize=(12, 8))
        labels = list(stats['category_counts'].keys())
        sizes = list(stats['category_counts'].values())
        
        # Clean up category labels
        labels = [label.replace('_', ' ').title() for label in labels]
        
        plt.pie(sizes, labels=None, colors=colors[:len(labels)], wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2})
        
        # Add a clean legend instead of labels on pie
        plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        plt.title('Distribution of Objection Categories', fontsize=16, pad=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'objection_categories_pie.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Handling effectiveness by category
        plt.figure(figsize=(14, 8))
        categories = list(stats['category_effectiveness'].keys())
        effectiveness = [stats['category_effectiveness'][cat]['effectiveness'] * 100 for cat in categories]
        
        # Sort by effectiveness
        sorted_indices = sorted(range(len(effectiveness)), key=lambda i: effectiveness[i])
        sorted_categories = [categories[i].replace('_', ' ').title() for i in sorted_indices]
        sorted_effectiveness = [effectiveness[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = plt.barh(sorted_categories, sorted_effectiveness, color=colors[0], height=0.6)
        
        # Add thin grid lines only on x-axis
        plt.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
        
        # Add value labels at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                    ha='left', va='center', fontweight='bold', color='#555555')
        
        plt.xlabel('Properly Handled (%)', fontsize=12, fontweight='bold', labelpad=15)
        plt.ylabel('Objection Category', fontsize=12, fontweight='bold', labelpad=15)
        plt.title('Handling Effectiveness by Objection Category', fontsize=16, pad=20, fontweight='bold')
        
        # Remove border from the plot
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'handling_effectiveness_by_category.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top 10 most common objections
        plt.figure(figsize=(16, 10))
        top_objections = stats['common_objections'][:10]
        
        # Prepare and clean data
        objection_texts = []
        for obj in top_objections:
            text = obj[0]
            if len(text) > 50:
                text = text[:47] + '...'
            objection_texts.append(text)
    
        objection_counts = [obj[1] for obj in top_objections]
        
        # Sort by frequency
        sorted_indices = sorted(range(len(objection_counts)), key=lambda i: objection_counts[i])
        sorted_texts = [objection_texts[i] for i in sorted_indices]
        sorted_counts = [objection_counts[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = plt.barh(sorted_texts, sorted_counts, color=colors[2], height=0.6)
        
        # Add thin grid lines only on x-axis
        plt.grid(axis='x', linestyle='--', alpha=0.3, zorder=0)
        
        # Add value labels at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, str(width), 
                    ha='left', va='center', fontweight='bold', color='#555555')
        
        plt.xlabel('Frequency', fontsize=12, fontweight='bold', labelpad=15)
        plt.ylabel('Objection', fontsize=12, fontweight='bold', labelpad=15)
        plt.title('Top 10 Most Common Objections', fontsize=16, pad=20, fontweight='bold')
        
        # Remove border from the plot
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_objections.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. NEW: Stacked bar chart showing properly vs poorly handled by category
        plt.figure(figsize=(14, 8))
        
        categories = sorted(stats['category_effectiveness'].keys(), 
                          key=lambda x: stats['category_counts'][x], 
                          reverse=True)[:10]  # Top 10 categories
    
        # Clean category names
        category_labels = [cat.replace('_', ' ').title() for cat in categories]
        
        # Data for stacked bars
        properly_handled = [stats['category_effectiveness'][cat]['properly_handled'] for cat in categories]
        poorly_handled = [stats['category_effectiveness'][cat]['total'] - stats['category_effectiveness'][cat]['properly_handled'] for cat in categories]
        
        # Create stacked bar chart
        plt.bar(category_labels, properly_handled, label='Properly Handled', color=colors[1])
        plt.bar(category_labels, poorly_handled, bottom=properly_handled, label='Poorly Handled', color=colors[2])
        
        plt.xlabel('Objection Category', fontsize=12, fontweight='bold', labelpad=15)
        plt.ylabel('Number of Objections', fontsize=12, fontweight='bold', labelpad=15)
        plt.title('Properly vs Poorly Handled Objections by Category', fontsize=16, pad=20, fontweight='bold')
        
        plt.legend(frameon=False)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
        
        # Remove border from the plot
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'handling_comparison_by_category.png'), dpi=300, bbox_inches='tight')
        plt.close()

        
    def generate_report(self, stats, output_dir="objection_analysis_output"):
        """Generate an HTML and Markdown report of the objection analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create Markdown report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        md_report = f"""# Solar Call Objection Analysis Report
        
Generated on: {timestamp}

## Overview Statistics

- **Total Reports Analyzed**: {stats['total_reports_analyzed']}
- **Reports With Objections**: {stats['total_reports_with_objections']} ({stats['total_reports_with_objections']/stats['total_reports_analyzed']*100:.1f}% of total)
- **Total Objections Identified**: {stats['total_objections']}
- **Properly Handled Objections**: {stats['properly_handled_total']} ({stats['properly_handled_total']/stats['total_objections']*100:.1f}%)
- **Poorly Handled Objections**: {stats['poorly_handled_total']} ({stats['poorly_handled_total']/stats['total_objections']*100:.1f}%)

## Objection Categories Analysis

| Category | Count | Percentage | Properly Handled | Effectiveness |
|----------|-------|------------|-----------------|---------------|
"""
        
        # Sort categories by count (descending)
        sorted_categories = sorted(stats['category_counts'].keys(), 
                                 key=lambda x: stats['category_counts'][x], 
                                 reverse=True)
        
        for category in sorted_categories:
            count = stats['category_counts'][category]
            percentage = count / stats['total_objections'] * 100
            effectiveness = stats['category_effectiveness'][category]
            
            md_report += f"| {category} | {count} | {percentage:.1f}% | {effectiveness['properly_handled']}/{effectiveness['total']} | {effectiveness['effectiveness']*100:.1f}% |\n"
        
        # Add examples for each category
        md_report += "\n## Common Objections by Category\n\n"
        
        for category in sorted_categories:
            md_report += f"### {category.replace('_', ' ').title()}\n\n"
            
            if category in stats['category_examples']:
                md_report += "**Common Examples:**\n\n"
                for i, example in enumerate(stats['category_examples'][category], 1):
                    md_report += f"{i}. \"{example}\"\n"
            
            if category in stats['poorly_handled_examples']:
                md_report += "\n**Commonly Mishandled Examples:**\n\n"
                for i, example in enumerate(stats['poorly_handled_examples'][category], 1):
                    md_report += f"{i}. \"{example}\"\n"
            
            md_report += "\n"
        
        # Recommendations section
        md_report += """## Key Recommendations

1. **Improve Data Privacy Objection Handling**: Train agents to better address concerns about data sources and privacy.

2. **Create Specific Objection Responses**: Develop specialized scripts for the top objection categories.

3. **Focus Training on Problem Areas**: Prioritize training for objection categories with the lowest handling effectiveness.

4. **Regular Monitoring**: Implement regular reviews of objection handling performance.

5. **Script Refinement**: Update call scripts to include better responses to the most common objections.

## Failed Analysis Files

"""
        if self.failed_files:
            md_report += f"- {len(self.failed_files)} files failed during analysis\n"
            for fail in self.failed_files[:10]:  # Show only first 10 failures
                md_report += f"  - {fail['file_path']}: {fail['error']}\n"
            if len(self.failed_files) > 10:
                md_report += f"  - ... and {len(self.failed_files) - 10} more\n"
        else:
            md_report += "- No files failed during analysis\n"
        
        # Save the markdown report
        with open(os.path.join(output_dir, 'objection_analysis_report.md'), 'w', encoding='utf-8') as f:
            f.write(md_report)

        # Convert to HTML (simple version)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Solar Call Objection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1, h2, h3, h4 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .chart-container {{ margin: 30px 0; text-align: center; }}
                .chart-container img {{ max-width: 100%; height: auto; }}
                .example-box {{ background-color: #f9f9f9; border-left: 4px solid #2c3e50; padding: 15px; margin: 15px 0; }}
                .properly-handled {{ border-left: 4px solid #2ecc71; }}
                .poorly-handled {{ border-left: 4px solid #e74c3c; }}
                .example-label {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-size: 12px; margin-bottom: 10px; }}
                .proper-label {{ background-color: #2ecc71; }}
                .poor-label {{ background-color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>Solar Call Objection Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <h2>Overview Statistics</h2>
            <ul>
                <li><strong>Total Reports Analyzed</strong>: {stats['total_reports_analyzed']}</li>
                <li><strong>Reports With Objections</strong>: {stats['total_reports_with_objections']} ({stats['total_reports_with_objections']/stats['total_reports_analyzed']*100:.1f}% of total)</li>
                <li><strong>Total Objections Identified</strong>: {stats['total_objections']}</li>
                <li><strong>Properly Handled Objections</strong>: {stats['properly_handled_total']} ({stats['properly_handled_total']/stats['total_objections']*100:.1f}%)</li>
                <li><strong>Poorly Handled Objections</strong>: {stats['poorly_handled_total']} ({stats['poorly_handled_total']/stats['total_objections']*100:.1f}%)</li>
            </ul>
            
            <div class="chart-container">
                <h3>Distribution of Objection Categories</h3>
                <img src="objection_categories_pie.png" alt="Objection Categories Pie Chart">
            </div>
            
            <div class="chart-container">
                <h3>Handling Effectiveness by Category</h3>
                <img src="handling_effectiveness_by_category.png" alt="Handling Effectiveness Chart">
            </div>
            
            <div class="chart-container">
                <h3>Top Most Common Objections</h3>
                <img src="top_objections.png" alt="Top Objections Chart">
            </div>
        
            <h2>Objection Categories Analysis</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Properly Handled</th>
                    <th>Effectiveness</th>
                </tr>
        """
        
        for category in sorted(stats['category_counts'].keys(), 
                             key=lambda x: stats['category_counts'][x], 
                             reverse=True):
            count = stats['category_counts'][category]
            percentage = count / stats['total_objections'] * 100
            effectiveness = stats['category_effectiveness'][category]
            
            html_report += f"""
                <tr>
                    <td>{category.replace('_', ' ').title()}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{effectiveness['properly_handled']}/{effectiveness['total']}</td>
                    <td>{effectiveness['effectiveness']*100:.1f}%</td>
                </tr>
            """
        
        html_report += """
            </table>
            
            <h2>Common Objections by Category</h2>
        """
        
        # Get detailed objection examples
        detailed_examples = self.extract_top_objection_examples(top_n=5, examples_per_category=2)
    
        for category, examples in detailed_examples.items():
            display_category = category.replace('_', ' ').title()
            html_report += f"<h3>{display_category}</h3>"
            
            # Add category stats
            html_report += f"<p>This category represents {self.category_counts.get(category, 0)} objections "
            html_report += f"({self.category_counts.get(category, 0)/len(self.objections_data)*100:.1f}% of all objections).</p>"
            
            # Add handling effectiveness info
            if category in self.category_effectiveness:
                effectiveness = self.category_effectiveness[category]
                html_report += f"<p><strong>Handling Effectiveness</strong>: {effectiveness['properly_handled']}/{effectiveness['total']} "
                html_report += f"({effectiveness['effectiveness']*100:.1f}%) properly handled.</p>"
            
            # Add examples with styling
            for i, example in enumerate(examples, 1):
                properly_handled = example['properly_handled']
                box_class = "properly-handled" if properly_handled else "poorly-handled"
                label_class = "proper-label" if properly_handled else "poor-label"
                label_text = "Properly Handled" if properly_handled else "Poorly Handled"
                
                html_report += f"""
                <div class="example-box {box_class}">
                    <span class="example-label {label_class}">{label_text}</span>
                    <h4>Objection Type: {example['objection_type']}</h4>
                    <p><strong>Customer</strong>: "{example['customer_text']}"</p>
                    <p><strong>Agent Response</strong>: "{example['agent_text']}"</p>
                """
                
                # Add analysis for poorly handled examples
                if not properly_handled:
                    html_report += f"""
                    <p><strong>Analysis</strong>: This objection was poorly handled. 
                    The agent's response did not effectively address the customer's concern.</p>
                    
                    <p><strong>Potential Improvement</strong>: Consider a more empathetic acknowledgment 
                    of the customer's concern and provide clear information addressing their specific objection.</p>
                    """
                
                html_report += "</div>"
    
        # Add recommendations section
        html_report += """
            <h2>Key Recommendations</h2>
            <ol>
                <li><strong>Improve Data Privacy Objection Handling</strong>: Train agents to better address concerns about data sources and privacy.</li>
                <li><strong>Create Specific Objection Responses</strong>: Develop specialized scripts for the top objection categories.</li>
                <li><strong>Focus Training on Problem Areas</strong>: Prioritize training for objection categories with the lowest handling effectiveness.</li>
                <li><strong>Regular Monitoring</strong>: Implement regular reviews of objection handling performance.</li>
                <li><strong>Script Refinement</strong>: Update call scripts to include better responses to the most common objections.</li>
            </ol>
        """
        
        if self.failed_files:
            html_report += f"<h2>Failed Analysis Files</h2><p>{len(self.failed_files)} files failed during analysis:</p><ul>"
            for fail in self.failed_files[:10]:
                html_report += f"<li>{fail['file_path']}: {fail['error']}</li>"
            if len(self.failed_files) > 10:
                html_report += f"<li>... and {len(self.failed_files) - 10} more</li>"
            html_report += "</ul>"
        
        html_report += """
        </body>
        </html>
        """
        
        # Save the HTML report
        with open(os.path.join(output_dir, 'objection_analysis_report.html'), 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"Reports saved to {output_dir}")
        return os.path.join(output_dir, 'objection_analysis_report.md')

    def extract_top_objection_examples(self, top_n=5, examples_per_category=3):
        """Extract detailed examples of the top objections with agent responses."""
        if not self.objections_data:
            print("No objections data available.")
            return {}
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(self.objections_data)
        
        # Get the top N categories by frequency
        top_categories = sorted(self.category_counts.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:top_n]
        top_category_names = [cat[0] for cat in top_categories]
        
        detailed_examples = {}
        for category in top_category_names:
            # Filter objections for this category
            category_df = df[df['category'] == category].copy()
            
            # Check if we have any examples
            if category_df.empty:
                continue
                
            # Get examples for this category
            examples = []
            # Try to get a mix of properly and poorly handled examples if available
            if sum(category_df['properly_handled']) > 0 and sum(~category_df['properly_handled']) > 0:
                # Get some properly handled examples
                proper_examples = category_df[category_df['properly_handled']].head(examples_per_category // 2)
                # Get some poorly handled examples
                poor_examples = category_df[~category_df['properly_handled']].head(examples_per_category - len(proper_examples))
                category_examples = pd.concat([proper_examples, poor_examples])
            else:
                # Just get what we have
                category_examples = category_df.head(examples_per_category)
            
            # Format the examples
            for _, row in category_examples.iterrows():
                examples.append({
                    'customer_text': row['customer_text'],
                    'agent_text': row['agent_text'],
                    'properly_handled': row['properly_handled'],
                    'objection_type': row['objection_type']
                })
            
            detailed_examples[category] = examples
        
        return detailed_examples

    def generate_objection_examples_report(self, detailed_examples, output_dir="objection_analysis_output"):
        """Generate a markdown report with detailed objection examples."""
        os.makedirs(output_dir, exist_ok=True)
        
        md_report = """# Top Objection Examples with Agent Responses
    
    This report provides detailed examples of the most common customer objections and how agents responded to them.
    
    """
        
        for category, examples in detailed_examples.items():
            # Format category name for display
            display_category = category.replace('_', ' ').title()
            md_report += f"## {display_category} Objections\n\n"
            
            # Add description of the category
            md_report += f"This category represents {self.category_counts.get(category, 0)} objections "
            md_report += f"({self.category_counts.get(category, 0)/len(self.objections_data)*100:.1f}% of all objections).\n\n"
            
            # Add handling effectiveness info
            if category in self.category_effectiveness:
                effectiveness = self.category_effectiveness[category]
                md_report += f"**Handling Effectiveness**: {effectiveness['properly_handled']}/{effectiveness['total']} "
                md_report += f"({effectiveness['effectiveness']*100:.1f}%) properly handled.\n\n"
            
            # Add examples
            for i, example in enumerate(examples, 1):
                status = "✅ Properly Handled" if example['properly_handled'] else "❌ Poorly Handled"
                md_report += f"### Example {i}: {status}\n\n"
                md_report += f"**Objection Type**: {example['objection_type']}\n\n"
                md_report += f"**Customer**: \"{example['customer_text']}\"\n\n"
                md_report += f"**Agent Response**: \"{example['agent_text']}\"\n\n"
                
                # Add analysis section (for poorly handled examples)
                if not example['properly_handled']:
                    md_report += "**Analysis**: This objection was poorly handled. "
                    md_report += "The agent's response did not effectively address the customer's concern.\n\n"
                    
                    # Add potential improvement suggestion
                    md_report += "**Potential Improvement**: Consider a more empathetic acknowledgment "
                    md_report += "of the customer's concern and provide clear information about how "
                    md_report += "their data is used and protected.\n\n"
                
                md_report += "---\n\n"
        
        # Add recommendations section
        md_report += """## Recommendations for Improving Objection Handling

    Based on the examples above, here are some recommendations for improving how your AI agent handles common objections:
    
    1. **Acknowledge Concerns**: Always acknowledge the customer's concern before attempting to address it.
    
    2. **Provide Clear Information**: Give specific, transparent information rather than vague reassurances.
    
    3. **Respect Customer Boundaries**: If a customer firmly states they're not interested, respect their decision rather than continuing to push.
    
    4. **Prepare for Common Objections**: Develop specific, well-crafted responses for the most common objection types.
    
    5. **Track Effectiveness**: Continue monitoring which responses are most effective at overcoming objections.
    """
        
        # Save the report
        report_path = os.path.join(output_dir, 'detailed_objection_examples.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        return report_path

    def generate_objection_improvement_suggestions(self, output_dir="objection_analysis_output"):
        """Generate improvement suggestions for handling common objections."""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.objections_data:
            print("No objections data available.")
            return None
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.objections_data)
        
        # Create a dictionary to hold improvement suggestions
        improvement_suggestions = {}
        
        # For each category, analyze poorly handled objections
        for category in self.category_counts.keys():
            # Filter objections for this category that were poorly handled
            poorly_handled = df[(df['category'] == category) & (~df['properly_handled'])]
            
            if poorly_handled.empty:
                continue
            
            # Collect common patterns in customer objections
            common_words = []
            for text in poorly_handled['customer_text']:
                # Split into words and remove common stop words
                words = text.lower().split()
                common_words.extend(words)
            
            # Count word frequency
            word_counts = Counter(common_words)
            
            # Identify common themes based on word frequency
            theme_keywords = [word for word, count in word_counts.most_common(10) if len(word) > 3]
            
            # Format the improvement suggestion
            suggestion = {
                'count': len(poorly_handled),
                'common_themes': theme_keywords,
                'example_objections': poorly_handled['customer_text'].head(3).tolist(),
                'example_poor_responses': poorly_handled['agent_text'].head(3).tolist(),
                'suggested_improvements': self._generate_suggested_response(category, theme_keywords)
            }
            
            improvement_suggestions[category] = suggestion
        
        # Generate a report with the suggestions
        md_report = """# Objection Handling Improvement Suggestions
    
    This report provides suggestions for improving how agents handle common customer objections.
    
    """
    
        # Sort categories by the count of poorly handled objections
        sorted_categories = sorted(improvement_suggestions.keys(), 
                                key=lambda x: improvement_suggestions[x]['count'], 
                                reverse=True)
        
        for category in sorted_categories:
            suggestion = improvement_suggestions[category]
            display_category = category.replace('_', ' ').title()
            
            md_report += f"## {display_category} Objections\n\n"
            md_report += f"**Poorly Handled Count**: {suggestion['count']}\n\n"
            
            md_report += "**Common Themes**: " + ", ".join(suggestion['common_themes'][:5]) + "\n\n"
            
            md_report += "### Example Problematic Exchanges\n\n"
            for i, (objection, response) in enumerate(zip(suggestion['example_objections'], 
                                                       suggestion['example_poor_responses']), 1):
                md_report += f"**Example {i}**:\n"
                md_report += f"- Customer: \"{objection}\"\n"
                md_report += f"- Agent: \"{response}\"\n\n"
            
            md_report += "### Suggested Improved Responses\n\n"
            for i, improvement in enumerate(suggestion['suggested_improvements'], 1):
                md_report += f"{i}. {improvement}\n"
            
            md_report += "\n---\n\n"
        
        # Save the report
        report_path = os.path.join(output_dir, 'objection_improvement_suggestions.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        return report_path

    def _generate_suggested_response(self, category, themes):
        """Generate suggested improved responses based on objection category and themes."""
        responses = []
        
        if category == 'data_privacy':
            responses = [
                "Acknowledge the customer's concern about their personal information.",
                "Clearly explain the specific source of the customer data (e.g., public records, partner company).",
                "Reassure them about data protection measures and compliance with privacy laws.",
                "Offer to remove their information from the calling list if they prefer.",
                "Be transparent about how their information will be used specifically for the solar assessment."
            ]
        elif category == 'not_interested':
            responses = [
                "Acknowledge their position respectfully without pushing back.",
                "Briefly mention the key benefit they might be missing (e.g., 'Many homeowners save 30% on electricity').",
                "Ask a single question to understand why they're not interested before accepting.",
                "Thank them for their time and end the call professionally.",
                "Avoid multiple attempts to change their mind if they've clearly stated they're not interested."
            ]
        elif category == 'property_unsuitable':
            responses = [
                "Acknowledge their property situation without contradicting them.",
                "Clarify if there are alternative options for their situation if applicable.",
                "Ask if they know anyone else who might benefit from solar programs.",
                "Thank them for their time and avoid trying to qualify obviously unsuitable properties.",
                "End the call professionally rather than continuing to push."
            ]
        elif category == 'trust_issues':
            responses = [
                "Acknowledge their skepticism as reasonable and valid.",
                "Provide specific company credentials and verification methods.",
                "Offer to send official information via email before proceeding.",
                "Explain the legitimate government/utility programs behind the offer.",
                "Suggest they research the company/program independently and offer a callback."
            ]
        else:
            responses = [
                "Listen fully to the customer's objection without interrupting.",
                "Acknowledge their concern specifically, showing you understood their point.",
                "Provide clear, relevant information that directly addresses their specific concern.",
                "Avoid generic scripts that don't address their particular objection.",
                "If the customer remains uninterested after addressing concerns, respect their decision."
            ]
        
        return responses
    
def analyze_md_content(content):
    """Analyze the provided markdown content."""
    analyzer = SolarCallAnalyzer(content)
    analyzer.analyze_report_content()
    
    stats = analyzer.generate_statistics()
    
    if not stats:
        print("No objection data was found in the report.")
        return None
    
    output_dir = "objection_analysis_output"
    
    analyzer.create_visualizations(stats, output_dir)
    
    report_path = analyzer.generate_report(stats, output_dir)
    
    detailed_examples = analyzer.extract_top_objection_examples(top_n=5, examples_per_category=3)
    examples_report_path = analyzer.generate_objection_examples_report(detailed_examples, output_dir)
    
    suggestions_report_path = analyzer.generate_objection_improvement_suggestions(output_dir)
    
    print(f"Analysis complete. Main report saved to {report_path}")
    print(f"Detailed objection examples saved to {examples_report_path}")
    print(f"Improvement suggestions saved to {suggestions_report_path}")
    print(f"Total objections identified: {stats['total_objections']}")
    
    if 'category_counts' in stats and stats['category_counts']:
        top_categories = sorted(stats['category_counts'].items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 objection categories:")
        for category, count in top_categories:
            print(f"  - {category}: {count} ({count/stats['total_objections']*100:.1f}%)")
    
    return stats
    
def main():
    """Main function to run the transcript processing and analysis."""
    try:
        report_content = None
        results = None
        # Part 1: Process transcripts from CSV
        try:
            df = pd.read_csv('data.csv')
            # Don't limit to specific rows when processing from API
            all_reports = {}
            if 'Transcription URL' not in df.columns:
                print("Error: 'Transcription URL' column not found in data.csv")
                return {"success": False, "error": "'Transcription URL' column not found"}
            
            for index, row in df.iterrows():
                transcript_url = row['Transcription URL']
                transcript_id = index + 1
                if pd.isna(transcript_url) or not transcript_url:
                    print(f"Skipping row {index+1}: No URL provided")
                    continue
                
                print(f"Processing transcript {transcript_id} from {transcript_url}")
                json_str = fetch_transcript_from_url(transcript_url)
                if json_str:
                    try:
                        report_content = analyze_call_transcript(json_str)
                        all_reports[f'transcript_{transcript_id}_report'] = report_content
                        # Save the image with unique name
                        if os.path.exists('call_analysis.png'):
                            os.rename('call_analysis.png', f'transcript_{transcript_id}_analysis.png')
                        print(f"Successfully processed transcript {transcript_id}")
                    except Exception as e:
                        print(f"Error analyzing transcript {transcript_id}: {e}")
                        return {"success": False, "error": f"Error analyzing transcript: {str(e)}"}
                else:
                    print(f"Failed to fetch transcript {transcript_id}")
                    return {"success": False, "error": f"Failed to fetch transcript from URL"}
            
            # Write to markdown file
            with open('all_reports.md', 'w') as f:
                for report_id, content in all_reports.items():
                    f.write(f"# {report_id}\n\n")
                    f.write(content)
                    f.write("\n\n---\n\n")
            
        except FileNotFoundError:
            print("Error: data.csv file not found")
            return {"success": False, "error": "data.csv file not found"}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
        
        # Part 2: Analyze the processed reports
        results = analyze_md_content(report_content)
        
        # Create a structured response
        response = {
            "success": True,
            "report": report_content,
            "analysis_results": results,
            "charts": []
        }
        
        # Add information about generated charts
        chart_files = [f for f in os.listdir('.') if f.startswith('transcript_') and f.endswith('_analysis.png')]
        if not chart_files and os.path.exists('call_analysis.png'):
            chart_files = ['call_analysis.png']
        
        # Ensure we have at least one chart
        if not chart_files:
            # Create a placeholder visualization if no charts were generated
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "No visualization available for this transcript", 
                         ha='center', va='center', fontsize=14)
                plt.axis('off')
                plt.savefig('default_visualization.png')
                chart_files = ['default_visualization.png']
                print("Created default visualization")
            except Exception as e:
                print(f"Failed to create default visualization: {e}")
        
        for chart_file in chart_files:
            response["charts"].append({
                "name": chart_file,
                "path": chart_file
            })
        
        # Print as JSON so it can be captured by the Node.js script
        import json
        print(json.dumps(response))
        return response
        
    except Exception as e:
        print(f"Unexpected error in main function: {e}")
        return {"success": False, "error": f"Unexpected error in main function: {str(e)}"}

if __name__ == "__main__":
    import pandas as pd
    import os
    import json
    main()