import pandas as pd
import time
import json
import logging
import openai
import concurrent.futures
import random
from tqdm import tqdm
import os
import argparse
from rag_handler import EmpatheticRAG, intelligent_search_cases, format_rag_examples_for_prompt

MAX_RETRIES = 3
INITIAL_SLEEP_TIME = 3

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def decide_sympathy_strategy(client, conversation_text, emotion_label, trigger_spans, cause_summary, cause_category, rag_examples_str):
    strategy_definitions = """
    - **ER (Emotional Reaction)**: Expressing your own emotions (e.g., warmth, compassion, concern) experienced after reading the seeker's situation. The goal is to establish empathic rapport. A strong response explicitly labels the felt emotion (e.g., "I feel really sad for you").
    - **IP (Interpretation)**: Communicating a cognitive understanding of the feelings and experiences inferred from the seeker's situation. This can be done by specifying the inferred feeling (e.g., "This must be terrifying") or by communicating understanding through descriptions of similar personal experiences.
    - **EX (Exploration)**: Improving understanding of the seeker by exploring feelings and experiences not explicitly stated. This shows active interest by asking a specific, gentle probing question (e.g., "Are you feeling alone right now?").
    """

    system_prompt = f"""You are an expert psychological counselor, acting as the final decision-making agent in a three-stage analysis pipeline. Your role is to synthesize the findings of the previous agents and decide on the most appropriate empathetic strategy.

    ### ANALYSIS PIPELINE OVERVIEW:
    1.  **Agent 1 (Emotion Classifier):** This agent analyzed the raw dialogue to identify the subject's dominant emotion (e.g., 'happiness'). It was aware that each broad emotion category is composed of many finer-grained feelings (e.g., 'happiness' can mean joy, pride, or sentimentality).
    2.  **Agent 2 (Cause Analyst):** This agent performed a deep-dive analysis to understand "why" the emotion occurred. It provided three key insights which you will receive:
        - **Trigger Spans:** The specific, emotionally-charged sentences spoken by the subject that act as the emotional "epicenter."
        - **Global Cause Summary:** A one-sentence narrative summary explaining the root cause of the emotion.
        - **Cause Category:** A formal classification of the cause based on psychological theories of emotion antecedents (e.g., 'Social Connection & Affection').
    
    ### HISTORICAL CASE REFERENCE
    Below are relevant historical success cases for your reference.
    
    Please analyze the strategies (ER, IP, or EX) used in these examples to gain inspiration, but your final decision must still be tailored to the current, unique conversation. You can reflect any insights from the cases in your reasoning.
    {rag_examples_str}
    
    ### YOUR TASK:
    Now, as the final expert, you will receive a complete case file containing the original dialogue and the analyses from Agents 1 & 2. Your sole responsibility is to synthesize these findings and **choose the single most appropriate empathetic response strategy** to use next. Do NOT write the actual response.
    
    ### Response Strategies:
    {strategy_definitions}
    
    You must strictly provide your response in a valid JSON format with two keys: "reasoning_for_choice" and "chosen_strategy" (which must be one of "ER", "IP", or "EX").
    
    ### JSON Schema:
    ```json
    {{
      "reasoning_for_choice": "string", // A brief, expert explanation for why you chose this specific strategy based on the provided context.
      "chosen_strategy": "string" // This MUST be one of the exact strings: "ER", "IP", or "EX".
    }}```
    """

    user_prompt = f"""Please decide on the best empathetic strategy for the following case file:
    
    ### CASE FILE START ###
    
    **1. Original Dialogue:**
    {conversation_text}
    
    **2. Analysis from Agent 1 (Emotion Identification):**
    - Identified Emotion: {emotion_label}
    
    **3. Analysis from Agent 2 (Cause & Trigger Identification):**
    - Emotion Trigger Sentences: {trigger_spans}
    - Summary of Cause: {cause_summary}
    - Category of Cause: {cause_category}
    
    ### CASE FILE END ###
    ### Your Analysis (JSON format):
    """

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            response_content = chat_completion.choices[0].message.content
            return json.loads(response_content)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            last_exception = e
            time.sleep((INITIAL_SLEEP_TIME ** attempt) + random.uniform(0, 1))
        except Exception as e:
            logger.error(f"A non-retriable error occurred: {e}")
            return {"error": str(e)}

    logger.error(f"Failed after {MAX_RETRIES} retries. Last error: {last_exception}")
    return {"error": str(last_exception)}


def process_row(args_tuple):
    index, row, api_keys, id_to_emotion_map, rag_system = args_tuple
    api_key = api_keys[index % len(api_keys)]
    try:
        if pd.isna(row['predicted_id']) or row['predicted_id'] < 0 or pd.isna(row['global_cause_summary']):
            return {**row, 'reasoning_for_choice': 'SKIPPED', 'chosen_strategy': 'SKIPPED',
                    'rag_examples_used_by_A3': 'SKIPPED'}

        emotion_id_for_query = str(int(row['predicted_id']))

        emotion_label_for_prompt = id_to_emotion_map.get(int(row['predicted_id']), "unknown")

        if 'prompt' not in row or pd.isna(row['prompt']):
            logger.warning(
                f"Row {index} is missing a 'prompt'. Falling back to 'utterances' for RAG query. Results may be suboptimal.")
            query_text = row['utterances']
        else:
            query_text = row['prompt']

        retrieved_cases = intelligent_search_cases(
            rag_system=rag_system,
            current_prompt=query_text,
            current_emotion=emotion_id_for_query,
            top_k=2
        )
        rag_examples_str = format_rag_examples_for_prompt(retrieved_cases)

        client = openai.OpenAI(
            api_key=api_key,
            base_url="", # input your own url here
            default_query={"api-version": "preview"},
            timeout=20.0
        )

        strategy_result = decide_sympathy_strategy(
            client,
            row['utterances'],
            emotion_label_for_prompt,  #
            row['trigger_spans'],
            row['global_cause_summary'],
            row['cause_category'],
            rag_examples_str
        )

        print(strategy_result.get('chosen_strategy'))

        return {
            **row,
            'reasoning_for_choice': strategy_result.get('reasoning_for_choice'),
            'chosen_strategy': strategy_result.get('chosen_strategy'),
            'rag_examples_used_by_A3': rag_examples_str
        }
    except Exception as e:
        logger.error(f"Critical error in process_row for row {index}: {e}")
        return {**row, 'reasoning_for_choice': f"ERROR: {e}", 'chosen_strategy': 'ERROR',
                'rag_examples_used_by_A3': 'ERROR'}

api = '' # input your own api here

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Agent 3 for empathetic strategy decision.')
    parser.add_argument('--input_csv_path', type=str, default='./output/.csv') # you need to input the path of the output file of agent2 here
    parser.add_argument('--mapping_path', type=str, default='./dataset/emotion_to_id_ekman.json')
    parser.add_argument('--output_path', type=str, default='./output')
    parser.add_argument('--api_keys', type=str, default=api)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    api_keys = [k.strip() for k in args.api_keys.split(',') if k.strip()]
    if not api_keys:
        raise ValueError("API keys are required. Provide them via the --api_keys argument or by setting the DEFAULT_API_KEYS variable.")

    logger.info("Initializing RAG system for all workers...")
    rag_system = EmpatheticRAG()
    logger.info("RAG system initialized successfully.")

    logger.info(f"Using {args.workers} threads with {len(api_keys)} API keys.")

    try:
        df = pd.read_csv(args.input_csv_path)
        with open(args.mapping_path, 'r') as f:
            emotion_to_id_map = json.load(f)
        id_to_emotion_map = {int(v): k for k, v in emotion_to_id_map.items()}
    except FileNotFoundError as e:
        logger.error(f"Could not find a necessary file: {e}")
        exit()

    tasks = [(index, row, api_keys, id_to_emotion_map, rag_system) for index, row in df.iterrows()]
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(process_row, tasks), total=len(tasks), desc="Deciding Empathetic Strategies with RAG"):
            if result:
                results.append(result)

    time_now = time.strftime("%Y%m%d-%H%M%S")
    final_output_path = f'{args.output_path}/final_strategy_output_rag_{time_now}.csv'
    final_df = pd.DataFrame(results)
    final_df.to_csv(final_output_path, index=False)

    logger.info(f"Complete strategy decisions saved to: {final_output_path}")
    print("\n--- Strategy Decision Results Preview ---")
    print(final_df[['utterances', 'predicted_id', 'chosen_strategy', 'reasoning_for_choice']].tail())
    logger.info("All done!")

