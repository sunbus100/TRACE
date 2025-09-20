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


def generate_final_response(client, conversation_text, emotion_label, trigger_spans, cause_summary, cause_category,
                            chosen_strategy, rag_examples_str: str):
    """
    The final agent, with a revised prompt emphasizing grounding in the user's context by modifying the existing structure.
    """
    strategy_definitions = """
    - **ER (Emotional Reaction)**: Express your own feelings (e.g., warmth, compassion) in response to the seeker's situation.
    - **IP (Interpretation)**: Communicate a cognitive understanding of the seeker's inferred feelings and experiences.
    - **EX (Exploration)**: Gently ask a specific, probing question to help the seeker explore their feelings further.
    """

    system_prompt = f"""You are an expert and compassionate AI counselor. Your role is to synthesize a complete case file and generate a final, nuanced empathetic response.

    ### ANALYSIS PIPELINE SUMMARY:
    1.  **Agent 1 (Emotion Classifier):** Identified the subject's dominant emotion.
    2.  **Agent 2 (Cause Analyst):** Pinpointed the emotional trigger and root cause.
    3.  **Agent 3 (Strategy Decider):** Recommended a primary response strategy to use.

    ### RESPONSE GENERATION GUIDELINES
    You must strictly reference the provided successful examples. Your final response should emulate the **tone, phrasing, and conciseness** of the `SUCCESSFUL EMPATHETIC RESPONSE` in the cases.

    While adopting their style, your response must still be **original**. Aim for **1-2 impactful sentences**.

    --- BEGIN EXAMPLES ---
    {rag_examples_str}
    --- END EXAMPLES ---

    ### ADVANCED TECHNIQUE: STRATEGY BLENDING
    Merely using the primary strategy is not enough. A top-tier empathetic response is often a clever combination of multiple strategies.

    **Blending Rule:** Your task is to **always attempt to blend strategies**. Start your response with the **Primary Strategy** recommended by Agent 3, then naturally integrate a **Secondary Strategy** to conclude or supplement the message.

    ### YOUR FINAL TASK:
    Generate the final empathetic response. The response must adhere to the following critical requirements:

    - **Context-Awareness (THE GOLDEN RULE):** This is your **highest priority**. Your response **MUST** directly connect with the user's words to avoid sounding generic.
        - **1. Review** the **"Emotion Trigger Sentences"** provided in the case file.
        - **2. Select** a specific, concrete detail from those sentences (e.g., "my puppy," "my boss").
        - **3. Weave** this specific detail into your response to prove you are listening. This is non-negotiable.

    - **Strategy-Guided:** The `chosen_strategy` should be the main theme, but skillfully blended with a secondary one as instructed above.
    - **Empathetic and Natural:** The tone should be supportive and human-like.
    - **Non-judgmental:** Do NOT give unsolicited advice.

    ### Empathy Strategy Definitions:
    {strategy_definitions}

    You must provide your response in a valid JSON format with a single key: "final_empathetic_response".

    ### JSON Schema:
    ```json
    {{
      "final_empathetic_response": "string"
    }}```
    """

    user_prompt = f"""Please generate the final empathetic response for the following case file, guided by the chosen primary strategy and blending others where appropriate:

    ### CASE FILE START ###

    **1. Original Dialogue:**
    {conversation_text}

    **2. Agent 1 Analysis (Emotion Identification):**
    - Identified Emotion: {emotion_label}

    **3. Agent 2 Analysis (Cause & Trigger Identification):**
    - Emotion Trigger Sentences: {trigger_spans}
    - Summary of Cause: {cause_summary}
    - Category of Cause: {cause_category}

    **4. Agent 3 Analysis (Strategy Decision):**
    - Chosen Primary Strategy: **{chosen_strategy}**

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
                temperature=0.5
            )
            return json.loads(chat_completion.choices[0].message.content)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            last_exception = e
            time.sleep((INITIAL_SLEEP_TIME ** attempt) + random.uniform(0, 1))
        except Exception as e:
            logger.error(f"A non-retriable error occurred: {e}")
            return {"error": str(e)}
    logger.error(f"Failed after {MAX_RETRIES} retries. Last error: {last_exception}")
    return {"error": str(last_exception)}


def process_row(args_tuple):
    index, row, api_keys, id_to_emotion_map, rag_system = args_tuple  # 新增rag_system
    api_key = api_keys[index % len(api_keys)]
    try:
        if pd.isna(row['chosen_strategy']) or row['chosen_strategy'] in ['SKIPPED', 'ERROR']:
            return {**row, 'final_empathetic_response': 'SKIPPED_DUE_TO_PREVIOUS_ERROR'}

        emotion_id_for_query = str(int(row['predicted_id']))

        emotion_label_for_prompt = id_to_emotion_map.get(int(row['predicted_id']), "unknown")

        if 'prompt' not in row or pd.isna(row['prompt']):
            logger.warning(f"Row {index} is missing a 'prompt'. Falling back to 'utterances' for RAG query. Results may be suboptimal.")
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
            base_url="",  # you need to input your own url here
            default_query={"api-version": "preview"},
            timeout=20.0
        )
        response_result = generate_final_response(
            client,
            row['utterances'],
            emotion_label_for_prompt,
            row['trigger_spans'],
            row['global_cause_summary'],
            row['cause_category'],
            row['chosen_strategy'],
            rag_examples_str,
        )

        return {
            **row,
            'final_empathetic_response': response_result.get('final_empathetic_response'),
            'rag_examples_used': rag_examples_str
        }
    except Exception as e:
        logger.error(f"Critical error in process_row for row {index}: {e}")
        return {**row, 'final_empathetic_response': f"ERROR: {e}", 'rag_examples_used': 'ERROR'}

api = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Agent 4 for final empathetic response generation.')

    parser.add_argument('--input_csv_path', type=str, default='./output/.csv') # your need to input the path of the output file of agent3 here
    parser.add_argument('--mapping_path', type=str, default='./dataset/emotion_to_id_ekman.json')
    parser.add_argument('--output_path', type=str, default='./output')
    parser.add_argument('--api_keys', type=str, default=api)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    api_keys = [k.strip() for k in args.api_keys.split(',') if k.strip()]
    if not api_keys:
        raise ValueError("API keys are required.")

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
        for result in tqdm(executor.map(process_row, tasks), total=len(tasks), desc="Generating Final Responses with RAG"):
            if result:
                results.append(result)

    time_now = time.strftime("%Y%m%d-%H%M%S")
    final_output_path = f'{args.output_path}/final_pipeline_output_rag_{time_now}.csv'
    final_df = pd.DataFrame(results)
    final_df.to_csv(final_output_path, index=False)

    logger.info(f"Complete pipeline results saved to: {final_output_path}")
    print("\n--- Final Empathetic Response Preview ---")
    print(final_df[['utterances', 'chosen_strategy', 'final_empathetic_response']].tail())
    logger.info("All done!")
