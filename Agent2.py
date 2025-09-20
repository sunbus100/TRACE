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


MAX_RETRIES = 3
INITIAL_SLEEP_TIME = 5

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def analyze_emotion_cause_with_theory(client, conversation_text, emotion_label):

    cause_taxonomy = [
        'Social Connection & Affection',
        'Conflict & Injustice',
        'Loss & Failure',
        'Achievement & Success',
        'Threat & Potential Danger',
        'Unexpectedness & Novelty'
    ]

    system_prompt = f"""You are a highly perceptive psychological analyst.
    
    ### BACKGROUND CONTEXT
    The dialogue you are about to analyze comes from a controlled experiment where a subject (the 'Seeker') is in a specific emotional state while talking to an experimenter (the 'Responder').
    
    A preliminary expert agent has already identified the dominant Ekman emotion for this dialogue. That agent was informed that each of the six Ekman categories is a broad label for more specific, fine-grained emotions. The emotion label you are given is the result of that nuanced, initial analysis.

    ### YOUR TASK
    You must perform two sub-tasks and provide the output in a single, valid JSON format.
    
    1.  **Identify Trigger Spans**: From the dialogue, extract the 1 or 2 most crucial sentences spoken by the 'Seeker' that directly trigger or most strongly express the given emotion.
    2.  **Summarize and Categorize the Global Cause**:
        a. Write a one-sentence summary that explains the overall reason for the emotion (e.g., "The subject is happy because they reconnected with a close friend.").
        b. Classify this summary into ONE of the following psychologically-grounded categories, which represent common antecedent events for emotions: {', '.join(cause_taxonomy)}.
    
    You MUST strictly adhere to the following JSON output format. Do not add any text before or after the JSON object.
    
    ### JSON Schema:
    ```json
    {{
      "trigger_spans": [
        "string" // An array containing the 1 or 2 most crucial sentences spoken by the 'Seeker' that trigger the emotion.
      ],
      "global_cause_summary": "string", // A one-sentence summary explaining the overall reason for the emotion.
      "cause_category": "string" // This MUST be one of the following exact strings: {json.dumps(cause_taxonomy)}.
    }}```
    """

    user_prompt = f"""Please perform the dual-granularity cause analysis for the following case:

    ### Dialogue:
    {conversation_text}
    
    ### Identified Emotion:
    {emotion_label}
    
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
    index, row, api_keys, id_to_emotion_map = args_tuple
    api_key = api_keys[index % len(api_keys)]
    try:
        predicted_emotion_id = row['predicted_id']
        if predicted_emotion_id < 0:
            return {**row, 'trigger_spans': None, 'global_cause_summary': 'PREDICTION_INVALID',
                    'cause_category': 'PREDICTION_INVALID'}

        emotion_label = id_to_emotion_map.get(predicted_emotion_id, "unknown")
        conversation_text = row['utterances']

        client = openai.OpenAI(
            api_key=api_key,
            base_url="",  #input your own url here
            default_query={"api-version": "preview"},
            timeout=20.0
        )
        analysis_result = analyze_emotion_cause_with_theory(client, conversation_text, emotion_label)
        print(analysis_result.get('trigger_spans'))

        return {
            **row,
            'trigger_spans': analysis_result.get('trigger_spans'),
            'global_cause_summary': analysis_result.get('global_cause_summary'),
            'cause_category': analysis_result.get('cause_category')
        }
    except Exception as e:
        logger.error(f"Critical error in process_row for row {index}: {e}")
        return {**row, 'trigger_spans': None, 'global_cause_summary': f"ERROR: {e}", 'cause_category': 'ERROR'}

api = '' # input your own api here

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Agent 2 for dual-granularity emotion cause analysis.')
    parser.add_argument('--input_csv_path', type=str, default='./output/.csv')  # your need to input the path of the output file of agent1 here
    parser.add_argument('--mapping_path', type=str, default='./dataset/emotion_to_id_ekman.json')
    parser.add_argument('--output_path', type=str, default='./output')
    parser.add_argument('--api_keys', type=str, default=api)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    api_keys = [k.strip() for k in args.api_keys.split(',') if k.strip()]
    if not api_keys:
        raise ValueError("API keys are required. Provide them via the --api_keys argument.")

    logger.info(f"Using {args.workers} threads with {len(api_keys)} API keys.")

    try:
        df = pd.read_csv(args.input_csv_path)
        with open(args.mapping_path, 'r') as f:
            emotion_to_id_map = json.load(f)
        id_to_emotion_map = {int(v): k for k, v in emotion_to_id_map.items()}
    except FileNotFoundError as e:
        logger.error(f"Could not find a necessary file: {e}")
        exit()

    tasks = [(index, row, api_keys, id_to_emotion_map) for index, row in df.iterrows()]
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(process_row, tasks), total=len(tasks), desc="Analyzing Emotion Causes"):
            if result:
                results.append(result)

    time_now = time.strftime("%Y%m%d-%H%M%S")
    final_output_path = f'{args.output_path}/cause_analysis_output_{time_now}.csv'
    final_df = pd.DataFrame(results)
    final_df.to_csv(final_output_path, index=False)

    logger.info(f"Complete analysis saved to: {final_output_path}")
    print("\n--- Analysis Results Preview ---")
    print(final_df[['utterances', 'predicted_id', 'trigger_spans', 'global_cause_summary', 'cause_category']].head())
    logger.info("All done!")
