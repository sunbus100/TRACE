import pandas as pd
import time
import json
import logging
import openai
import concurrent.futures
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import os

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_SLEEP_TIME = 5


def get_emotion_definitions():
    ekman_emotion_map_str = {
        'joyful': 'happiness', 'excited': 'happiness', 'proud': 'happiness', 'grateful': 'happiness',
        'content': 'happiness',
        'impressed': 'happiness', 'caring': 'happiness', 'trusting': 'happiness', 'faithful': 'happiness',
        'hopeful': 'happiness', 'confident': 'happiness', 'sentimental': 'happiness', 'sad': 'sadness',
        'lonely': 'sadness', 'disappointed': 'sadness', 'devastated': 'sadness', 'nostalgic': 'sadness',
        'guilty': 'sadness', 'ashamed': 'sadness', 'embarrassed': 'sadness', 'angry': 'anger',
        'furious': 'anger', 'annoyed': 'anger', 'jealous': 'anger', 'afraid': 'fear', 'terrified': 'fear',
        'anxious': 'fear', 'apprehensive': 'fear', 'disgusted': 'disgust', 'surprised': 'surprise',
        'anticipating': 'surprise'
    }
    definitions = {ekman: [] for ekman in set(ekman_emotion_map_str.values())}
    for fine_grained, ekman in ekman_emotion_map_str.items():
        definitions[ekman].append(fine_grained)
    definition_prompt_string = ""
    for ekman_category, fine_grained_list in definitions.items():
        definition_prompt_string += f"- **{ekman_category.capitalize()}**: This broad category can represent finer emotions such as: {', '.join(fine_grained_list)}.\n"
    return definition_prompt_string


def classify_emotion_with_expert_prompt(client, conversation_text, emotion_map, definitions):
    emotion_labels = list(emotion_map.keys())
    system_prompt = f"""You are a psychology expert specializing in emotion classification. Your task is to analyze a dialogue from a controlled experiment and identify the subject's core emotion.
    
    **IMPORTANT CONTEXT:** The emotion labels in this dataset were originally 32 fine-grained emotions, which have been mapped to the six standard Ekman categories. Therefore, you must understand the nuances within each broad category.
    
    Here are the definitions for the six categories based on the original labels:
    {definitions}
    Your task is to analyze the conversation and determine which of the six main categories is the most fitting. Consider these subtle meanings. For example, a conversation expressing 'sentimental' should be classified as 'happiness', and one expressing 'annoyed' should be classified as 'anger'.
    
    Analyze the conversation's context, flow, and word choice. Then, provide your final classification in a valid JSON format with "reasoning" (a brief explanation) and "final_emotion" (one of the six main categories).
    """
    
    user_prompt = f"""Please classify the following conversation based on the expert definitions provided:
    
    "{conversation_text}"
    """
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"}, temperature=0.0, max_tokens=150
            )
            response_content = chat_completion.choices[0].message.content
            response_json = json.loads(response_content)
            predicted_emotion = response_json.get("final_emotion", "").lower()
            if predicted_emotion in emotion_map:
                return emotion_map[predicted_emotion], response_content
            else:
                logger.warning(f"LLM returned a valid JSON but with an unknown emotion: '{predicted_emotion}'")
                return -2, response_content
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            last_exception = e
            time.sleep((INITIAL_SLEEP_TIME ** attempt) + random.uniform(0, 1))
        except Exception as e:
            logger.error(f"A non-retriable error occurred: {e}")
            return -1, str(e)
    logger.error(f"Failed to process after {MAX_RETRIES} retries. Last error: {last_exception}")
    return -1, str(last_exception)


def process_row(args_tuple):

    i, row, api_keys, emotion_map, definitions = args_tuple
    api_key = api_keys[i % len(api_keys)]
    try:
        llm_client = openai.OpenAI(
            api_key=api_key,
            base_url="",  # input your own url here
            default_query={"api-version": "preview"},
            timeout=20.0
        )
        text_to_classify = row['utterances']
        predicted_id, raw_response = classify_emotion_with_expert_prompt(llm_client, text_to_classify, emotion_map,
                                                                         definitions)

        return {**row, 'predicted_id': predicted_id, 'llm_response': raw_response}
    except Exception as e:
        logger.error("Critical error in process_row for row %s: %s", i, str(e))
        return {**row, 'predicted_id': -1, 'llm_response': f"PROCESS_ROW_ERROR: {e}"}

api = '' # input your own api here

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Classify emotions using LLM APIs with expert definitions.')
    parser.add_argument('--dataset_path', type=str, default='./dataset/data_test.csv',
                        help='Path to the test CSV with clean speaker roles.')
    parser.add_argument('--mapping_path', type=str, default='./dataset/emotion_to_id_ekman.json',
                        help='Path to the emotion-to-id JSON map.')
    parser.add_argument('--output_path', type=str, default='./output', help='Directory to save results.')
    parser.add_argument('--api_keys', type=str, default=api, help='Comma-separated OpenAI API keys.')
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel worker threads.')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    api_keys = [k.strip() for k in args.api_keys.split(',') if k.strip()]
    if not api_keys:
        raise ValueError("API keys are required.")

    emotion_definitions = get_emotion_definitions()

    time_now = time.strftime("%Y%m%d-%H%M%S")
    output_csv_path = f'{args.output_path}/api_output_expert_prompt_{time_now}.csv'
    metric_path = f'{args.output_path}/api_metric_expert_prompt_{time_now}.json'

    logger.info(f"Using {args.workers} threads with {len(api_keys)} API keys.")

    df = pd.read_csv(args.dataset_path)
    df.dropna(inplace=True)
    with open(args.mapping_path, 'r') as f:
        emotion_to_id_map = json.load(f)

    tasks = [(i, row, api_keys, emotion_to_id_map, emotion_definitions) for i, row in df.iterrows()]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(process_row, tasks), total=len(tasks),
                           desc="Classifying with Expert Definitions"):
            results.append(result)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv_path, index=False)
    logger.info(f"Full results saved to {output_csv_path}")

    try:
        eval_df = out_df[out_df['predicted_id'] >= 0].copy()
        if eval_df.empty:
            logger.warning("No valid rows for evaluation.")
        else:
            y_true = eval_df['emotion_id'].values
            y_pred = eval_df['predicted_id'].values
            report = classification_report(y_true, y_pred, target_names=list(emotion_to_id_map.keys()),
                                           output_dict=True)
            report['accuracy'] = accuracy_score(y_true, y_pred)
            logger.info("--- Evaluation Report ---")
            logger.info(json.dumps(report, indent=4))
            with open(metric_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Evaluation metrics saved to {metric_path}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    logger.info("All done!")
