import pandas as pd
import json
import ast

# Read the CSV file
# train_df = pd.read_csv('/home/yyhuang/SVG/Train/flow_grpo/dataset/svg/train.csv')
# train_question_df = pd.read_csv('/home/yyhuang/SVG/Train/flow_grpo/dataset/svg/train.csv')
train_df = pd.read_csv('/home/yyhuang/SVG/Train/flow_grpo/dataset/svg/test.csv')
train_question_df = pd.read_csv('/home/yyhuang/SVG/Train/flow_grpo/dataset/svg/test.csv')

# Process the question data
train_question_df = train_question_df.groupby('id').apply(lambda df: df.to_dict(orient='list'))
train_question_df = train_question_df.reset_index(name='qa')

train_question_df['question'] = train_question_df.qa.apply(lambda qa: json.dumps(qa['question'], ensure_ascii=False))

train_question_df['choices'] = train_question_df.qa.apply(
    lambda qa: json.dumps(
        [ast.literal_eval(x) for x in qa['choices']], ensure_ascii=False
    )
)

train_question_df['answer'] = train_question_df.qa.apply(lambda qa: json.dumps(qa['answer'], ensure_ascii=False))

# Merge dataframes
train_df = pd.merge(train_df[['id', 'description']], train_question_df, how='left', on='id')
train_df.drop_duplicates(subset=['id'], inplace=True)
train_df.reset_index(drop=True, inplace=True)

# Create multiple_choice_qa field
train_df['multiple_choice_qa'] = train_df.apply(
    lambda r: {
    'question': json.loads(r.question),
    'choices': json.loads(r.choices),
    'answer': json.loads(r.answer)
    },
    axis=1,
)

# Create output JSONL file
output_path = '/home/yyhuang/SVG/Train/flow_grpo/dataset/svg/test_metadata.jsonl'

# Write to JSONL format
with open(output_path, 'w', encoding='utf-8') as f:
    for _, row in train_df.iterrows():
        # Create dictionary with only the required fields
        data = {
            'id': row['id'],
            'description': row['description'],
            'multiple_choice_qa': row['multiple_choice_qa']
        }
        # Write each record as a JSON line
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Conversion complete. Output saved to {output_path}")