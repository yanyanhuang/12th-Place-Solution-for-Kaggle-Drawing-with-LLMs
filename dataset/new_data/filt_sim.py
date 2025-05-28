import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from tqdm import tqdm

def load_data(jsonl_path):
    """加载JSONL数据"""
    print(f"Loading data from {jsonl_path}...")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    return df

def basic_deduplication(df):
    """基础去重：删除完全相同的description"""
    print("Performing basic deduplication...")
    original_len = len(df)
    
    # 获取唯一的description
    unique_descriptions = df['description'].drop_duplicates()
    print(f"Unique descriptions: {len(unique_descriptions)} (from {df['description'].nunique()} unique values)")
    
    # 保留第一次出现的记录
    df_dedup = df.drop_duplicates(subset=['description'], keep='first')
    
    print(f"After basic deduplication: {len(df_dedup)} rows (removed {original_len - len(df_dedup)} rows)")
    return df_dedup

def semantic_deduplication(df, similarity_threshold=0.9, batch_size=100):
    """语义去重：删除语义相似的description"""
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    descriptions = df['description'].unique().tolist()
    print(f"Processing {len(descriptions)} unique descriptions...")
    
    # 计算embedding
    print("Computing embeddings...")
    embeddings = model.encode(descriptions, batch_size=batch_size, show_progress_bar=True)
    
    # 计算余弦相似度矩阵
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # 找到需要删除的description索引
    remove_indices = set()
    removed_pairs = []
    
    print(f"Finding similar pairs with threshold {similarity_threshold}...")
    for i in tqdm(range(len(descriptions))):
        if i in remove_indices:
            continue
            
        # 找到与当前description相似的其他description
        for j in range(i + 1, len(descriptions)):
            if j in remove_indices:
                continue
                
            if similarity_matrix[i][j] >= similarity_threshold:
                removed_pairs.append((descriptions[i], descriptions[j], similarity_matrix[i][j]))
                # 将j标记为要删除（保留i，删除j）
                remove_indices.add(j)
    
    # 创建保留的description列表（不在删除列表中的）
    keep_descriptions = [descriptions[i] for i in range(len(descriptions)) if i not in remove_indices]
    
    print(f"Found {len(removed_pairs)} similar pairs")
    print(f"Keeping {len(keep_descriptions)} out of {len(descriptions)} descriptions")
    print(f"Removed {len(descriptions) - len(keep_descriptions)} descriptions")
    
    # 过滤数据
    df_filtered = df[df['description'].isin(keep_descriptions)].copy()
    
    return df_filtered, removed_pairs

def save_results(df_filtered, removed_pairs, output_path, removed_pairs_path):
    """保存结果"""
    print(f"Saving filtered data to {output_path}...")
    
    # 保存为JSONL格式
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df_filtered.iterrows():
            json.dump(row.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    
    if removed_pairs:
        print(f"Saving removed pairs to {removed_pairs_path}...")
        removed_df = pd.DataFrame(removed_pairs, 
                                columns=['description_1', 'description_2', 'similarity_score'])
        removed_df.to_csv(removed_pairs_path, index=False)
    
    print(f"Final data shape: {df_filtered.shape}")

def main():
    parser = argparse.ArgumentParser(description='Deduplicate JSONL data based on description similarity')
    parser.add_argument('--input', default='/home/yyhuang/SVG/Train/flow_grpo/dataset/new_data/train_metadata.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('--output', default='/home/yyhuang/SVG/Train/flow_grpo/dataset/new_data/train_filtered.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--removed', default='/home/yyhuang/SVG/Train/flow_grpo/dataset/new_data/removed_pairs.csv',
                       help='File to save removed similar pairs')
    parser.add_argument('--threshold', type=float, default=0.9,
                       help='Similarity threshold for semantic deduplication (default: 0.9)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for embedding computation (default: 100)')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_data(args.input)
    
    # 基础去重
    df_dedup = basic_deduplication(df)
    
    # 语义去重
    df_filtered, removed_pairs = semantic_deduplication(df_dedup, 
                                                       similarity_threshold=args.threshold,
                                                       batch_size=args.batch_size)
    
    # 保存结果
    save_results(df_filtered, removed_pairs, args.output, args.removed)
    
    print("\nSummary:")
    print(f"Original rows: {len(df)}")
    print(f"After basic deduplication: {len(df_dedup)}")
    print(f"After semantic deduplication: {len(df_filtered)}")
    print(f"Total removed: {len(df) - len(df_filtered)}")
    print(f"Removal rate: {(len(df) - len(df_filtered)) / len(df) * 100:.2f}%")

if __name__ == "__main__":
    main()
