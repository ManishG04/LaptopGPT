import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple

# Constants for clustering
N_CLUSTERS = 8  # Number of laptop clusters
FEATURE_WEIGHTS = {
    'RAM (in GB)': 0.15,
    'Screen Size (in inch)': 0.10,
    'Weight (in kg)': 0.10,
    'Performance_Score': 0.25,
    'Portability': 0.20,
    'Value_Score': 0.20
}

def prepare_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare clusters using K-means clustering."""
    # Select and normalize features for clustering
    features = list(FEATURE_WEIGHTS.keys())
    scaler = MinMaxScaler()
    normalized_features = pd.DataFrame(
        scaler.fit_transform(df[features]),
        columns=features,
        index=df.index
    )
    
    # Apply feature weights
    for feature, weight in FEATURE_WEIGHTS.items():
        normalized_features[feature] *= weight
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    df['Cluster'] = kmeans.fit_predict(normalized_features)
    
    print(f"\nCluster distribution:")
    for cluster in range(N_CLUSTERS):
        cluster_size = len(df[df['Cluster'] == cluster])
        print(f"Cluster {cluster}: {cluster_size} laptops")
    
    return df

def find_recommendations(best_match: pd.Series, candidates: pd.DataFrame, n: int) -> List[Dict]:
    """Find similar laptops using cluster-based KNN approach."""
    if len(candidates) <= 1:
        return []
    
    best_cluster = best_match['Cluster']
    print(f"\nBest match from cluster {best_cluster}")
    
    # First, get recommendations from the same cluster
    same_cluster = candidates[candidates['Cluster'] == best_cluster].copy()
    
    # Then, find nearest neighbors within the cluster
    features = list(FEATURE_WEIGHTS.keys())
    scaler = MinMaxScaler()
    
    # Normalize features for the cluster
    cluster_features = pd.DataFrame(
        scaler.fit_transform(same_cluster[features]),
        columns=features,
        index=same_cluster.index
    )
    
    # Apply weights
    for feature, weight in FEATURE_WEIGHTS.items():
        cluster_features[feature] *= weight
    
    # Find nearest neighbors within cluster
    n_neighbors = min(len(same_cluster), n + 1)
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn_model.fit(cluster_features)
    
    # Get recommendations from same cluster
    distances, indices = nn_model.kneighbors(
        cluster_features.loc[best_match.name].values.reshape(1, -1)
    )
    
    recommendations = []
    seen_laptops = set()
    
    # Add recommendations from same cluster
    for distance, idx in zip(distances[0], indices[0]):
        if idx == best_match.name or idx in seen_laptops:
            continue
        
        laptop = same_cluster.iloc[idx]
        similarity = 100 * np.exp(-distance)
        laptop_info = format_laptop_info(laptop)
        laptop_info['similarity_score'] = round(similarity, 1)
        laptop_info['cluster'] = int(laptop['Cluster'])
        recommendations.append(laptop_info)
        seen_laptops.add(idx)
    
    # If we need more recommendations, get from other clusters
    if len(recommendations) < n:
        other_clusters = candidates[candidates['Cluster'] != best_cluster]
        for _, laptop in other_clusters.iterrows():
            if len(recommendations) >= n:
                break
            laptop_info = format_laptop_info(laptop)
            laptop_info['similarity_score'] = 50  # Default similarity for other clusters
            laptop_info['cluster'] = int(laptop['Cluster'])
            recommendations.append(laptop_info)
    
    print(f"\nFound {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"- {rec['name']} (Cluster {rec['cluster']}, Similarity: {rec['similarity_score']})")
    
    return recommendations

# Initialize clusters when loading the module
df = pd.read_csv('data/CleanedLaptopData.csv')
df = prepare_clusters(df)



PERFORMANCE_RANGES = {'BASIC': (15, 30), 'PRODUCTIVITY': (30, 45), 'CREATIVE': (45, 60), 'GAMING': (60, 100)}
PORTABILITY_RANGES = {'DESKTOP_REPLACEMENT': (0, 30), 'ALL_PURPOSE': (30, 65), 'ULTRAPORTABLE': (65, 100)}
VALUE_RANGES = {'BUDGET': (0, 35), 'MID_RANGE': (35, 50), 'HIGH_END': (50, 100)}
N_CLUSTERS = 10
WEIGHTS = {
    'RAM (in GB)': 0.15, 'Screen Size (in inch)': 0.05, 'Weight (in kg)': 0.1,
    'Normalized_CPU_Ranking': 0.15, 'Normalized_GPU_Benchmark': 0.15, 'Portability': 0.1,
    'Performance_Score': 0.15, 'Value_Score': 0.1, 'user rating': 0.05
}

scaler = MinMaxScaler()
feature_columns = WEIGHTS.keys()
normalized_features = pd.DataFrame(
    scaler.fit_transform(df[feature_columns]),
    columns=feature_columns, index=df.index
)
normalized_features *= list(WEIGHTS.values())

df['Cluster'] = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(normalized_features)
nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean').fit(normalized_features.values)

def validate_input(input_json: Dict) -> bool:
    """Check required fields and validate ranges."""
    required_fields = ['specifications', 'price_range', 'performance_range', 'portability_range']
    for field in required_fields:
        if field not in input_json:
            raise ValueError(f"Missing required field: {field}")
    for key, (min_val, max_val) in {
        'price_range': (15990, 301990),
        'performance_range': (0, 100),
        'portability_range': (0, 100)
    }.items():
        if not (min_val <= input_json[key]['min'] <= input_json[key]['max'] <= max_val):
            raise ValueError(f"Invalid values in {key}")
    return True


def filter_laptops(preferences: Dict) -> pd.DataFrame:
    """Filter laptops based on price, performance, portability, and specifications."""
    filtered = df.copy()
    
    # Debug initial count
    print(f"\nDEBUG: Starting with {len(filtered)} laptops")
    
    # Store original laptops before filtering for fallback
    original_filtered = filtered.copy()
    
    # Price filter
    filtered = filtered[
        (filtered['Price (in Indian Rupees)'] >= preferences['price_range']['min']) &
        (filtered['Price (in Indian Rupees)'] <= preferences['price_range']['max'])
    ]
    print(f"After price filter ({preferences['price_range']['min']}-{preferences['price_range']['max']}): {len(filtered)} laptops")
    
    # Performance filter
    filtered = filtered[
        (filtered['Performance_Score'] >= preferences['performance_range']['min']) &
        (filtered['Performance_Score'] <= preferences['performance_range']['max'])
    ]
    print(f"After performance filter ({preferences['performance_range']['min']}-{preferences['performance_range']['max']}): {len(filtered)} laptops")
    
    # Portability filter
    filtered = filtered[
        (filtered['Portability'] >= preferences['portability_range']['min']) &
        (filtered['Portability'] <= preferences['portability_range']['max'])
    ]
    print(f"After portability filter ({preferences['portability_range']['min']}-{preferences['portability_range']['max']}): {len(filtered)} laptops")
    
    specs = preferences['specifications']
    
    # RAM filter (allow exact or higher)
    if 'RAM (in GB)' in specs:
        filtered = filtered[filtered['RAM (in GB)'] >= specs['RAM (in GB)']]
        print(f"After RAM filter (>= {specs['RAM (in GB)']}GB): {len(filtered)} laptops")
        print(f"Available RAM sizes: {sorted(filtered['RAM (in GB)'].unique())}")
    
    # Storage filter (allow exact or higher)
    if 'Storage' in specs:
        min_storage = float(specs['Storage'])
        # Extract numeric values from storage strings
        filtered['Storage_Numeric'] = filtered['Storage'].astype(str).str.extract(r'(\d+)').astype(float)
        filtered = filtered[filtered['Storage_Numeric'] >= min_storage]
        filtered = filtered.drop('Storage_Numeric', axis=1)  # Clean up temporary column
        print(f"After storage filter (>= {min_storage}GB): {len(filtered)} laptops")
        print(f"Available storage sizes: {sorted(filtered['Storage'].unique())}")
    
    # Screen size filter with wider tolerance (±1 inch for larger screens)
    if 'Screen Size (in inch)' in specs:
        target_size = float(specs['Screen Size (in inch)'])
        tolerance = 1.0 if target_size >= 17 else 0.5  # Wider tolerance for larger screens
        filtered = filtered[
            (filtered['Screen Size (in inch)'] >= target_size - tolerance) &
            (filtered['Screen Size (in inch)'] <= target_size + tolerance)
        ]
        print(f"After screen size filter ({target_size}\" ± {tolerance}\"): {len(filtered)} laptops")
        print(f"Available screen sizes: {sorted(filtered['Screen Size (in inch)'].unique())}")
    
    # If no laptops match all criteria, relax screen size constraint
    if filtered.empty and 'Screen Size (in inch)' in specs:
        print("\nNo exact matches found. Relaxing screen size constraint...")
        filtered = original_filtered[
            (original_filtered['Price (in Indian Rupees)'] >= preferences['price_range']['min']) &
            (original_filtered['Price (in Indian Rupees)'] <= preferences['price_range']['max']) &
            (original_filtered['Performance_Score'] >= preferences['performance_range']['min']) &
            (original_filtered['Performance_Score'] <= preferences['performance_range']['max']) &
            (original_filtered['Portability'] >= preferences['portability_range']['min']) &
            (original_filtered['Portability'] <= preferences['portability_range']['max']) &
            (original_filtered['RAM (in GB)'] >= specs.get('RAM (in GB)', 0))
        ]
        
        if 'Storage' in specs:
            filtered['Storage_Numeric'] = filtered['Storage'].astype(str).str.extract(r'(\d+)').astype(float)
            filtered = filtered[filtered['Storage_Numeric'] >= float(specs['Storage'])]
            filtered = filtered.drop('Storage_Numeric', axis=1)
    
    # Print detailed information about remaining laptops
    if not filtered.empty:
        print("\nMatching laptops:")
        for _, laptop in filtered.iterrows():
            print(
                f"- {laptop['name']}: "
                f"RAM: {laptop['RAM (in GB)']}GB, "
                f"Storage: {laptop['Storage']}, "
                f"Performance: {laptop['Performance_Score']:.1f}"
            )
    else:
        print("\nNo laptops found matching the criteria.")
    
    return filtered


def format_laptop_info(laptop: pd.Series) -> Dict:
    """Format laptop details for output."""
    try:
        return {
            'name': str(laptop.get('name', 'Unknown')),
            'price': int(laptop.get('Price (in Indian Rupees)', 0)),
            'specifications': {
                'RAM': float(laptop.get('RAM (in GB)', 0)),
                'Storage': str(laptop.get('Storage', '0')),
                'Screen Size': float(laptop.get('Screen Size (in inch)', 0)),
                'Weight': float(laptop.get('Weight (in kg)', 0))
            },
            'scores': {
                'performance': round(float(laptop.get('Performance_Score', 0)), 2),
                'portability': round(float(laptop.get('Portability', 0)), 2),
                'value': round(float(laptop.get('Value_Score', 0)), 2)
            },
            'similarity_score': 0,
            'cluster': int(laptop.get('Cluster', 0))
        }
    except Exception as e:
        print(f"Error formatting laptop info: {e}")
        # Return a safe default structure
        return {
            'name': 'Unknown',
            'price': 0,
            'specifications': {
                'RAM': 0,
                'Storage': '0',
                'Screen Size': 0,
                'Weight': 0
            },
            'scores': {
                'performance': 0,
                'portability': 0,
                'value': 0
            },
            'similarity_score': 0,
            'cluster': 0
        }

def get_recommendations(input_json: Dict, num_recommendations: int = 5) -> Dict:
    """Get laptop recommendations based on user preferences."""
    try:
        validate_input(input_json)
        filtered_laptops = filter_laptops(input_json)
        
        if filtered_laptops.empty:
            return {
                'status': 'error',
                'message': 'No matches found',
                'best_match': None,
                'similar_recommendations': []
            }

        best_match = filtered_laptops.nlargest(1, ['user rating', 'Performance_Score']).iloc[0]
        similar_recommendations = find_recommendations(best_match, filtered_laptops, num_recommendations - 1)
        
        return {
            'status': 'success',
            'message': f'Found {len(filtered_laptops)} matching laptops',
            'best_match': format_laptop_info(best_match),
            'similar_recommendations': similar_recommendations
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'best_match': None,
            'similar_recommendations': []
        }


# Test input
# input_preferences = {
#     "specifications": {
#         "RAM (in GB)": 8,
#         "Storage": "512"
#     },
#     "price_range": {
#         "min": 200000,
#         "max": 301990
#     },
#     "performance_range": {
#         "min": 75,  
#         "max": 100
#     },
#     "portability_range": {
#         "min": 0,  
#         "max": 40
#     }
# }

# # Get recommendations
# recommendations = get_recommendations(input_preferences)
# print(recommendations)