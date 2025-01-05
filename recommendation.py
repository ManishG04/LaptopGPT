import pandas as pd
from typing import Dict, List
import json

df = pd.read_csv('data/CleanedLaptopData.csv')

def filter_laptops(preferences: Dict) -> Dict:
    """Filter laptops based on preferences and return JSON formatted results."""
    try:
        filtered_df = df.copy()
        print(f"\nDEBUG: Starting with {len(filtered_df)} laptops")
        
        # 1. Essential Filters First (Price, RAM, Storage)
        if 'price_range' in preferences:
            filtered_df = filtered_df[
                (filtered_df['Price (in Indian Rupees)'] >= preferences['price_range']['min']) &
                (filtered_df['Price (in Indian Rupees)'] <= preferences['price_range']['max'])
            ]
            print(f"After price filter: {len(filtered_df)} laptops")

        specs = preferences.get('specifications', {})
        
        # RAM Filter
        if 'RAM (in GB)' in specs:
            min_ram = float(specs['RAM (in GB)'])
            filtered_df = filtered_df[filtered_df['RAM (in GB)'] >= min_ram]
            print(f"After RAM filter: {len(filtered_df)} laptops")

        # Storage Filter
        if 'Storage' in specs:
            min_storage = float(specs['Storage'])
            filtered_df = filtered_df[filtered_df['Storage'] >= min_storage]
            print(f"After storage filter: {len(filtered_df)} laptops")

        # 2. Check if we have enough results after essential filters
        if len(filtered_df) < 20:
            print("Warning: Very few results after essential filters. Relaxing constraints...")
            return filter_laptops_with_relaxed_constraints(preferences)

        # 3. Apply Performance and Portability filters with wider ranges
        if 'performance_range' in preferences:
            min_perf = max(0, preferences['performance_range']['min'] - 10)  # Relax by 10 points
            max_perf = min(100, preferences['performance_range']['max'] + 10)
            filtered_df = filtered_df[
                (filtered_df['Performance_Score'] >= min_perf) &
                (filtered_df['Performance_Score'] <= max_perf)
            ]
            print(f"After performance filter: {len(filtered_df)} laptops")

        if 'portability_range' in preferences:
            min_port = max(0, preferences['portability_range']['min'] - 10)  # Relax by 10 points
            max_port = min(100, preferences['portability_range']['max'] + 10)
            filtered_df = filtered_df[
                (filtered_df['Portability'] >= min_port) &
                (filtered_df['Portability'] <= max_port)
            ]
            print(f"After portability filter: {len(filtered_df)} laptops")

        # 4. Optional Filters (Processor, GPU)
        if 'processor_min' in specs and len(filtered_df) > 10:  # Only apply if we have enough results
            processor_name = specs['processor_min'].lower()
            if 'i' in processor_name or 'ryzen' in processor_name:
                if 'i3' in processor_name:
                    filtered_df = filtered_df[filtered_df['Processor name'].str.contains('i[3-9]', case=False, regex=True)]
                elif 'i5' in processor_name:
                    filtered_df = filtered_df[filtered_df['Processor name'].str.contains('i[5-9]', case=False, regex=True)]
                elif 'i7' in processor_name:
                    filtered_df = filtered_df[filtered_df['Processor name'].str.contains('i[7-9]', case=False, regex=True)]
                elif 'i9' in processor_name:
                    filtered_df = filtered_df[filtered_df['Processor name'].str.contains('i9', case=False)]
                elif 'ryzen' in processor_name:
                    filtered_df = filtered_df[filtered_df['Processor name'].str.contains('ryzen', case=False)]
            print(f"After processor filter: {len(filtered_df)} laptops")

        if specs.get('dedicated_graphics') and len(filtered_df) > 10:
            filtered_df = filtered_df[filtered_df['Dedicated Graphic Memory Capacity'] > 0]
            print(f"After GPU filter: {len(filtered_df)} laptops")

        # Rest of the function remains the same...
        return format_results(filtered_df)

    except Exception as e:
        print(f"Error in filter_laptops: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "filtered_laptops": []
        }

def filter_laptops_with_relaxed_constraints(preferences: Dict) -> Dict:
    """Apply filters with relaxed constraints when initial filtering is too restrictive."""
    relaxed_preferences = preferences.copy()
    
    # Relax price range by 20%
    if 'price_range' in relaxed_preferences:
        price_range = relaxed_preferences['price_range']
        range_width = price_range['max'] - price_range['min']
        price_range['min'] = max(15990, price_range['min'] - (range_width * 0.2))
        price_range['max'] = min(301990, price_range['max'] + (range_width * 0.2))

    # Relax performance and portability ranges
    if 'performance_range' in relaxed_preferences:
        perf_range = relaxed_preferences['performance_range']
        perf_range['min'] = max(0, perf_range['min'] - 20)
        perf_range['max'] = min(100, perf_range['max'] + 20)

    if 'portability_range' in relaxed_preferences:
        port_range = relaxed_preferences['portability_range']
        port_range['min'] = max(0, port_range['min'] - 20)
        port_range['max'] = min(100, port_range['max'] + 20)

    # Remove processor requirement if present
    if 'specifications' in relaxed_preferences:
        specs = relaxed_preferences['specifications']
        if 'processor_min' in specs:
            del specs['processor_min']

    print("Applying relaxed constraints...")
    return filter_laptops(relaxed_preferences)

def format_results(filtered_df: pd.DataFrame) -> Dict:
    """Format the filtered results into the required JSON structure."""
    results = {
        "status": "success",
        "total_matches": len(filtered_df),
        "filtered_laptops": []
    }
    
    seen_configs = set()
    
    for _, laptop in filtered_df.iterrows():
        config_signature = (
            laptop['Processor name'].lower(),
            laptop['RAM (in GB)'],
            laptop['Storage'],
            laptop['gpu name '].strip().lower() if not pd.isna(laptop['gpu name ']) else "integrated",
            laptop['Screen Size (in inch)']
        )
        
        if config_signature in seen_configs:
            continue
            
        seen_configs.add(config_signature)
        
        laptop_dict = {
            "name": laptop['name'],
            "price": int(laptop['Price (in Indian Rupees)']),
            "specifications": {
                "processor": laptop['Processor name'],
                "ram": f"{int(laptop['RAM (in GB)'])}GB",
                "storage": f"{int(laptop['Storage'])}GB",
                "gpu": laptop['gpu name '].strip() if not pd.isna(laptop['gpu name ']) else "Integrated Graphics",
                "screen_size": f"{float(laptop['Screen Size (in inch)']):.1f}\"",
                "weight": f"{float(laptop['Weight (in kg)']):.2f} kg",
                "battery": f"{float(laptop['battery_backup']):.1f} hours"
            },
            "scores": {
                "performance": float(laptop['Performance_Score']),
                "portability": float(laptop['Portability']),
                "value": float(laptop['Value_Score'])
            }
        }
        
        results["filtered_laptops"].append(laptop_dict)
        
        if len(results["filtered_laptops"]) >= 10:
            break
    
    return results

# Example test case
# if __name__ == "__main__":
#     # Example preferences with performance and portability ranges
#     test_preferences = {
#         "specifications": {
#             "RAM (in GB)": 16,
#             "Storage": 512,
#             "dedicated_graphics": True
#             # processor_min is now optional
#         },
#         "price_range": {
#             "min": 120000,
#             "max": 240000
#         },
#         "performance_range": {
#             "min": 0,
#             "max": 100
#         },
#         "portability_range": {
#             "min": 0,
#             "max": 100
#         }
#     }
    
#     # Get recommendations
#     results = filter_laptops(test_preferences)
    
#     # Print results
#     if results["status"] == "success":
#         print(f"\nFound {results['total_matches']} total matches")
#         print(f"Showing top {len(results['filtered_laptops'])} unique laptops:\n")
        
#         for i, laptop in enumerate(results['filtered_laptops'], 1):
#             print(f"{i}. {laptop['name']}")
#             print(f"   Price: ₹{laptop['price']:,}")
#             print(f"   Specs: {laptop['specifications']['processor']}, "
#                   f"{laptop['specifications']['ram']}, "
#                   f"{laptop['specifications']['storage']}, "
#                   f"{laptop['specifications']['gpu']}")
#             print(f"   Performance Score: {laptop['scores']['performance']:.1f}")
#             print(f"   Portability Score: {laptop['scores']['portability']:.1f}")
#             print(f"   Value Score: {laptop['scores']['value']:.1f}")
#             print()
#     else:
#         print(f"Error: {results['message']}")
