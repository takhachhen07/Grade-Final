import pandas as pd
import numpy as np

def run_apriori_mining(df, min_support=0.1, min_confidence=0.4):
    """
    Apriori Algorithm implementation for Association Rule Mining.
    
    Transforms student metrics into categorical transaction items:
    - Gender (Male/Female)
    - Attendance (High_Attendance / Low_Attendance)
    - Study Hours (High_Study_Hours / Low_Study_Hours)
    - Internal Marks (High_Internal_Marks / Low_Internal_Marks)
    - Previous Grade (Grade_A / Grade_B / Grade_C / Grade_D_F)
    - Result (Outcome_Pass / Outcome_Fail)
    """
    if df is None or df.empty:
        return {'rules': [], 'frequent_itemsets': [], 'stats': {}}

    # Step 1: Discretize dataset into categorical item transactions
    transactions = []
    for idx, row in df.iterrows():
        items = set()
        
        # Gender
        if 'Gender' in row and pd.notna(row['Gender']):
            items.add(f"Gender_{row['Gender']}")
            
        # Attendance
        att = row.get('Attendance', 0)
        if att >= 80:
            items.add("High_Attendance (≥80%)")
        else:
            items.add("Low_Attendance (<80%)")

        # Study Hours
        sh = row.get('Study_Hours', 0)
        if sh >= 10:
            items.add("High_Study_Hours (≥10h)")
        else:
            items.add("Low_Study_Hours (<10h)")

        # Internal Marks
        im = row.get('Internal_Marks', 0)
        if im >= 30:
            items.add("High_Internal_Marks (≥30/50)")
        else:
            items.add("Low_Internal_Marks (<30/50)")

        # Previous Grade
        pg = str(row.get('Previous_Grade', 'B')).upper()
        if pg in ['A', 'B']:
            items.add("High_Previous_Grade (A/B)")
        else:
            items.add("Low_Previous_Grade (C/D/F)")

        # Result
        res = str(row.get('Result', 'Pass')).capitalize()
        items.add(f"Outcome_{res}")

        transactions.append(items)

    N = len(transactions)
    if N == 0:
        return {'rules': [], 'frequent_itemsets': [], 'stats': {}}

    # Step 2: Compute Frequent 1-Itemsets
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[item] = item_counts.get(item, 0) + 1

    frequent_1 = {frozenset([item]): count / N for item, count in item_counts.items() if (count / N) >= min_support}

    all_frequent_itemsets = dict(frequent_1)
    current_frequent = frequent_1

    # Step 3: Compute Frequent k-Itemsets (k = 2, 3)
    for k in range(2, 4):
        candidate_itemsets = set()
        itemsets_list = list(current_frequent.keys())
        
        for i in range(len(itemsets_list)):
            for j in range(i + 1, len(itemsets_list)):
                union_set = itemsets_list[i].union(itemsets_list[j])
                if len(union_set) == k:
                    candidate_itemsets.add(union_set)

        next_frequent = {}
        for candidate in candidate_itemsets:
            count = sum(1 for t in transactions if candidate.issubset(t))
            support = count / N
            if support >= min_support:
                next_frequent[candidate] = support

        if not next_frequent:
            break
            
        all_frequent_itemsets.update(next_frequent)
        current_frequent = next_frequent

    # Step 4: Generate Association Rules from Frequent Itemsets
    rules = []
    for itemset, itemset_support in all_frequent_itemsets.items():
        if len(itemset) < 2:
            continue
            
        # For each subset of itemset
        items = list(itemset)
        # Check all non-empty proper subsets
        for i in range(1, len(items)):
            import itertools
            for antecedent_tuple in itertools.combinations(items, i):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                
                antecedent_support = all_frequent_itemsets.get(antecedent, 0)
                if antecedent_support > 0:
                    confidence = itemset_support / antecedent_support
                    
                    if confidence >= min_confidence:
                        consequent_support = all_frequent_itemsets.get(consequent, 0)
                        if consequent_support == 0:
                            consequent_support = sum(1 for t in transactions if consequent.issubset(t)) / N
                            
                        lift = round(confidence / consequent_support, 2) if consequent_support > 0 else 1.0

                        rules.append({
                            'antecedent': list(antecedent),
                            'consequent': list(consequent),
                            'antecedent_str': " AND ".join([x.replace('_', ' ') for x in antecedent]),
                            'consequent_str': " AND ".join([x.replace('_', ' ') for x in consequent]),
                            'support': round(itemset_support * 100, 1),
                            'confidence': round(confidence * 100, 1),
                            'lift': lift
                        })

    # Sort rules by Confidence descending then Lift descending
    rules.sort(key=lambda x: (x['confidence'], x['lift']), reverse=True)

    # Convert frequent itemsets to list format for template rendering
    freq_list = [
        {
            'itemset': " + ".join([x.replace('_', ' ') for x in itemset]),
            'size': len(itemset),
            'support': round(sup * 100, 1)
        }
        for itemset, sup in all_frequent_itemsets.items()
    ]
    freq_list.sort(key=lambda x: (x['size'], x['support']), reverse=True)

    return {
        'rules': rules[:50],  # top 50
        'frequent_itemsets': freq_list[:30],
        'stats': {
            'total_transactions': N,
            'min_support_pct': round(min_support * 100, 1),
            'min_confidence_pct': round(min_confidence * 100, 1),
            'rules_found': len(rules),
            'frequent_itemsets_count': len(all_frequent_itemsets)
        }
    }
