def reformat_bounds(min_data, max_data):
    # Créer un dictionnaire formaté avec min et max pour chaque colonne
    formatted_bounds = {}
    
    for col in min_data:
        if col in max_data:
            formatted_bounds[col] = {'min': min_data[col], 'max': max_data[col]}
        else:
            print(f"Warning: Missing max value for column {col}")
    
    return formatted_bounds


def reformat_one_hot(qualitatives_min):
    formatted_data = {}
    
    for col, values in qualitatives_min.items():
        # Reformater chaque valeur en retirant le préfixe
        formatted_values = [value.split('_')[1] for value in values]
        formatted_data[col] = formatted_values
        
    return formatted_data
