from typing import Dict, List

def reformat_bounds(min_data: Dict[str, float], max_data: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Reformats the given minimum and maximum data dictionaries into a structured format.

    Args:
        min_data (Dict[str, float]): A dictionary containing minimum values for each column.
        max_data (Dict[str, float]): A dictionary containing maximum values for each column.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where each key is a column name, and the value is another dictionary 
                                     with 'min' and 'max' keys representing the respective bounds.
    """
    formatted_bounds: Dict[str, Dict[str, float]] = {}

    for col in min_data:
        if col in max_data:
            formatted_bounds[col] = {'min': min_data[col], 'max': max_data[col]}
        else:
            print(f"Warning: Missing max value for column {col}")

    return formatted_bounds


def reformat_one_hot(qualitatives_min: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Reformats one-hot encoded qualitative data by extracting categorical values.

    Args:
        qualitatives_min (Dict[str, List[str]]): A dictionary where keys are column names and values 
                                                 are lists of one-hot encoded strings.

    Returns:
        Dict[str, List[str]]: A dictionary where each key is a column name, and the value is a list 
                              of extracted categorical values (after splitting on '_').
    """
    formatted_data: Dict[str, List[str]] = {}

    for col, values in qualitatives_min.items():
        formatted_values = [value.split('_')[1] for value in values]
        formatted_data[col] = formatted_values

    return formatted_data
