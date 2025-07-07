import json
import pandas as pd

# --- Configuration ---
RAW_DATA_PATH = '../data/raw/abstracts_sessions_authors_topics.json'
CLEANED_DATA_PATH = '../data/processed/cleaned_dataframe.csv'

def load_raw_data (filepath:str) -> pd.DataFrame:
    with open(filepath) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def explode_authors(df):
    # Filter out rows with empty author lists
    df = df[df['authors'].map(lambda x: len(x) > 0 if isinstance(x, list) else False)].copy()
    # Explode the lists so each author dict becomes its own row
    df = df.explode('authors').reset_index(drop=True)
    # Normalize the dicts into columns
    authors_normalized = pd.json_normalize(df['authors'])
    df = pd.concat(
        [df.drop(columns=['authors']).reset_index(drop=True),
         authors_normalized.reset_index(drop=True)],
        axis=1
    )
    return df

def drop_unused_columns(df):
    columns_to_drop = ['submission_date', 'publication_date', 'content', 'word_count',
                       'keywords', 'inserted', 'updated', 'owner_ref', 'last_editor_ref',
                       'state_key', 'deleted', 'version_number', 'transaction_number',
                       'sequence', 'external_identifiers', 'cpo__co2_id', 'cpo__sinner_id']
    df = df.drop(columns = columns_to_drop, axis=1)
    return df

def convert_object_columns_to_string(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Drop nulls and check if all remaining values are strings
            non_null = df[col].dropna()
            if non_null.map(type).eq(str).all():
                df[col] = df[col].astype('string')
    return df

def convert_language_ref_column(df):
    df.rename(columns={'language_ref': 'language'}, inplace=True)
    language_map = {
        254: 'German',
        255: 'English'
    }
    df['language'] = df['language'].map(language_map)
    df['language'] = df['language'].astype('string')
    return df

def convert_affiliationcountry_ref_column(df):
    country_code_map = {3: "Albania", 4: "Algeria", 11: "Argentina", 12: "Armenia", 14: "Australia",
                        15: "Austria", 16: "Azerbaijan", 18: "Bahrain", 19: "Bangladesh", 21: "Belarus",
                        22: "Belgium", 29: "Bosnia and Herzegovina", 32: "Brazil", 35: "Bulgaria",
                        41: "Canada", 45: "Chile", 46: "China", 49: "Colombia", 54: "Costa Rica",
                        56: "Croatia", 57: "Cuba", 59: "Cyprus", 60: "Czech Republic", 61: "Denmark",
                        64: "Dominican Republic", 65: "Ecuador", 66: "Egypt", 67: "El Salvador", 70: "Estonia",
                        75: "Finland", 76: "France", 82: "Georgia", 83: "Germany", 84: "Ghana", 86: "Greece",
                        87: "Unknown", 91: "Unknown", 100: "Hong Kong", 101: "Hungary", 102: "Iceland",
                        103: "India", 104: "Indonesia", 105: "Iran", 107: "Ireland", 109: "Israel",
                        110: "Italy", 112: "Japan", 115: "Unknown", 116: "Kenya", 119: "South Korea",
                        121: "Kyrgyzstan", 123: "Latvia", 124: "Lebanon", 129: "Lithuania", 130: "Luxembourg",
                        131: "Macau", 132: "North Macedonia", 135: "Malaysia", 137: "Mali", 138: "Malta",
                        144: "Mexico", 146: "Moldova", 149: "Montenegro", 151: "Morocco", 152: "Mozambique",
                        153: "Myanmar", 156: "Nepal", 157: "Netherlands", 159: "New Zealand", 162: "Nigeria",
                        166: "Norway", 168: "Pakistan", 170: "Palestine", 171: "Panama", 173: "Paraguay",
                        174: "Peru", 175: "Philippines", 177: "Poland", 178: "Portugal", 180: "Qatar",
                        182: "Romania", 183: "Russia", 197: "Serbia", 200: "Singapore", 202: "Unknown",
                        203: "Slovenia", 206: "South Africa", 209: "Spain", 210: "Sri Lanka", 211: "Unknown",
                        215: "Sweden", 216: "Switzerland", 218: "Taiwan", 221: "Thailand", 227: "Tunisia",
                        228: "Turkey", 232: "Uganda", 233: "Ukraine", 234: "United Arab Emirates",
                        235: "United Kingdom", 236: "United States", 239: "Uzbekistan", 241: "Venezuela"
                        }
    df = df.rename(columns={'affiliationcountry_ref': 'affiliationcountry'})
    df['affiliationcountry'] = (
        df['affiliationcountry'].map(country_code_map).astype('string')
    )
    # All missing values in "affiliationcountry" happen to be Kosovo
    df["affiliationcountry"] = df["affiliationcountry"].fillna("Kosovo")
    return df

def drop_duplicate_rows(df):
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def fix_missing_values_content_raw(df):
    df["content_raw"] = df["affiliationcountry"].fillna("")
    return df


#TODO Main