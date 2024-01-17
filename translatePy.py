import json
import deepl

def translate_text(text, translator):
    try:
        result = translator.translate_text(text, target_lang="DE")
        return result.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return text

def translate_json(input_file, output_file, translator):
    with open(input_file, 'r') as file:
        data = json.load(file)

    translated_data = {key: translate_text(value, translator) for key, value in data.items()}

    with open(output_file, 'w') as file:
        json.dump(translated_data, file, indent=4)

# DeepL API setup
auth_key = "3693706c-8eea-0c28-5b8f-5e9b94aa4b7b:fx"  # Replace with your DeepL Auth Key
translator = deepl.Translator(auth_key)

# Example usage
input_filename = 'bird_map_Cap.json'
output_filename = 'bird_map_Ger.json'
translate_json(input_filename, output_filename, translator)
