import json

def extract_completions(json_file_path, output_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    completions = [conversion["completion"] for conversion in data]

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for completion in completions:
            output_file.write(completion + '\n')

extract_completions('corpus/wikipedia-cn-20230720-filtered.json', 'corpus/wikipedia.txt')

