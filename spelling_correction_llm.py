import language_tool_python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the language tool
tool = language_tool_python.LanguageTool('en-US')

# Load pre-trained model tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Knowledge base of common spelling errors and corrections
knowledge_base = [
    {"original": "thier", "corrected": "their"},
    {"original": "becuase", "corrected": "because"},
    {"original": "definately", "corrected": "definitely"},
    {"original": "lov", "corrected": "love"},
    {"original": "ar", "corrected": "are"},
    {"original": "reed", "corrected": "read"},
    {"original": "grate", "corrected": "great"},
    {"original": "beutiful", "corrected": "beautiful"},
    {"original": "cookk", "corrected": "cook"},
]

# Create a retrieval mechanism
def retrieve_similar_examples(input_text, knowledge_base, top_n=2):
    kb_texts = [entry['original'] for entry in knowledge_base]
    vectorizer = TfidfVectorizer().fit_transform([input_text] + kb_texts)
    vectors = vectorizer.toarray()
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    similar_indices = similarities.argsort()[-top_n:][::-1]
    return [knowledge_base[idx] for idx in similar_indices]

# Function to use language_tool_python for initial correction
def initial_correction(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Define a prompt for spelling correction using RAG
prompt_template = """
You are a highly intelligent AI capable of correcting spelling errors. Use the initial corrections and similar examples as a guide and ensure the final corrected text keeps the meaning intact. Provide only the corrected version without any additional explanation.

Initial correction:
Original: "{original_text}"
Corrected: "{initial_text}"

Similar examples:
{examples}

Final correction:
"""

# Function to generate corrected text using LLM
def correct_spelling_with_rag(original, initial, examples):
    examples_text = "\n".join([f'Original: "{ex["original"]}", Corrected: "{ex["corrected"]}"' for ex in examples])
    prompt = prompt_template.format(original_text=original, initial_text=initial, examples=examples_text)
    response = generator(prompt, max_new_tokens=50, num_return_sequences=1, pad_token_id=50256, truncation=True)
    generated_text = response[0]['generated_text']
    corrected_text = generated_text.split("Final correction:")[-1].strip().split("\n")[0]

    # Verify the correction
    if len(corrected_text.split()) >= len(original.split()) and corrected_text != original:
        return corrected_text
    else:
        return initial  # Return the initial correction if the LLM correction seems inadequate

# Correct the spelling errors in the data
data = [
    "lovee", 
    "reed", 
    "boks", 
    "grate", 
    "beutiful", 
    "cookk",
    "I lov to reaad",
    "Hee is a grate painter",
    "Thiss is a beutiful place",
    "Shee is an amazing cookk",
    "They ar goood friends",
    "Plase help me",
    "Wher is the nearesst store",
    "Thee weather is nic today"
]

intermediate_data = [initial_correction(sentence) for sentence in data]
corrected_data = []
for orig, intermediate in zip(data, intermediate_data):
    similar_examples = retrieve_similar_examples(orig, knowledge_base)
    corrected = correct_spelling_with_rag(orig, intermediate, similar_examples)
    corrected_data.append(corrected)

# Print the corrected texts
for original, intermediate, corrected in zip(data, intermediate_data, corrected_data):
    print(f"Original: {original}")
    print(f"Intermediate: {intermediate}")
    print(f"Corrected: {corrected}\n")
