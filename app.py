"""import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Study-Buddy"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...") """


"""import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from googletrans import Translator  # Import the Translator class from googletrans library

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Study-Buddy"
translator = Translator()  # Create an instance of the Translator class

def translate_text(text, target_language='en'):
    # Use the translate method to translate the text to the target language
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

def get_chatbot_response(user_input):
    # Your existing chatbot logic remains unchanged
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Translate the response back to the user's language
                return translate_text(random.choice(intent['responses']), target_language='en')  # Assuming English is the target language
    else:
        return translate_text("I do not understand...", target_language='en')  # Assuming English is the target language

print("Let's chat! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input == "quit":
        break

    # Detect the language of the user's input
    user_language = translator.detect(user_input).lang

    # Translate user input to English for processing
    translated_input = translate_text(user_input, target_language='en')

    # Get the chatbot response and translate it back to the user's language
    chatbot_response = get_chatbot_response(translated_input)
    translated_response = translate_text(chatbot_response, target_language=user_language)

    print(f"{bot_name}: {translated_response}")"""
'''before GUI'''
'''import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from googletrans import Translator  # Import the Translator class from googletrans library
import speech_recognition as sr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Study-Buddy"
translator = Translator()  # Create an instance of the Translator class
recognizer = sr.Recognizer()

def translate_text(text, target_language='en'):
    # Use the translate method to translate the text to the target language
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

def get_chatbot_response(user_input):
    # Your existing chatbot logic remains unchanged
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Translate the response back to the user's language
                return translate_text(random.choice(intent['responses']), target_language='en')  # Assuming English is the target language
    else:
        return translate_text("I do not understand...", target_language='en')  # Assuming English is the target language

def get_text_input():
    return input("You: ")

def get_voice_input():
    print("Listening... Speak now.")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    print("Processing...")

    try:
        user_input = recognizer.recognize_google(audio)
        print("Voice Input:", user_input)
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I could not understand your voice.")
        return ""
    except sr.RequestError as e:
        print(f"Error with the voice recognition service; {e}")
        return ""

print("Let's chat! (type 'quit' to exit)")
while True:
    print("Choose input method:")
    print("1. Text")
    print("2. Voice")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        user_input = get_text_input()
    elif choice == '2':
        user_input = get_voice_input()
    else:
        print("Invalid choice. Please enter 1 or 2.")
        continue

    if user_input.lower() == "quit":
        break

    # Detect the language of the user's input
    user_language = translator.detect(user_input).lang

    # Translate user input to English for processing
    translated_input = translate_text(user_input, target_language='en')

    # Get the chatbot response and translate it back to the user's language
    chatbot_response = get_chatbot_response(translated_input)
    translated_response = translate_text(chatbot_response, target_language=user_language)

    print(f"{bot_name}: {translated_response}")'''



'''

import tkinter as tk
from tkinter import ttk
import threading
import queue
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from googletrans import Translator
import speech_recognition as sr

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot GUI")

        self.initialize_chatbot()

        self.create_widgets()

    def initialize_chatbot(self):
        with open('intents.json', 'r') as json_data:
            self.intents = json.load(json_data)

        FILE = "data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.bot_name = "Study-Buddy"
        self.translator = Translator()
        self.recognizer = sr.Recognizer()
        self.listening = False
        self.queue = queue.Queue()

    def create_widgets(self):
        # Increase the width and height of the Text widget
        self.output_text = tk.Text(self.root, wrap="word", width=60, height=15, state="disabled")
        self.output_text.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Increase the width of the Entry widget
        self.input_entry = tk.Entry(self.root, width=60)
        self.input_entry.grid(row=1, column=0, padx=10, pady=10)

        # Increase the width of the Button widgets
        self.send_button = tk.Button(self.root, text="Send", command=self.send_message, width=20)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.voice_button = tk.Button(self.root, text="Hold to Speak", command=self.toggle_voice_input, width=40)
        self.voice_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Bind mouse press and release events to the toggle_voice_input function
        self.voice_button.bind("<ButtonPress-1>", lambda event: self.toggle_voice_input(event, start=True))
        self.voice_button.bind("<ButtonRelease-1>", lambda event: self.toggle_voice_input(event, start=False))

        self.root.after(100, self.check_queue)

    def send_message(self):
        user_input = self.input_entry.get()
        self.input_entry.delete(0, "end")

        if user_input.lower() == "quit":
            self.root.destroy()

        user_language = self.translator.detect(user_input).lang
        translated_input = self.translate_text(user_input, target_language='en')

        chatbot_response = self.get_chatbot_response(translated_input)
        user_language = self.translator.detect(user_input).lang
        translated_response = self.translate_text(chatbot_response, target_language=user_language)

        self.display_message(f"You: {user_input}")
        self.display_message(f"{self.bot_name}: {translated_response}")

    def toggle_voice_input(self, event=None, start=True):
        if start:
            self.voice_button.config(text="Listening...", state=tk.DISABLED)
            self.listening = True
            threading.Thread(target=self.process_voice_input).start()
        else:
            self.voice_button.config(text="Hold to Speak", state=tk.NORMAL)
            self.listening = False

    def process_voice_input(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

        try:
            user_input = self.recognizer.recognize_google(audio)
            translated_input = self.translate_text(user_input, target_language='en')

            chatbot_response = self.get_chatbot_response(translated_input)
            user_language = self.translator.detect(user_input).lang
            translated_response = self.translate_text(chatbot_response, target_language=user_language)

            self.display_message(f"Voice Input: {user_input}")
            self.display_message(f"{self.bot_name}: {translated_response}")

        except sr.UnknownValueError:
            self.display_message("Sorry, I could not understand your voice.")
        except sr.RequestError as e:
            self.display_message(f"Error with the voice recognition service; {e}")        # existing code ...

    def display_message(self, message):
        self.output_text.config(state="normal")
        self.output_text.insert("end", f"{message}\n")
        self.output_text.see("end")
        self.output_text.config(state="disabled")

    def translate_text(self, text, target_language='en'):
        translated_text = self.translator.translate(text, dest=target_language).text
        return translated_text

    def get_chatbot_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return "I do not understand..."

    def check_queue(self):
        try:
            message = self.queue.get_nowait()
            self.display_message(message)
        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    root.mainloop()
'''

'''imp
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from googletrans import Translator
import speech_recognition as sr
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot GUI")

        self.initialize_chatbot()
        self.create_widgets()

    def initialize_chatbot(self):
        with open('intents.json', 'r') as json_data:
            self.intents = json.load(json_data)

        FILE = "data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.bot_name = "Study-Buddy"
        self.translator = Translator()
        self.recognizer = sr.Recognizer()
        self.listening = False
        self.queue = queue.Queue()

        # Summarizer and question-answering pipelines
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

        # Load the model and tokenizer explicitly for text generation
        self.text_generator_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.text_generator_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.text_generator = pipeline("text-generation", model=self.text_generator_model, tokenizer=self.text_generator_tokenizer, config={"max_new_tokens": 50, "truncation": True})

    def create_widgets(self):
        self.output_text = tk.Text(self.root, wrap="word", width=60, height=15, state="disabled")
        self.output_text.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.input_entry = tk.Entry(self.root, width=60)
        self.input_entry.grid(row=1, column=0, padx=10, pady=10)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message, width=20)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.voice_button = tk.Button(self.root, text="Hold to Speak", command=self.toggle_voice_input, width=40)
        self.voice_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.voice_button.bind("<ButtonPress-1>", lambda event: self.toggle_voice_input(event, start=True))
        self.voice_button.bind("<ButtonRelease-1>", lambda event: self.toggle_voice_input(event, start=False))

        self.upload_button = tk.Button(self.root, text="Upload PDF", command=self.upload_pdf, width=40)
        self.upload_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.quiz_button = tk.Button(self.root, text="Generate Quiz", command=self.generate_quiz, width=40)
        self.quiz_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.root.after(100, self.check_queue)

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.summarize_pdf(file_path)
            self.extract_text_for_quiz(file_path)

    def summarize_pdf(self, file_path):
        text = self.extract_text_from_pdf(file_path)
        if text:
            summary = self.summarizer(text, max_length=150, min_length=30, do_sample=False)
            self.display_message(f"PDF Summary: {summary[0]['summary_text']}")

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    def extract_text_for_quiz(self, file_path):
        self.pdf_text_for_quiz = self.extract_text_from_pdf(file_path)

    def generate_quiz(self):
        if hasattr(self, 'pdf_text_for_quiz'):
            questions = self.create_questions_from_text(self.pdf_text_for_quiz)
            self.display_message("Generated Quiz:")
            for i, q in enumerate(questions):
                self.display_message(f"Q{i+1}: {q['question'].strip()}\nA: {q['answer'].strip()}")
        else:
            self.display_message("Please upload a PDF first.")
    
    def create_questions_from_text(self, text):
        chunks = text.split('. ')
        questions = []
        for chunk in chunks:
            context = chunk.strip()
            if len(context) < 10:
                continue  # Skip if the context is too short

        # Generate the question
            prompt = f"Based on the following text, create a concise question: {context}"
            generated_text = self.text_generator(prompt,
                                             max_new_tokens=30,  # Limit to 30 new tokens
                                             num_return_sequences=1,
                                             truncation=True)[0]['generated_text']

        # Post-process generated text to remove the prompt and keep only the question
            question = generated_text.replace(prompt, '').strip()
            if '?' not in question:
                question += '?'  # Ensure the question ends with a question mark

        # Get the answer from the context using a QA model
            answer = self.qa_pipeline({'question': question, 'context': context})['answer']
        
        # Append the question-answer pair to the list
            questions.append({'question': question, 'answer': answer})
        
        # Add a limit to the number of questions generated
            if len(questions) >= 5:  # Limit to 5 questions for better organization
                break

        return questions


    def send_message(self):
        user_input = self.input_entry.get()
        self.input_entry.delete(0, "end")

        if user_input.lower() == "quit":
            self.root.destroy()

        user_language = self.translator.detect(user_input).lang
        translated_input = self.translate_text(user_input, target_language='en')

        chatbot_response = self.get_chatbot_response(translated_input)
        translated_response = self.translate_text(chatbot_response, target_language=user_language)

        self.display_message(f"You: {user_input}")
        self.display_message(f"{self.bot_name}: {translated_response}")

    def toggle_voice_input(self, event=None, start=True):
        if start:
            self.voice_button.config(text="Listening...", state=tk.DISABLED)
            self.listening = True
            threading.Thread(target=self.process_voice_input).start()
        else:
            self.voice_button.config(text="Hold to Speak", state=tk.NORMAL)
            self.listening = False

    def process_voice_input(self):
        with sr.Microphone() as source:
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    voice_text = self.recognizer.recognize_google(audio)
                    self.queue.put(f"Voice Input: {voice_text}")
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.queue.put("Sorry, I did not understand that.")
                except sr.RequestError as e:
                    self.queue.put(f"Could not request results; {e}")

    def get_chatbot_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"])

        return "I do not understand..."

    def translate_text(self, text, target_language):
        return self.translator.translate(text, dest=target_language).text

    def display_message(self, message):
        self.output_text.config(state="normal")
        self.output_text.insert("end", message + "\n")
        self.output_text.config(state="disabled")
        self.output_text.see("end")

    def check_queue(self):
        try:
            message = self.queue.get_nowait()
            self.display_message(message)
        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    root.mainloop()'''
'''

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from deep_translator import GoogleTranslator
import speech_recognition as sr
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot GUI")

        self.initialize_chatbot()
        self.create_widgets()

    def initialize_chatbot(self):
        with open('intents.json', 'r') as json_data:
            self.intents = json.load(json_data)

        FILE = "data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.bot_name = "Study-Buddy"
        self.translator = GoogleTranslator()  # Replaced googletrans with deep-translator
        self.recognizer = sr.Recognizer()
        self.listening = False
        self.queue = queue.Queue()

        # Summarizer and question-answering pipelines
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

        # Load the model and tokenizer explicitly for text generation
        self.text_generator_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.text_generator_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.text_generator = pipeline("text-generation", model=self.text_generator_model, tokenizer=self.text_generator_tokenizer, config={"max_new_tokens": 50, "truncation": True})

    def create_widgets(self):
        self.output_text = tk.Text(self.root, wrap="word", width=60, height=15, state="disabled")
        self.output_text.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.input_entry = tk.Entry(self.root, width=60)
        self.input_entry.grid(row=1, column=0, padx=10, pady=10)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message, width=20)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.voice_button = tk.Button(self.root, text="Hold to Speak", command=self.toggle_voice_input, width=40)
        self.voice_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.voice_button.bind("<ButtonPress-1>", lambda event: self.toggle_voice_input(event, start=True))
        self.voice_button.bind("<ButtonRelease-1>", lambda event: self.toggle_voice_input(event, start=False))

        self.upload_button = tk.Button(self.root, text="Upload PDF", command=self.upload_pdf, width=40)
        self.upload_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.quiz_button = tk.Button(self.root, text="Generate Quiz", command=self.generate_quiz, width=40)
        self.quiz_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.root.after(100, self.check_queue)

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.summarize_pdf(file_path)
            self.extract_text_for_quiz(file_path)

    def summarize_pdf(self, file_path):
        text = self.extract_text_from_pdf(file_path)
        if text:
            summary = self.summarizer(text, max_length=150, min_length=30, do_sample=False)
            self.display_message(f"PDF Summary: {summary[0]['summary_text']}")

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    def extract_text_for_quiz(self, file_path):
        self.pdf_text_for_quiz = self.extract_text_from_pdf(file_path)

    def generate_quiz(self):
        if hasattr(self, 'pdf_text_for_quiz'):
            questions = self.create_questions_from_text(self.pdf_text_for_quiz)
            self.display_message("Generated Quiz:")
            for i, q in enumerate(questions):
                self.display_message(f"Q{i+1}: {q['question'].strip()}\nA: {q['answer'].strip()}")
        else:
            self.display_message("Please upload a PDF first.")
    
    def create_questions_from_text(self, text):
        chunks = text.split('. ')
        questions = []
        for chunk in chunks:
            context = chunk.strip()
            if len(context) < 10:
                continue  # Skip if the context is too short

            # Generate the question
            prompt = f"Based on the following text, create a concise question: {context}"
            generated_text = self.text_generator(prompt,
                                                 max_new_tokens=30,  # Limit to 30 new tokens
                                                 num_return_sequences=1,
                                                 truncation=True)[0]['generated_text']

            # Post-process generated text to remove the prompt and keep only the question
            question = generated_text.replace(prompt, '').strip()
            if '?' not in question:
                question += '?'  # Ensure the question ends with a question mark

            # Get the answer from the context using a QA model
            answer = self.qa_pipeline({'question': question, 'context': context})['answer']
        
            # Append the question-answer pair to the list
            questions.append({'question': question, 'answer': answer})
        
            # Add a limit to the number of questions generated
            if len(questions) >= 5:  # Limit to 5 questions for better organization
                break

        return questions

    def send_message(self):
        user_input = self.input_entry.get()
        self.input_entry.delete(0, "end")

        if user_input.lower() == "quit":
            self.root.destroy()

        user_language = self.translator.detect(user_input)
        translated_input = self.translator.translate(user_input, target="en")

        chatbot_response = self.get_chatbot_response(translated_input)
        translated_response = self.translator.translate(chatbot_response, target=user_language)

        self.display_message(f"You: {user_input}")
        self.display_message(f"{self.bot_name}: {translated_response}")

    def toggle_voice_input(self, event=None, start=True):
        if start:
            self.voice_button.config(text="Listening...", state=tk.DISABLED)
            self.listening = True
            threading.Thread(target=self.process_voice_input).start()
        else:
            self.voice_button.config(text="Hold to Speak", state=tk.NORMAL)
            self.listening = False

    def process_voice_input(self):
        with sr.Microphone() as source:
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    voice_text = self.recognizer.recognize_google(audio)
                    self.queue.put(f"Voice Input: {voice_text}")
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.queue.put("Sorry, I did not understand that.")
                except sr.RequestError as e:
                    self.queue.put(f"Could not request results; {e}")

    def get_chatbot_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"])

        return "I do not understand..."

    def display_message(self, message):
        self.output_text.config(state="normal")
        self.output_text.insert("end", message + "\n")
        self.output_text.config(state="disabled")
        self.output_text.see("end")

    def check_queue(self):
        try:
            message = self.queue.get_nowait()
            self.display_message(message)
        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    root.mainloop()

'''
'''
from flask import Flask, render_template, request, jsonify
import torch
import json
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from googletrans import Translator
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

def initialize_chatbot():
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Study-Buddy"
    translator = Translator()

    # Summarizer and question-answering pipelines
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Load the model and tokenizer explicitly for text generation
    text_generator_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    text_generator_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    text_generator = pipeline("text-generation", model=text_generator_model, tokenizer=text_generator_tokenizer, config={"max_new_tokens": 50, "truncation": True})

    return model, intents, all_words, tags, bot_name, translator, summarizer, qa_pipeline, text_generator

model, intents, all_words, tags, bot_name, translator, summarizer, qa_pipeline, text_generator = initialize_chatbot()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    translated_input = translator.translate(user_input, dest='en').text
    sentence = tokenize(translated_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                translated_response = translator.translate(response, dest=translator.detect(user_input).lang).text
                return jsonify({"response": translated_response})
    else:
        return jsonify({"response": "I do not understand..."})

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    file = request.files['file']
    file_path = "./uploads/" + file.filename
    file.save(file_path)
    text = extract_text_from_pdf(file_path)
    summary = summarize_pdf(text)
    return jsonify({"summary": summary})

def summarize_pdf(text):
    if text:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    return ""

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    text = request.json['text']
    questions = create_questions_from_text(text)
    return jsonify({"questions": questions})

def create_questions_from_text(text):
    chunks = text.split('. ')
    questions = []
    for chunk in chunks:
        context = chunk.strip()
        if len(context) < 10:
            continue  # Skip if the context is too short

        # Generate the question
        prompt = f"Based on the following text, create a concise question: {context}"
        generated_text = text_generator(prompt,
                                         max_new_tokens=30,  # Limit to 30 new tokens
                                         num_return_sequences=1,
                                         truncation=True)[0]['generated_text']

        # Post-process generated text to remove the prompt and keep only the question
        question = generated_text.replace(prompt, '').strip()
        if '?' not in question:
            question += '?'  # Ensure the question ends with a question mark

        # Get the answer from the context using a QA model
        answer = qa_pipeline({'question': question, 'context': context})['answer']

        # Append the question-answer pair to the list
        questions.append({'question': question, 'answer': answer})

        # Add a limit to the number of questions generated
        if len(questions) >= 5:  # Limit to 5 questions for better organization
            break

    return questions

if __name__ == "__main__":
    app.run(debug=True)
'''
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import random
import json
import torch
import sentencepiece
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from googletrans import Translator
import speech_recognition as sr
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import nltk
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot GUI")

        self.initialize_chatbot()
        self.create_widgets()

    def initialize_chatbot(self):
        with open('intents.json', 'r') as json_data:
            self.intents = json.load(json_data)

        FILE = "data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        self.all_words = data['all_words']
        self.tags = data['tags']
        model_state = data["model_state"]

        self.model = NeuralNet(input_size, hidden_size, output_size)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.bot_name = "Study-Buddy"
        self.translator = Translator()
        self.recognizer = sr.Recognizer()
        self.listening = False
        self.queue = queue.Queue()

        # Summarizer and question-answering pipelines
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        self.qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qa-qg-hl")
        self.qg_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qa-qg-hl")

        # Load the model and tokenizer explicitly for text generation
        self.text_generator_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.text_generator_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.text_generator = pipeline("text-generation", model=self.text_generator_model, tokenizer=self.text_generator_tokenizer, config={"max_new_tokens": 50, "truncation": True})

    def create_widgets(self):
        self.output_text = tk.Text(self.root, wrap="word", width=60, height=15, state="disabled")
        self.output_text.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.input_entry = tk.Entry(self.root, width=60)
        self.input_entry.grid(row=1, column=0, padx=10, pady=10)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message, width=20)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.voice_button = tk.Button(self.root, text="Hold to Speak", command=self.toggle_voice_input, width=40)
        self.voice_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.voice_button.bind("<ButtonPress-1>", lambda event: self.toggle_voice_input(event, start=True))
        self.voice_button.bind("<ButtonRelease-1>", lambda event: self.toggle_voice_input(event, start=False))

        self.upload_button = tk.Button(self.root, text="Upload PDF", command=self.upload_pdf, width=40)
        self.upload_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.quiz_button = tk.Button(self.root, text="Generate Quiz", command=self.generate_quiz, width=40)
        self.quiz_button.grid(row=4, column=0, columnspan=2, pady=10)

        self.root.after(100, self.check_queue)

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.summarize_pdf(file_path)
            self.extract_text_for_quiz(file_path)

    def summarize_pdf(self, file_path):
        text = self.extract_text_from_pdf(file_path)
        if text:
            summary = self.summarizer(text, max_length=150, min_length=30, do_sample=False)
            self.display_message(f"PDF Summary: {summary[0]['summary_text']}")

    def extract_text_from_pdf(self, file_path):
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text
    '''
    def extract_text_for_quiz(self, file_path):
        self.pdf_text_for_quiz = self.extract_text_from_pdf(file_path)'''
    def extract_text_for_quiz(self, file_path):
        self.pdf_text_for_quiz = self.extract_text_from_pdf(file_path)
        self.pdf_text_paragraphs = self.pdf_text_for_quiz.split('\n\n')  # Split based on paragraphs
    '''
    def generate_quiz(self):
        if hasattr(self, 'pdf_text_for_quiz'):
            questions = self.create_questions_from_text(self.pdf_text_for_quiz)
            self.display_message("Generated Quiz:")
            for i, q in enumerate(questions):
                self.display_message(f"Q{i+1}: {q['question'].strip()}\nA: {q['answer'].strip()}")
        else:
            self.display_message("Please upload a PDF first.")'''
    
    def generate_quiz(self):
        if hasattr(self, 'pdf_text_paragraphs'):
            questions = self.create_questions_from_text(self.pdf_text_for_quiz)
            self.display_message("Generated Quiz:")
            for i, q in enumerate(questions):
                self.display_message(f"Q{i+1}: {q['question'].strip()}")
            # Display context for review
                self.display_message(f"Context: {q['context'].strip()}")
        else:
            self.display_message("Please upload a PDF first.")

    
    
    
    
    '''
    def create_questions_from_text(self, text):
        chunks = text.split('. ')
        questions = []
        for chunk in chunks:
            context = chunk.strip()
            if len(context) < 10:
                continue  # Skip if the context is too short

        # Generate the question
            prompt = f"Based on the following text, create a concise question: {context}"
            generated_text = self.text_generator(prompt,
                                             max_new_tokens=30,  # Limit to 30 new tokens
                                             num_return_sequences=1,
                                             truncation=True)[0]['generated_text']

        # Post-process generated text to remove the prompt and keep only the question
            question = generated_text.replace(prompt, '').strip()
            if '?' not in question:
                question += '?'  # Ensure the question ends with a question mark

        # Get the answer from the context using a QA model
            answer = self.qa_pipeline({'question': question, 'context': context})['answer']
        
        # Append the question-answer pair to the list
            questions.append({'question': question, 'answer': answer})
        
        # Add a limit to the number of questions generated
            if len(questions) >= 5:  # Limit to 5 questions for better organization
                break

        return questions'''
    
    
    def create_questions_from_text(self, text):
        questions = []
        for paragraph in self.pdf_text_paragraphs:
            if len(paragraph.strip()) < 30:  # Skip short paragraphs
                continue

        # Format input as expected by the model
            input_text = f"generate question: {paragraph}"
            input_ids = self.qg_tokenizer.encode(input_text, return_tensors='pt')

        # Generate question
            output = self.qg_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
            question = self.qg_tokenizer.decode(output[0], skip_special_tokens=True)

        # Append the question to the list
            questions.append({'question': question, 'context': paragraph})

            if len(questions) >= 5:  # Limit to 5 questions
                break

        return questions


    def send_message(self, user_input=None):
        if user_input is None:
            user_input = self.input_entry.get()
            self.input_entry.delete(0, "end")

        if user_input.lower() == "quit":
            self.root.destroy()

        user_language = self.translator.detect(user_input).lang
        translated_input = self.translate_text(user_input, target_language='en')

        chatbot_response = self.get_chatbot_response(translated_input)
        translated_response = self.translate_text(chatbot_response, target_language=user_language)

        self.display_message(f"You: {user_input}")
        self.display_message(f"{self.bot_name}: {translated_response}")

    '''
    def toggle_voice_input(self, event=None, start=True):
        if start:
            self.voice_button.config(text="Listening...", state=tk.DISABLED)
            self.listening = True
            threading.Thread(target=self.process_voice_input).start()
        else:
            self.voice_button.config(text="Hold to Speak", state=tk.NORMAL)
            self.listening = False'''

    def toggle_voice_input(self, event=None, start=True):
        if start:
            print("Starting voice input...")
            self.voice_button.config(text="Listening...", state=tk.DISABLED)
            self.listening = True
            threading.Thread(target=self.process_voice_input, daemon=True).start()  # Daemon thread
        else:
            print("Stopping voice input...")
            self.voice_button.config(text="Hold to Speak", state=tk.NORMAL)
            self.listening = False
        
    '''
    def process_voice_input(self):
        with sr.Microphone() as source:
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    voice_text = self.recognizer.recognize_google(audio)
                    self.queue.put(f"Voice Input: {voice_text}")
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.queue.put("Sorry, I did not understand that.")
                except sr.RequestError as e:
                    self.queue.put(f"Could not request results; {e}")'''
    def process_voice_input(self):
        with sr.Microphone() as source:
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    voice_text = self.recognizer.recognize_google(audio)
                
                # Call send_message with voice_text
                    self.send_message(voice_text)

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.queue.put("Sorry, I did not understand that.")
                except sr.RequestError as e:
                    self.queue.put(f"Could not request results; {e}")


    def get_chatbot_response(self, user_input):
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = self.model(X)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent["responses"])

        return "I do not understand..."

    def translate_text(self, text, target_language):
        return self.translator.translate(text, dest=target_language).text

    def display_message(self, message):
        self.output_text.config(state="normal")
        self.output_text.insert("end", message + "\n")
        self.output_text.config(state="disabled")
        self.output_text.see("end")

    def check_queue(self):
        try:
            message = self.queue.get_nowait()
            self.display_message(message)
        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    root.mainloop()








