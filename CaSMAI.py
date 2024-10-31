import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio
import ollama

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline

# File Information -----------------------------------------------------------------------------------------------------
# Calculations and Systems Management Artifical Intelligence - Wessley Dennis
# Supplementary Material by Zachary Hasley

# Llama Models
# medllama2:latest                      a53737ec0c72    3.8 GB    4 minutes ago
# ALIENTELLIGENCE/shakespeare:latest    c4c7ee6a87a6    4.7 GB    29 minutes ago
# codellama:latest                      8fdf8f752f6e    3.8 GB    7 days ago
# llama3.1:8b                           42182419e950    4.7 GB    9 days ago
# llama3.2:latest                       a80c4f17acd5    2.0 GB    10 days ago
# dolphin-llama3:8b-256k                9f4257eb39a8    4.7 GB    2 months ago

# Diffusers
# "C:\\Users\\weden\\PycharmProjects\\ISRAH_Projects_01\\other_models\\v1-5-pruned-emaonly.safetensors" (Local)
# "nota-ai/bk-sdm-small" (Local Hidden)
# "C:\\Users\\weden\\PycharmProjects\\ISRAH_Projects_01\\other_models\\waiANINSFWPONYXL_v90.safetensors" (Local)

# Primary Classes for Use ----------------------------------------------------------------------------------------------

class CoreInputOutput:

    def __init__(self, model=None, attached_cores=None, default_intro="Given this list of descriptions: ", defualt_prompt= "Which description best relates to this question, respond with just the number of the description."):

        self.model = model
        self.attached_cores = attached_cores
        self.default_prompt = defualt_prompt
        self.defualt_intro =default_intro

    def audio_input_module(self):

        print("Code 01B Interaction allowed. Please say a Query.")

        # Configure Speech_Text
        speech_text_model = Model(r"C:\Users\weden\PycharmProjects\ISRAH_Projects_01\vosk-model-small-en-us-0.15")
        recognizer = KaldiRecognizer(speech_text_model, 16000)

        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        stream.start_stream()

        # Array to catch streamed text inputs
        promptarray = []
        caught_phrase = []

        while len(promptarray) <= 20:
            data = stream.read(4096)
            if recognizer.AcceptWaveform(data):
                text = recognizer.Result()

                caught_phrase = text[14:-3]
                # print(caught_phrase)
                promptarray.append(text[14:-3])

            if "hey michael" in caught_phrase:
                break

        return caught_phrase

    def text_input_module(self):
        return input("Code 01A interaction allowed...Please Input a query: ")

    def universal_input_module(self, interaction_level = 0, diagnostics_mode=False, diagnostics_array=None):

        if diagnostics_mode and diagnostics_array is not None:
            results = self.input_diagnostics_module(diagnostics_array)
            print("Results: ")
            print(results)
        else:
            if interaction_level == 1:
                input_text = self.audio_input_module()
                self.central_interperatation_module(input_text)
            else:
                input_text = self.text_input_module()
                self.central_interperatation_module(input_text)

    def central_interperatation_module(self, general_query_as_text):

        prompt = general_query_as_text
        prompt =self.defualt_intro + self.core_distiller() + " " + self.default_prompt + " " + prompt

        print(prompt)

        # print(prompt)

        response = ollama.chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])

        print(response['message']['content'])

        # If core number is invalid return core 0 or attempt to rerun once if still invalid return core zero
        # if system returns text, return core 0

    def universal_output_module(self, output, output_type):

        if output_type in ["Image", "image", "Picture", "picture", "Illustration", "illustration"]:
            output.save("output_image.png")
        else:
            print(output)

    def core_distiller(self):
        # the core distiller takes the descriptions from each core, and attempts to parse them into a single prompt
        # expecting the core list to be a list of SystemCores

        core_descriptions = "Description 0 describes general inquiries, not related to any other listed description"

        for index, item in enumerate(self.attached_cores):
            core_descriptions = core_descriptions + "Description " + str(index + 1) + " " + item.core_description + ", "

        return core_descriptions

    def phaneron_basic(self):
        '''This module serves as a sort of dumb memory for understanding short term content about the interaction'''

    def input_diagnostics_module(self, list_dictionary_queries):

        correct = 0
        number_queries = len(list_dictionary_queries)


        for index, item in enumerate(list_dictionary_queries):
            prompt = item["query"]
            prompt = self.defualt_intro + self.core_distiller() + " " + self.default_prompt + " " + prompt

            # print(prompt)

            # print(prompt)

            response = ollama.chat(model=self.model, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])

            answer = response['message']['content']
            print(f"Answer to Query# {index}")
            print(answer)
            if answer == item["goal"]:
                correct += 1

        return [correct, number_queries]


class SystemCore:
    def __init__(self, core_description=None, core_input_preference=None, core_output_preference=None, core_model=None, ollama_core=True):
        self.core_description = core_description
        self.core_input_prefrence = core_input_preference
        self.core_output_preference = core_output_preference
        self.core_model = core_model
        self.ollama_core = ollama_core

    def prompt_llama_core(self, prompt):
        response = ollama.chat(model=self.core_model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])

        return response['message']['content']

    def prompt_core(self, prompt):

        if self.ollama_core:
            text_result = self.prompt_llama_core(prompt)
            return text_result

        else:
            return "Failed to generate response from non ollama core. Currently non text based ollama cores have not " \
                   "been implemented. Did you forget to set ollama_core to True? "

class DiffuseCore(SystemCore):
    '''
    :parameter: pipeline must include method - for example if using StableDiffusion from pretrained, you must include
    it, i.e. pipeline=StableDiffusionPipeline.from_single_file
    '''
    def __init__(self, core_description=None, core_input_preference=None, core_output_preference=None, core_model=None, ollama_core=True, pipeline=StableDiffusionPipeline.from_single_file):
        super().__init__(core_description=core_description, core_input_preference=core_input_preference, core_output_preference=core_output_preference, core_model=core_model, ollama_core=ollama_core)
        self.pipeline = pipeline

    def prompt_core(self, prompt):

        if self.ollama_core:
            text_result = self.prompt_llama_core(prompt)
            return text_result

        else:
            pipe = self.pipeline(
                self.core_model,
                use_safetensors=True,
                torch_dtype=torch.float32
            )

            pipe = pipe.to("cuda")
            pipe.safety_checker = None
            image = pipe(prompt, height=512, width=512).images[0]
            return image
            # image.save("output_image.png")




# Core Initialization


Code_Core = SystemCore(core_description=" related to generating Code, Functions, Classes and Methods in various Programming Languges, as well as troubleshooting complex Code errors and problems",
                       core_input_preference="Text",
                       core_output_preference="Text",
                       core_model="codellama",
                       ollama_core=True)


Diffuse_Core = DiffuseCore(core_description=" for creating artwork, making drawings, producing images, and illustrating concepts. This is for illustrating concepts and making pictures only,",
                          core_input_preference="Text",
                          core_output_preference="Image",
                          ollama_core=False,
                          core_model="C:\\Users\\weden\\PycharmProjects\\ISRAH_Projects_01\\other_models\\v1-5-pruned-emaonly.safetensors",
                          pipeline=StableDiffusionPipeline.from_single_file
                          )

Shakespear_Core = SystemCore(core_description=" related to generating witty responses, romantic literature, poems, and anything theatre or drama related",
                       core_input_preference="Text",
                       core_output_preference="Text",
                       core_model="ALIENTELLIGENCE/shakespeare:latest",
                       ollama_core=True)

Medical_Core = SystemCore(core_description=" related to medicine, medical diagnosis, figuring out illnesses or medical conditions based on a set of symptoms given",
                       core_input_preference="Text",
                       core_output_preference="Text",
                       core_model="Amedllama2:latest",
                       ollama_core=True)


AttachedCores=[Code_Core, Diffuse_Core, Shakespear_Core, Medical_Core]

CaSMAI = CoreInputOutput(model="llama3.2", attached_cores=AttachedCores)

test_cases = [
    {"query": "Patient is feeling sick, exhibiting symptoms of diarrhea, fever and sore throat. Do you have a medical diagnosis?", "goal": "4"},
    {"query": "I want a witty poem that tells of a romance between two star crossed lovers.", "goal": "3"},
    {"query": "How would I write a class in python for dynamic linked lists?", "goal": "1"},
    {"query": "Could you draw me a picture of the mona lisa?", "goal": "2"},
    {"query": "Can you write me a function in C for a list that dynamically allocates memory based on the amount of items stored in it?", "goal": "1"},
    {"query": "Tell me a funny theater related joke.", "goal": "3"},
    {"query": "How are you today?", "goal": "0"},
    {"query": "Why is the sky blue?", "goal": "0"},
    {"query": "Can you show me what an illustration of an apple looks like?", "goal": "2"},
    {"query": "I have a patient who is showing symptoms of migraines, fever, sore throat and rash. Do you know what's wrong with them?", "goal": "4"}
]

CaSMAI.universal_input_module(diagnostics_mode=True, diagnostics_array=test_cases)

