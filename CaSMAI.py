import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio
import ollama

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline

from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

import pandas as pd

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

    def universal_input_module(self, interaction_level = 0, diagnostics_mode=False, diagnostics_array=None, debug_mode=False, llmcic=True, core_select_only=False, maxtries=2):

        if diagnostics_mode and diagnostics_array is not None:
            if llmcic:
                results = self.llm_input_diagnostics_module(diagnostics_array)
                print("Results: ")
                print(results)
            else:
                results = self.ssm_input_diagnostics_module(diagnostics_array)
                print("Results: ")
                print(results)
        else:
            if llmcic:
                if interaction_level == 1:
                    input_text = self.audio_input_module()
                    core_index = self.llm_central_interperatation_module(input_text, debug_mode=debug_mode, maxtries=maxtries)
                    if core_index == -1:
                        result = self.llm_central_interperatation_module(input_text, debug_mode=False, use_as_core=True)
                        self.universal_output_module(result, "Text")
                    else:
                        if core_select_only:
                            print(core_index-1)
                        else:
                            result = self.attached_cores[core_index-1].prompt_core(input_text)
                            self.universal_output_module(result, self.attached_cores[core_index-1].core_output_preference)
                else:
                    input_text = self.text_input_module()
                    core_index = self.llm_central_interperatation_module(input_text, debug_mode=debug_mode, maxtries=maxtries)
                    if core_index == -1:
                        result = self.llm_central_interperatation_module(input_text, debug_mode=False, use_as_core=True)
                        self.universal_output_module(result, "Text")
                    else:
                        if core_select_only:
                            print(core_index-1)
                        else:
                            result = self.attached_cores[core_index-1].prompt_core(input_text)
                            self.universal_output_module(result, self.attached_cores[core_index-1].core_output_preference)
            else:
                if interaction_level == 1:
                    input_text = self.audio_input_module()
                    core_index = self.ssm_central_interperatation_module(input_text, debug_mode=debug_mode)
                    if core_select_only:
                        print(core_index)
                    else:
                        result = self.attached_cores[core_index].prompt_core(input_text)
                        self.universal_output_module(result, self.attached_cores[core_index].core_output_preference)
                else:
                    input_text = self.text_input_module()
                    core_index = self.ssm_central_interperatation_module(input_text, debug_mode=debug_mode)
                    if core_select_only:
                        print(core_index)
                    else:
                        result = self.attached_cores[core_index].prompt_core(input_text)
                        self.universal_output_module(result, self.attached_cores[core_index].core_output_preference)

    def llm_central_interperatation_module(self, general_query_as_text, debug_mode=False, maxtries=1, use_as_core=False):

        prompt = general_query_as_text
        prompt =self.defualt_intro + self.core_distiller() + " " + self.default_prompt + " " + prompt

        # print(prompt)

        chance = 0
        if use_as_core:
            response = ollama.chat(model=self.model, messages=[
                {
                    'role': 'user',
                    'content': general_query_as_text,
                },
            ])

            print("\n")
            return response['message']['content']
        else:
            for i in range(maxtries):

                response = ollama.chat(model=self.model, messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ])

                if debug_mode:
                    print(prompt)
                    print(response['message']['content'])

                # Implement looping queries to eliminate hallucinated results

                try:
                    core_index = int(response['message']['content'][0])
                    return core_index
                except ValueError as e:
                    if chance >= maxtries:
                        print("input recieved from CIC: ")
                        print(response['message']['content'])
                        print(e)
                        print("Defaulting to Core 0...")
                        return -1
                    else:
                        chance += 1
            # If core number is invalid return core 0 or attempt to rerun once if still invalid return core zero
            # if system returns text, return core 0

    def ssm_central_interperatation_module(self, general_query_as_text, debug_mode=False):

        # this module uses semantic similarity to determine core

        # sentences from which to choose from
        sentences = self.core_distiller(singlestring=False)

        # test query
        test_query = general_query_as_text

        #  display inputs for debugging
        if debug_mode:
            print('Input Query: ', test_query)
            for index, item in enumerate(sentences):
                print(str(index) + " " + item)
            print("--- Debug Mode ------------------------------------------------------------------------------------")

        # vectorize input
        test_vec = model.encode([test_query])[0]

        sim_matrix = []

        # aquire similarity scores (the higher the better)
        for sent in sentences:
            similarity_score = 1 - distance.cosine(test_vec, model.encode([sent])[0])
            sim_matrix.append(similarity_score)

            if debug_mode:
                print(f'\nFor {sent}\nSimilarity Score = {similarity_score} ')

        return sim_matrix.index(max(sim_matrix))

    def universal_output_module(self, output, output_type):

        if output_type in ["Image", "image", "Picture", "picture", "Illustration", "illustration"]:
            output.save("output_image.png")
        else:
            print(output)

    def core_distiller(self, singlestring=True):
        # the core distiller takes the descriptions from each core, and attempts to parse them into a single prompt
        # expecting the core list to be a list of SystemCores

        if singlestring:
            core_descriptions = "Description 0 describes general inquiries, not related to any other listed description"

            for index, item in enumerate(self.attached_cores):
                core_descriptions = core_descriptions + "Description " + str(index + 1) + " " + item.core_description + ", "

            return core_descriptions
        else:
            description_00 = "Description 0 describes general inquiries, not related to any other listed description"
            core_descriptions = [description_00]

            for index, item in enumerate(self.attached_cores):
                core_descriptions.append(item)

            return core_descriptions

    def phaneron_basic(self):
        # not implemented yet
        '''This module serves as a sort of dumb memory for understanding short term content about the interaction'''

    def llm_input_diagnostics_module(self, list_dictionary_queries):

        correct = 0
        number_queries = len(list_dictionary_queries)


        for index, item in enumerate(list_dictionary_queries):
            prompt = item["Query"]
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
            if answer == item["Goal"]:
                correct += 1

        return [correct, number_queries]

    def ssm_input_diagnostics_module(self, list_dictionary_queries):

        correct = 0
        number_queries = len(list_dictionary_queries)

        # sentences from which to choose from
        sentences = self.core_distiller(singlestring=False)




        for index, item in enumerate(list_dictionary_queries):
            prompt = item["Query"]

            # vectorize input
            test_vec = model.encode([prompt])[0]

            sim_matrix = []

            # aquire similarity scores (the higher the better)
            for sent in sentences:
                similarity_score = 1 - distance.cosine(test_vec, model.encode([sent])[0])
                sim_matrix.append(similarity_score)

            answer = str(sim_matrix.index(max(sim_matrix)))

            print(f"Answer to Query# {index}")
            print(answer)
            if answer == item["Goal"]:
                correct += 1

        return [correct, number_queries]

    def individual_diagnostics_module(self, list_dictionary_queries):

        core_analysis_list = self.attached_cores

        all_results = []

        for item in list_dictionary_queries:
            item_goal = int(item["Goal"])

            core_0_response = self.llm_central_interperatation_module(item["Query"], use_as_core=True)
            # check for actual core:
            if item_goal == 0:
                core_0_check = self.llm_central_interperatation_module(item["Query"], use_as_core=True)
            else:
                core_0_check = self.attached_cores[item_goal-1].prompt_core(item["Query"], test_mode=True)

            core_0_result = {
                "Prompt": item["Query"],
                "Test Core": 0,
                "Core Response": core_0_response,
                "Preferred Core": item["Goal"],
                "Preferred Core Response": core_0_check,
                "Score": self.semantic_similarity(core_0_response, core_0_check)
                }

            all_results.append(core_0_result)

            # ----------------------------------------------------------------------------------------------------------
            core_1_response = self.attached_cores[0].prompt_core(item["Query"], test_mode=True)
            # check for actual core:
            if item_goal == 0:
                core_1_check = self.llm_central_interperatation_module(item["Query"], use_as_core=True)
            else:
                core_1_check = self.attached_cores[item_goal-1].prompt_core(item["Query"], test_mode=True)

            core_1_result = {
                "Prompt": item["Query"],
                "Test Core": 1,
                "Core Response": core_1_response,
                "Preferred Core": item["Goal"],
                "Preferred Core Response": core_1_check,
                "Score": self.semantic_similarity(core_1_response, core_1_check)
            }

            all_results.append(core_1_result)

            # ----------------------------------------------------------------------------------------------------------
            core_2_response = self.attached_cores[1].prompt_core(item["Query"], test_mode=True)
            # check for actual core:
            if item_goal == 0:
                core_2_check = self.llm_central_interperatation_module(item["Query"], use_as_core=True)
            else:
                core_2_check = self.attached_cores[item_goal-1].prompt_core(item["Query"], test_mode=True)

            core_2_result = {
                "Prompt": item["Query"],
                "Test Core": 2,
                "Core Response": core_2_response,
                "Preferred Core": item["Goal"],
                "Preferred Core Response": core_2_check,
                "Score": self.semantic_similarity(core_2_response, core_2_check)
            }

            all_results.append(core_2_result)

            # ----------------------------------------------------------------------------------------------------------
            core_3_response = self.attached_cores[2].prompt_core(item["Query"], test_mode=True)
            # check for actual core:
            if item_goal == 0:
                core_3_check = self.llm_central_interperatation_module(item["Query"], use_as_core=True)
            else:
                core_3_check = self.attached_cores[item_goal-1].prompt_core(item["Query"], test_mode=True)

            core_3_result = {
                "Prompt": item["Query"],
                "Test Core": 3,
                "Core Response": core_3_response,
                "Preferred Core": item["Goal"],
                "Preferred Core Response": core_3_check,
                "Score": self.semantic_similarity(core_3_response, core_3_check)
            }

            all_results.append(core_3_result)

            # ----------------------------------------------------------------------------------------------------------
            core_4_response = self.attached_cores[3].prompt_core(item["Query"], test_mode=True)
            # check for actual core:
            if item_goal == 0:
                core_4_check = self.llm_central_interperatation_module(item["Query"], use_as_core=True)
            else:
                core_4_check = self.attached_cores[item_goal-1].prompt_core(item["Query"], test_mode=True)

            core_4_result = {
                "Prompt": item["Query"],
                "Test Core": 4,
                "Core Response": core_4_response,
                "Preferred Core": item["Goal"],
                "Preferred Core Response": core_4_check,
                "Score": self.semantic_similarity(core_4_response, core_4_check)
            }

            all_results.append(core_4_result)

            # ----------------------------------------------------------------------------------------------------------

        results = pd.DataFrame.from_records(all_results)

        results.to_csv("Results")

    def semantic_similarity(self, input_string, goal_string):
        test_vec = model.encode([input_string])[0]
        # aquire similarity scores (the higher the better)
        similarity_score = 1 - distance.cosine(test_vec, model.encode([goal_string])[0])
        return similarity_score



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

    def prompt_core(self, prompt, test_mode=False):

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

    def prompt_core(self, prompt, test_mode=False):

        if test_mode:
            return "This Would have generated an Image..."

        else:
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
                image = pipe(prompt, height=256, width=256).images[0]
                image.save("output_image.png")
                return image
                #image.save("output_image.png")


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
                       core_model="ALIENTELLIGENCE/shakespeare",
                       ollama_core=True)

Medical_Core = SystemCore(core_description=" related to medicine, medical diagnosis, figuring out illnesses or medical conditions based on a set of symptoms given",
                       core_input_preference="Text",
                       core_output_preference="Text",
                       core_model="medllama2:latest",
                       ollama_core=True)


AttachedCores=[Code_Core, Diffuse_Core, Shakespear_Core, Medical_Core]

CaSMAI = CoreInputOutput(model="llama3.2", attached_cores=AttachedCores)

test_cases = [
    {"Query": "Patient is feeling sick, exhibiting symptoms of diarrhea, fever and sore throat. Do you have a medical diagnosis?", "Goal": "4"},
    {"Query": "I want a witty poem that tells of a romance between two star crossed lovers.", "Goal": "3"},
    {"Query": "How would I write a class in python for dynamic linked lists?", "Goal": "1"},
    {"Query": "Could you draw me a picture of the mona lisa?", "Goal": "2"},
    {"Query": "Can you write me a function in C for a list that dynamically allocates memory based on the amount of items stored in it?", "Goal": "1"},
    {"Query": "Tell me a funny theater related joke.", "Goal": "3"},
    {"Query": "How are you today?", "Goal": "0"},
    {"Query": "Why is the sky blue?", "Goal": "0"},
    {"Query": "Can you show me what an illustration of an apple looks like?", "Goal": "2"},
    {"Query": "I have a patient who is showing symptoms of migraines, fever, sore throat and rash. Do you know what's wrong with them?", "Goal": "4"}
]

New_Test_Cases = [
    {"Query": "What are the symptoms of Type 2 diabetes?", "Goal": 1},
    {"Query": "What is the difference between Type 1 and Type 2 diabetes?", "Goal": 1},
    {"Query": "Can you explain how the human heart works?", "Goal": 1},
    {"Query": "What are some common side effects of ibuprofen?", "Goal": 1},
    {"Query": "What are the stages of wound healing?", "Goal": 1},
    {"Query": "How does cholesterol affect heart health?", "Goal": 1},
    {"Query": "What lifestyle changes can help prevent hypertension?", "Goal": 1},
    {"Query": "Can you diagnose me with a sore throat and fever?", "Goal": 1},
    {"Query": "What are the symptoms of an anxiety disorder?", "Goal": 1},
    {"Query": "How does insulin work in the body?", "Goal": 1},

    {"Query": "Generate a futuristic city skyline with flying cars.", "Goal": 2},
    {"Query": "Create an image of a fantasy dragon perched on a mountain.", "Goal": 2},
    {"Query": "Generate a surrealist landscape with melting clocks.", "Goal": 2},
    {"Query": "Create a realistic portrait of a knight in armor.", "Goal": 2},
    {"Query": "Generate a sci-fi scene with robots exploring an alien planet.", "Goal": 2},
    {"Query": "Create a peaceful forest glade with sunlight filtering through the trees.", "Goal": 2},
    {"Query": "Generate a minimalist abstract art piece with geometric shapes.", "Goal": 2},
    {"Query": "Create an underwater world with colorful fish and coral reefs.", "Goal": 2},
    {"Query": "Generate an ancient city in ruins with overgrown vegetation.", "Goal": 2},
    {"Query": "Create an image of a magical creature flying over a rainbow.", "Goal": 2},

    {"Query": "Shall I compare thee to a summer's day?", "Goal": 3},
    {"Query": "Write a sonnet in the style of Shakespeare about love and loss.", "Goal": 3},
    {"Query": "Pen a Shakespearean dialogue between two courtiers.", "Goal": 3},
    {"Query": "Write a soliloquy expressing the inner turmoil of a king.", "Goal": 3},
    {"Query": "Compose a poem in iambic pentameter on the fleeting nature of time.", "Goal": 3},
    {"Query": "What would Hamlet say to Ophelia if he were still alive?", "Goal": 3},
    {"Query": "Write a Shakespearean sonnet about the power of ambition.", "Goal": 3},
    {"Query": "Create a witty exchange between two Elizabethan jesters.", "Goal": 3},
    {"Query": "Write a tragic monologue in the style of Macbeth.", "Goal": 3},
    {"Query": "Write a poem in the style of Shakespeare about the changing seasons.", "Goal": 3},

    {"Query": "Write a Python function to calculate the factorial of a number.", "Goal": 4},
    {"Query": "Fix the error in this code: `for i in range(10): print(i 1)`", "Goal": 4},
    {"Query": "How can I implement a binary search algorithm in Python?", "Goal": 4},
    {"Query": "Write a Python script to read a CSV file and display the contents.", "Goal": 4},
    {"Query": "Write a function that reverses a string in Python.", "Goal": 4},
    {"Query": "Debug this code: `if x = 10: print(x)`", "Goal": 4},
    {"Query": "How do I use recursion to solve a problem in Python?", "Goal": 4},
    {"Query": "Create a Python program to implement a basic calculator.", "Goal": 4},
    {"Query": "Write a Python function to find the prime numbers up to 100.", "Goal": 4},
    {"Query": "How can I use a dictionary in Python to count the frequency of words?", "Goal": 4},

    {"Query": "What is the capital of France?", "Goal": 5},
    {"Query": "What is the distance from Earth to the Moon?", "Goal": 5},
    {"Query": "Explain the theory of relativity in simple terms.", "Goal": 5},
    {"Query": "What are the benefits of meditation for mental health?", "Goal": 5},
    {"Query": "Who wrote the novel '1984'?", "Goal": 5},
    {"Query": "What is the best way to learn a new language?", "Goal": 5},
    {"Query": "How do plants perform photosynthesis?", "Goal": 5},
    {"Query": "What is the largest ocean in the world?", "Goal": 5},
    {"Query": "Can you explain quantum physics in layman's terms?", "Goal": 5},
    {"Query": "What are the top 10 most popular programming languages?", "Goal": 5},

    {"Query": "What is the treatment for a bacterial infection?", "Goal": 1},
    {"Query": "How do vaccines work to protect the immune system?", "Goal": 1},
    {"Query": "Can you explain how an MRI works?", "Goal": 1},
    {"Query": "What are the symptoms of depression?", "Goal": 1},
    {"Query": "How does the body fight off viruses?", "Goal": 1},
    {"Query": "What are some treatments for chronic migraines?", "Goal": 1},
    {"Query": "Can you explain the process of organ transplantation?", "Goal": 1},
    {"Query": "What are the signs of an allergic reaction?", "Goal": 1},
    {"Query": "What causes high blood pressure?", "Goal": 1},
    {"Query": "What are some early signs of Alzheimer's disease?", "Goal": 1},

    {"Query": "Generate an image of a futuristic city with sleek, tall skyscrapers.", "Goal": 2},
    {"Query": "Create an artistic depiction of a sunset over the ocean.", "Goal": 2},
    {"Query": "Generate a portrait of an astronaut floating in space.", "Goal": 2},
    {"Query": "Create an image of a robot in a lush jungle.", "Goal": 2},
    {"Query": "Generate a scene from an ancient Greek temple surrounded by columns.", "Goal": 2},
    {"Query": "Create an image of an intergalactic space battle with spaceships and lasers.", "Goal": 2},
    {"Query": "Generate a scene of a medieval knight riding a horse through a forest.", "Goal": 2},
    {"Query": "Create an underwater landscape with a school of fish.", "Goal": 2},
    {"Query": "Generate a scene of a haunted house on a stormy night.", "Goal": 2},
    {"Query": "Create a painting of a serene mountain landscape with snow.", "Goal": 2},

    {"Query": "Write a short story in the style of Shakespeare about a noble’s fall from grace.", "Goal": 3},
    {"Query": "Compose a Shakespearean dialogue between two lovers in a secret garden.", "Goal": 3},
    {"Query": "Write a poem about betrayal in the style of Shakespeare.", "Goal": 3},
    {"Query": "Write a Shakespearean-style lament for a fallen hero.", "Goal": 3},
    {"Query": "Write a Shakespearean sonnet about the fleeting nature of beauty.", "Goal": 3},
    {"Query": "Create a soliloquy in the style of Hamlet, pondering life’s meaning.", "Goal": 3},
    {"Query": "Write a Shakespearean monologue on the perils of unchecked ambition.", "Goal": 3},
    {"Query": "Write a Shakespearean-style poem about the virtues of patience.", "Goal": 3},
    {"Query": "Write a poetic dialogue between two philosophers in Shakespearean language.", "Goal": 3},
    {"Query": "Write a Shakespearean sonnet about the passing of time.", "Goal": 3}
]

micro_test_cases = [
    {"Query": "Patient is feeling sick, exhibiting symptoms of diarrhea, fever and sore throat. Do you have a medical diagnosis?", "Goal": "4"},
    {"Query": "I want a witty poem that tells of a romance between two star crossed lovers.", "Goal": "3"}

]
# CaSMAI.universal_input_module(diagnostics_mode=True, diagnostics_array=test_cases)
# CaSMAI.universal_input_module(debug_mode=True)

CaSMAI.individual_diagnostics_module(micro_test_cases)


