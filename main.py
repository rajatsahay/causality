import os
import glob
import json

import numpy as np
import cv2
from scipy import spatial
import pandas as pd
from detector import get_coords

import ast
import openai
import tiktoken


GPT_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = "XXXXXXXXXXXX" # ADD API KEY HERE

#####################################################################################

def init_openai():
    if len(sys.argv) < 2:
        print("Usage: python3 + " + GPT_MODEL)
        sys.exit(1)
    
    openai.api_key = OPENAI_API_KEY

'''
def get_coords(video_id):
    """
    Gets coordinates of each object from object detection (blobs + frame differencing) model
    """
    # shifted to separate file
'''

def get_object_properties(video_id):
    """
    Gets the object properties (shape, color, material) from the ground truth annotation file (downloaded separately)
    """
    annotation_path = "/home/rajat/research/causality/annotation_validation/annotation_14000-15000/annotation_" + video_id + ".json" # CHANGE PATH
    annotation_file = open(annotation_path)
    annotation_data = json.load(annotation_file)

    return annotation_data["object_property"]

def define_system_prompt(video_id):
    """
    Constructs the system prompt given to GPT model. System Prompt includes:
        Object Properties from JSON file
        Framewise coordinates from object detection system
        Definition of collision
    """
    system_prompt = "Given below are the properties of certain objects in a JSON format. The properties include the color, shape, and material of the objects.\n\n"
    system_prompt = get_object_properties(video_id) # adds object properties (color, shape etc) to the system prompt
    system_prompt += "\n\n"
    system_prompt += "The objects defined above are moving on an XY Cartesian plane and their individual X and Y coordinates have been recorded for every frame. These are provided below.\n\n"
    system_prompt += get_coords(video_id) # adds framewise coordinates to the system prompt
    system_prompt += "\n\n"
    system_prompt += "We can define a collision as a phenomenon where the difference between either the X or the Y coordinates of two different objects is less than 3 units. \n\n" # change definition of collision (07/05)
    system_prompt += "Based on the given data, answer the questions posed by the user."

    return system_prompt


def main():

    # ADD PATH TO DIRECTORY CONTAINING VIDEO FILES HERE
    video_ids = []
    vid_path = "/home/rajat/research/causality/video_validation/video_14000-15000/" # currently using only 1000 videos due to API rate limits
    for path in os.walk(vid_path):
        for id in path[2]:
            video_ids.append(id[6:11])

    # ADD PATH TO validation.json FILE HERE
    qa_path = "/home/rajat/research/causality/validation.json"
    qa_file = open(qa_path)
    qa_data = json.load(qa_file)

    for video_id in video_ids:
        # define system prompt
        system_prompt = define_system_prompt(video_id)

        # define user prompt
        for qa_id in range(len(qa_data)):
            if qa_data[qa_id]["video_filename"][6:11] == video_id:
                for question in range(len(qa_data[qa_id]["questions"])):
                    if qa_data[qa_id]["questions"][question]["question_type"] == "descriptive":
                        # for descriptive questions
                        question_type = "descriptive"
                        user_prompt = qa_data[qa_id]["questions"][question]["question"] # define user prompt
                        answer = qa_data[qa_id]["questions"][question]["answer"] # correct answer
                    else:
                        # for all other questions (with multiple choices)
                        question_type = qa_data[qa_id]["questions"][question]["question_type"]

                        '''
                        if question_type == "counterfactual":
                            user_prompt += "This is a counterfactual question. Please answer accordingly. \n\n"
                        '''

                        user_prompt = qa_data[qa_id]["questions"][question]["question"] # define user prompt
                        user_prompt += "\n"
                        for choice in range(len(qa_data[qa_id]["questions"][question]["choices"])):
                            user_prompt_choice = "Choice " + qa_data[qa_id]["questions"][question]["choices"][choice]["choice_id"] + " : " + qa_data[qa_id]["questions"][question]["choices"][choice]["choice"]
                            user_prompt += user_prompt_choice
                        
                        for correct_choice in range(len(qa_data[qa_id]["questions"][question]["choices"])):
                            if qa_data[qa_id]["questions"][question]["choices"][correct_choice]["answer"] == "correct":
                                answer = qa_data[qa_id]["questions"][question]["choices"][correct_choice]["choice_id"] # correct answer
        
        # call openai api
        response = openai.ChatCompletion.create(
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            model = GPT_MODEL,
            temperature = 0.5        # lower temperature gives more consistent results, higher temperature gives more creative results
        )

        gpt_answer = response["choices"][0]["message"]["content"]

        # create log file
        try:
            with open("gpt_logs/" + video_id + ".txt", 'w') as f:
                    f.write("Video ID: " + video_id)
                    f.write()
                    f.write("Question : " + user_prompt)
                    f.write("Question Tye : " + question_type)
                    f.write("Answer : " + answer)
                    f.write("GPT Answer : " + gpt_answer)
            except FileNotFoundError:
                print("The directory 'gpt_logs' does not exist at this location.")


if __name__ == "__main__":
    init_openai()
    main()

'''
response = openai.ChatCompletion.create(
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ],
    model = GPT_MODEL,
    temperature = 0
)

print(response['choices'][0]['message']['content'])
'''
