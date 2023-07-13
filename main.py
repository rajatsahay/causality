import os
import glob
import json
import time

import numpy as np
import cv2
from scipy import spatial
import pycocotools._mask as _mask
import matplotlib.pyplot as plt
import torch
#import pandas as pd
# from detector import get_coords # deprecated

#import ast
import openai
#import tiktoken


GPT_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" # ADD API KEY HERE

# default vars
H = 100
W = 150
bbox_size = 24

#####################################################################################

def init_openai():
    '''
    if len(sys.argv) < 2:
        print("Usage: python3 + " + GPT_MODEL)
        sys.exit(1)
    '''
    
    openai.api_key = OPENAI_API_KEY


def normalize(x, mean, std):
    return (x - mean) / std

def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]

def convert_mask_to_bbox(mask, H=100, W=150, bbox_size=24):
    h, w = mask.shape[0], mask.shape[1]
    x = np.repeat(np.arange(h), w).reshape(h, w)
    y = np.tile(np.arange(w), h).reshape(h, w)
    x = np.sum(mask * x) / np.sum(mask) * (float(H) / h)
    y = np.sum(mask * y) / np.sum(mask) * (float(W) / w)
    bbox = int(x - bbox_size / 2), int(y - bbox_size / 2), bbox_size, bbox_size
    ret = np.ones((2, bbox_size, bbox_size))
    ret[0, :, :] *= x
    ret[1, :, :] *= y
    return bbox, torch.FloatTensor(ret)

def get_object_properties(video_id):
    """
    Gets the object properties (shape, color, material) from the ground truth annotation file (downloaded separately)
    """
    annotation_path = "/Users/rajat/Downloads/annotation_validation/annotation_14000-15000/annotation_" + video_id + ".json" # CHANGE PATH
    annotation_file = open(annotation_path)
    annotation_data = json.load(annotation_file)

    return annotation_data["object_property"]

def object_match(video_id):
    id_dict = {}
    annotations = get_object_properties(video_id)
    for idx in range(len(annotations)):
        id_dict.update({
            annotations[idx]["object_id"] : [annotations[idx]["color"], annotations[idx]["material"], annotations[idx]["shape"]]
            })
    return id_dict


def get_coords(video_id):
    """
    Returns framewise coordinates of each object from proposal_(video_id).json file (derendered proposals downloaded separately)
    Also writes it to a separate log file for later use
    """
    proposal_path = "/Users/rajat/Downloads/derender_proposals/proposal_" + video_id + ".json"
    prop_file = open(proposal_path)
    prop_data = json.load(prop_file)

    idx = prop_data["frames"]
    print("\n") # separator for terminal output

    try:
        with open("gpt_logs_test/" + video_id + "_coords.txt", 'w') as c:
            for i in range(len(idx)):
                c.write("\nFrame " + str(idx[i]["frame_index"]) + " : " + "\n")
                c.write("-------- \n")
                for j in range(len(idx[i]["objects"])):
                    # assigning the correct object id to every object (as per the annotation file)
                    object_id = 0
                    object_properties = [idx[i]["objects"][j]["color"], idx[i]["objects"][j]["material"], idx[i]["objects"][j]["shape"]]
                    #print(object_properties)
                    
                    object_annotations = object_match(video_id)
                    
                    try:
                        object_id = [obj_id for obj_id in object_annotations if object_annotations[obj_id] == object_properties][0]
                    except IndexError:
                        object_id = "X"
                        print("Mismatched object properties in Video", video_id, "Frame", str(idx[i]["frame_index"]) + ". Object marked as Object X.")
                    

                    mask_raw = decode(idx[i]["objects"][j]["mask"])
                    mask = cv2.resize(mask_raw, (W,H), interpolation=cv2.INTER_NEAREST)
                    bbox, pos = convert_mask_to_bbox(mask_raw, H, W, bbox_size)
                    pos_mean = torch.FloatTensor(np.array([H/2., W/2.]))
                    pos_mean = pos_mean.unsqueeze(1).unsqueeze(1)
                    pos_std = pos_mean

                    pos = normalize(pos, pos_mean, pos_std)
                    #print("Object " + str(object_id) + ":" + str(bbox[0:2]))
                    c.write("Object " + str(object_id) + ":" + str(bbox[0:2]) + "\n")
                c.write("\n")
    except FileNotFoundError:
        print("The directory 'gpt_coords' does not exist at this location.")


def define_system_prompt(video_id):
    """
    Constructs the system prompt given to GPT model. System Prompt includes:
        Object Properties from JSON file
        Framewise coordinates from object detection system
        Definition of collision
    """
    system_prompt = "Given below are the properties of certain objects in a JSON format. The properties include the color, shape, and material of the objects.\n\n"
    system_prompt += str(get_object_properties(video_id)) # adds object properties (color, shape etc) to the system prompt in a JSON format
    system_prompt += "\n\n"
    system_prompt += "The objects defined above are moving on an XY Cartesian plane and their individual X and Y coordinates have been recorded for every frame. These are provided below.\n\n"
    
    get_coords(video_id) # adds framewise coordinates to the system prompt
    try:
        with open("gpt_logs_test/" + video_id + "_coords.txt", 'r') as f:
            coords = f.read()
            system_prompt += coords
    except FileNotFoundError:
        print("Something went wrong. Please check if the object coordinates are generated and stored at the right location.")

    #system_prompt += coords
    system_prompt += "\n\n"
    #system_prompt += "We can define a collision as a phenomenon where the difference between either the X or the Y coordinates of two different objects is less than 3 units. \n\n" # change definition of collision (07/05)
    system_prompt += "A collision is defined as the phenomenon where the difference between both the X and Y coordinates of both objects is less than 12 units." # based on newer bboxes
    system_prompt += "Based on the given data, answer the questions posed by the user."

    return system_prompt


def main():

    # ADD PATH TO DIRECTORY CONTAINING VIDEO FILES HERE
    video_ids = []
    vid_path = "/Users/rajat/Desktop/microsoft/video_test/" # currently using only 1000 videos due to API rate limits
    for path in os.walk(vid_path):
        for id in path[2]:
            video_ids.append(id[6:11])
    #print(video_ids)

    # ADD PATH TO validation.json FILE HERE
    qa_path = "/Users/rajat/Desktop/microsoft/validation.json"
    qa_file = open(qa_path)
    qa_data = json.load(qa_file)

    for video_id in video_ids:
        # define system prompt
        system_prompt = define_system_prompt(video_id)

        # define user prompt
        for qa_id in range(len(qa_data)):
            if qa_data[qa_id]["video_filename"][6:11] == video_id:
                try:
                    with open("gpt_logs_test/" + video_id + ".txt", 'w') as f:
                        f.write("Video ID: " + video_id + "\n")
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
                                    user_prompt_choice = "Choice " + str(qa_data[qa_id]["questions"][question]["choices"][choice]["choice_id"]) + " : " + qa_data[qa_id]["questions"][question]["choices"][choice]["choice"]
                                    user_prompt += user_prompt_choice
                                    user_prompt += "\n"
                                
                                correct_choices = []
                                for correct_choice in range(len(qa_data[qa_id]["questions"][question]["choices"])):
                                    if qa_data[qa_id]["questions"][question]["choices"][correct_choice]["answer"] == "correct":
                                        correct_choices.append(qa_data[qa_id]["questions"][question]["choices"][correct_choice]["choice_id"])
                                        
                                    answer = ','.join(str(x) for x in correct_choices) # correct answer
                
                            gpt_answer = "<api call here>" # debug

                            # call openai api
                            time.sleep(30) # to ensure rate limit is not exceeded
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

                            # for debugging purposes. system prompt is not stored in the log file by default. 
                            # uncomment the below three lines if you want them stored in the log file alongside the user prompt
                            # note: system prompt includes framewise coordinates for all objects, making the log file difficult to follow
                            '''
                            f.write("\n\n\n")
                            f.write("System Prompt: " + system_prompt)
                            f.write("\n\n\n")
                            '''

                            f.write("\n")
                            f.write("Question : " + user_prompt)
                            f.write("\n")
                            f.write("Question Type : " + question_type)
                            f.write("\n")
                            f.write("Answer : " + str(answer))
                            f.write("\n")
                            f.write("GPT Answer : " + gpt_answer)
                            f.write("\n")
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