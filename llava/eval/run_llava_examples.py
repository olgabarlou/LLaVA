import argparse
import torch
import os

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    directory = '/kaggle/input/sd-images-v4'
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_file = os.path.join(directory, filename)
            image_files = [image_file]
            #image_files = image_parser(args)
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(outputs)
            del images_tensor, input_ids, output_ids
            with open('/kaggle/working/lyrics.txt', 'a') as f:
                f.write(f"Results for {filename}:\n")
                f.write(outputs)
                f.write('\n--------------------------\n')




def eval_model_old(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name #, load_8bit=True
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    #7badbc83810546339c357962bf82f85d - pop
    lyrics1 = "shine bright like a diamond shine bright like a diamond find light in the beautiful sea i choose to be happy you and i you and i we are like diamonds in the sky you are a shooting star i see a vision of ecstasy when you hold me i am alive we are like diamonds in the sky i knew that we would become one right away oh, right away at first sight i felt the energy of sun rays i saw the life so shine bright tonight you and i we are beautiful like diamonds in the sky eye to eye so alive we are beautiful like diamonds in the sky oh oh we are beautiful like diamonds in the sky oh oh we are beautiful like diamonds in the sky palms rise to the universe as we moonshine and molly feel the warmth we will never die we are like diamonds in the sky you are a shooting star i see a vision of ecstasy when you hold me i am alive we are like diamonds in the sky at first sight i felt the energy of sun rays i saw the life so shine bright tonight you and i we are beautiful like diamonds in the sky eye to eye so alive we are beautiful like diamonds in the sky oh oh we are beautiful like diamonds in the sky oh oh we are beautiful like diamonds in the sky shine bright like a diamond shine bright like a diamond shine bright like a diamond so shine bright tonight you and i we are beautiful like diamonds in the sky eye to eye so alive we are beautiful like diamonds in the sky shine bright like a diamond shine bright like a diamond shine bright like a diamond"
    #2726c7a522b44494a7dcd2913d5b4448 - rock
    lyrics2 = "i got a one way ticket on a hell bound train with nothing to lose and nothing to gain nobody ever taught me how to live i am feeling like i am lost like i will never be found i am twisted and i am turned around nobody ever taught me how to love i am hurting everybody i am hurting my self i am desperate so what do you do? when it. all comes down on you do you run and hide? or face the truth so what do you do? when it. all comes down on you do you run and hide or face the truth if you where to tell me that i died today this it is what i had have to say i never really had the time to live and if you where to give me just another chance another life another dance all i really want to do it is love i am hurting everybody i am hurting my self i am desperate so what do you do? when it. all comes down on you do you run and hide? or face the truth so what do you do? when it. all comes down on you do you run and hide or face the truth when i was sitting down you could be the one with open arms with open eyes you are jumping off the edge in hoping you can find except your fate for what it. it is into the great unknown so what do you do? when it. all comes down on you do you run and hide? or face the truth so what do you do? when it. all comes down on you do you run and hide or face the truth so what do you do? when it. all turned around on you do you run and hide or face the truth? so what do you do? what do you do? what do you do? do you run and hide or face the truth got a one way ticket on a hell bound train"
    #6a80e424e001469d918b4e2a3ce4df45 -alternative
    lyrics3 = " tell me why we never cared to do this when we still had time we will never have to give upif we never try i know i will only want it. when it. it is gone into the fire show me now i wish that i could fake it. but i do not know how i know we will never make it. but i cannot stop now we are only just beginning and it. it is over too late no time, it. it is over and now cannot wait now it. it is too late no time, it. it is over and now cannot wait and now we all fall down into the fire and my wishes have all come true we all fall down i do not want it. if i cannot be with you leave me here i will never see tomorrow until my eyes are clear we never could run faster than the passing years i know that i will not miss you until you are gone into the fire cross my heart we will never have to let this end if we do not start we will never see the light until we step into the dark we are only just beginning and it. it is over too late no time, it. it is over and now cannot wait now it. it is too late no time, it. it is over and now cannot wait and now we all fall down into the fire and my wishes have all come true we all fall down i do not want it. if i cannot be with you there it is a lesson that we learn in the pages that we burn it. it is written in the ashes of the fire below all the world it is spinning around as we crash into the ground we will never be together now"
    #b62b2a6e5bd94b57835d3ea2b567e910 - pop
    lyrics4 = "girl, girl, girl when i hear him talk ooh my mind gets blocked girl speak up because my jaw it is locked which it is good good for me girl it. hides like a warning sign i am too blind to see speak upyes i am coming down with an ice cold fever ice cold fever ice cold fever still got my hands they are clinging so i just keep going i do not know where i belong could i belong to you? no i do not know where i belong could i belong girl, girl, girl get it. while it. it is hot they say you see i am burning up here i want to but i just cannot stop and it. hurts it. hurts like hell girl you turn me inside out and upside down ooh you got me head over heels i am stuck yes i am coming down with an ice cold fever ice cold fever ice cold fever alright still got my hands they are clinging so i just keep going i do not know where i belong could i belong to you? no i do not know where i belong could i belong could i belong still got my hands they are clinging so i just keep going i do not know where i belong could i belong to you? no i do not know where i belong could i belong oh girl could i belong to you you you to you yeah you"
    #847bb881d5684aab9f7be5ab6f614d47 - alternative
    lyrics5 = "hey ho, on the road again moving on, forward stick and stones want break our bones we are in the car, on the highway it. so magical, feeling like no one it is got a hold you are a catalyst to your own happiness you know cause it.  your heart, it.  alive, it.  pumping blood and it.  your heart, it.  alive, it.  pumping blood and the whole wide world it is whistling hey ho on the run again drive it is strong, onwards stick and stones want take it.  course got the part, in the front seat it.  the best of worlds, feeling like nothing can go correct you are the decider of the world, you will get to know cause it.  your heart, it.  alive, it.  pumping blood and it.  your heart, it.  alive, it.  pumping blood and the whole wide world it is whistling hey ho, on the road again on the highway on the highway hey ho, on the road again on the highway on the highway cause it.  your heart, it.  alive, it.  pumping blood and it.  your heart, it.  alive, it.  pumping blood"
    #acb9956e2bb54e2eb53aa60bd954854d - rock 
    lyrics6 = "i had like to be under the sea in an octopus it is garden in the shade. he had led us in knows where we have been in his octopus it is garden in the shade. i had ask my friends to come and see an octopus it is garden with me. i had like to be under the sea in an octopus it is garden in the shade. we would be warm below the storm in our little hide away beneath the waves. resting our heads on the sea bed in an octopus it is garden near a cave. we would sing and dance around because we know we cannot be found. i had like to be under the sea in an octopus it is garden in the shade. we would shout and swim about the coral that lies beneath the waves. oh what joy for every girl and boy knowing they are happy and they are safe. we would be so happy you and me no one there to tell us what to do."
    
    prompt = "Create song lyrics that match the atmosphere and overall sentiment depicted in this image. Some examples of lyrics are:"
    prompt += f"\n Example 1: {lyrics1} \n Example 2: {lyrics2} \n Example 3: {lyrics3} \n Example 4: {lyrics4} \n Example 5: {lyrics5} \n Example 6: {lyrics6}"

    model_path = "liuhaotian/llava-v1.6-mistral-7b" #"liuhaotian/llava-v1.5-7b"
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        #"image_file": image_file,
        "sep": ",",
        "temperature": 0.6,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
        })()

    eval_model(args)
