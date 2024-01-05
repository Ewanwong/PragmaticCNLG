def prepare_target_distractor_inputs(input_texts, target_prompts, distractor_prompts):
    """
    input_texts: a list of strings
    target_prompts: a list of prompts that encourage models to produce attributes of interest
    distractor_prompts: a list of prompts that encourage models to produce attributes as distractors
    """
    num_input = len(input_texts)
    num_target = len(target_prompts)
    num_distractors = len(distractor_prompts)
    inputs = []
    inputs += input_texts
    for target_prompt in target_prompts:
        for input_text in input_texts:
            inputs += [target_prompt + input_text]
    for distractor_prompt in distractor_prompts:
        for input_text in input_texts:
            inputs += [distractor_prompt + input_text]

    index_map = {}
    for i in range(num_input):
        index_map[i] = {}
        index_map[i]['targets'] = [i+num_input+j*num_target for j in range(num_target)]
        index_map[i]['distractors'] = [i+num_input+j*num_distractors for j in range(num_distractors)]
    return inputs, index_map