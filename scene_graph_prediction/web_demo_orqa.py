# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import re
from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import TextIteratorStreamer

from scene_graph_prediction.main import config_loader
from scene_graph_prediction.scene_graph_helpers.model.scene_graph_prediction_model_oracle import ORQAWrapperQA


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7864, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    args = parser.parse_args()
    return args


def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace('`', r'\`')
                    line = line.replace('<', '&lt;')
                    line = line.replace('>', '&gt;')
                    line = line.replace(' ', '&nbsp;')
                    line = line.replace('*', '&ast;')
                    line = line.replace('_', '&lowbar;')
                    line = line.replace('-', '&#45;')
                    line = line.replace('.', '&#46;')
                    line = line.replace('!', '&#33;')
                    line = line.replace('(', '&#40;')
                    line = line.replace(')', '&#41;')
                    line = line.replace('$', '&#36;')
                lines[i] = '<br>' + line
    text = ''.join(lines)
    return text


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def _is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video' in item:
                new_item = {'type': 'video', 'video': item['video']}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages


def _transform_messages_orqa(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = ''
        images = []
        for item in message['content']:
            if 'image' in item:
                images.append(item['image'].replace('file://', '').strip())
            elif 'text' in item:
                new_content += item['text']
        # prepend image tags
        new_content = '<image>' * len(images) + new_content
        new_message = {'role': message['role'], 'content': new_content, 'images': images}
        transformed_messages.append(new_message)
    # add final message (essentially generation prompt)
    transformed_messages.append({'role': 'assistant', 'content': ''})
    return transformed_messages


def _launch_demo(args, model_wrapper):
    def call_local_model(model_wrapper, original_messages):
        images = []
        messages = _transform_messages_orqa(original_messages)
        for message in messages:  # extract all the images.
            image = message.pop('images', [])
            images.extend(image)
        processed_messages = model_wrapper.mm_plugin.process_messages(messages, images, [], model_wrapper.processor)
        # encode
        input_ids, _ = model_wrapper.template.encode_oneturn(model_wrapper.tokenizer, processed_messages, system=None, tools=None)
        all_features = [{
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "images": images,
        }]
        batch_features = model_wrapper.data_collator(all_features)  # returns a batch ready for model
        # Move batch to model device if not done
        for k, v in batch_features.items():
            if isinstance(v, torch.Tensor):
                batch_features[k] = v.to(model_wrapper.model.device)

        tokenizer = model_wrapper.processor.tokenizer
        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {'max_new_tokens': 512, 'streamer': streamer, **batch_features, 'use_cache': True}

        thread = Thread(target=model_wrapper.model.generate, kwargs=gen_kwargs)
        thread.start()

        generated_text = ''
        for new_text in streamer:
            generated_text += new_text
            yield generated_text

    def create_predict_fn():

        def predict(_chatbot, task_history):
            nonlocal model_wrapper
            chat_query = _chatbot[-1][0]
            query = task_history[-1][0]
            if len(chat_query) == 0:
                _chatbot.pop()
                task_history.pop()
                return _chatbot
            print('User: ' + _parse_text(query))
            history_cp = copy.deepcopy(task_history)
            full_response = ''
            messages = []
            content = []
            for q, a in history_cp:
                if isinstance(q, (tuple, list)):
                    if _is_video_file(q[0]):
                        content.append({'video': f'file://{q[0]}'})
                    else:
                        content.append({'image': f'file://{q[0]}'})
                else:
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
            messages.pop()

            for response in call_local_model(model_wrapper, messages):
                _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))

                yield _chatbot
                full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print('Qwen-VL-Chat: ' + _parse_text(full_response))
            yield _chatbot

        return predict

    predict = create_predict_fn()

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ''

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value='')

    def reset_state(_chatbot, task_history):
        task_history.clear()
        _chatbot.clear()
        _gc()
        return []

    #
    # with gr.Blocks() as demo:
    #     chatbot = gr.Chatbot(label='ORQA', elem_classes='control-height', height=900)
    #     query = gr.Textbox(lines=2, label='Input')
    #     task_history = gr.State([])
    #
    #     with gr.Row():
    #         addfile_btn = gr.UploadButton('📁 Upload', file_types=['image', 'video'])
    #         submit_btn = gr.Button('🚀 Submit')
    #         empty_bin = gr.Button('🧹 Clear History')
    #
    #     submit_btn.click(add_text, [chatbot, task_history, query],
    #                      [chatbot, task_history]).then(predict, [chatbot, task_history], [chatbot], show_progress=True)
    #     submit_btn.click(reset_user_input, [], [query])
    #     empty_bin.click(reset_state, [chatbot, task_history], [chatbot], show_progress=True)
    #     addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)
    #
    # demo.queue().launch(
    #     share=args.share,
    #     inbrowser=args.inbrowser,
    #     server_port=args.server_port,
    #     server_name=args.server_name,
    # )

    with gr.Blocks(css="""
        .suggestion-btn {
            font-size: 0.8em;
            margin: 2px;
        }
        .suggestions-container {
            display: flex;
            flex-wrap: wrap;
        }
    """) as demo:
        chatbot = gr.Chatbot(label='ORQA', elem_classes='control-height', height=900)
        query = gr.Textbox(lines=2, label='Input')

        with gr.Row():
            addfile_btn = gr.UploadButton('📁 Upload', file_types=['image', 'video'])
            submit_btn = gr.Button('🚀 Submit')
            empty_bin = gr.Button('🧹 Clear History')

        suggestions = [
            "How many people are in the OR?",
            "Who is in the OR?",
            "What is the interaction of head surgeon with patient?",
            "What is the color of the drill?",
            "What is currently happening?",
            "Where is the hammer in the image?",
            "How far is the patient from the anaesthesia machine?",
            "Which tools are currently being used?",
            "What is the current scene graph?",
            "List all entities in the OR.",
            "What is the surgeon looking at?",
            "Summarize the information on the monitor."
        ]

        with gr.Row(elem_classes="suggestions-container"):
            for sug in suggestions:
                btn = gr.Button(sug, elem_classes="suggestion-btn")
                btn.click(lambda s=sug: s, outputs=query)

        task_history = gr.State([])
        submit_btn.click(add_text, [chatbot, task_history, query],
                         [chatbot, task_history]).then(predict, [chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    config = config_loader(args.config)
    # Create the QA wrapper object
    print(f"Loading ORQAWrapperQA with model: {args.model_path}")
    model_wrapper = ORQAWrapperQA(config=config, model_path=args.model_path, with_sg_grounding=False)  # or True if you want to inject scene graph placeholders
    _launch_demo(args, model_wrapper)


if __name__ == '__main__':  # uvloop==0.21.0
    main()
