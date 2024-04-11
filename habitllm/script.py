"""
script.py - main entrance of the script into the Text Generation Web UI system extensions

HabitLLML: a continuous learning extension for Text Gen Web UI.

"""

from pathlib import Path

import gradio as gr
import torch
from transformers import LogitsProcessor


from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

from .inference_specific_context_system import (
    add_files_to_vector_store,
    perform_similarity_search,
    INF_SPECIFIC_DST,
)

from .model_persistent_context_system import ModelPersistentContext

import extensions.habitllm.parameters as parameters
from .routine_handler import run_routine

params = {
    "display_name": "Habit LLM",
    "is_tab": False,
    "open": True,
}

MPC: ModelPersistentContext = None
# ---------------- Routines ----------------


def _run_routine(routine: str):
    current_routine = parameters.get_active_routine()
    if routine == current_routine:
        yield "### Routine is already running"
    else:
        run_routine(routine)


# ---------------- Context stores ----------------


def _update_context_store_config(
    inference_specific_context: int, model_persistent_context: int
):
    parameters.set_inference_specific_context_chunks(inference_specific_context)
    parameters.set_model_persistent_context_chunks(model_persistent_context)

    parameters.set_is_inference_specific_context(inference_specific_context != 0)
    parameters.set_is_model_persistent_context(model_persistent_context != 0)

    assert (
        parameters.get_inference_specific_context_chunks() == inference_specific_context
    )
    assert parameters.get_model_persistent_context_chunks() == model_persistent_context


# ---------------- Inference specific context system ----------------


def _feed_data_into_vector_store(files: list[str] | None):
    yield "### Reading and processing the input files..."
    add_files_to_vector_store(files)
    yield "### Done!"


def _clear_data(files_input: gr.Files):
    print("Clearing data")
    print(files_input)
    files_input.clear()
    yield "### Data Cleared!"


# ---------------- Model Persistent Context System ----------------


def _setup_persistent_context_module():
    global MPC
    if MPC is None:
        MPC = ModelPersistentContext()


def _ingest_data_into_persistent_db(files: list[str] | None):
    yield "### Reading and processing the input files..."
    files = Path(INF_SPECIFIC_DST).glob("**/*")
    MPC.add_files_to_vector_store(files)
    yield "### Done!"


# ---------------- Custom Extension ----------------
class MyLogits(LogitsProcessor):
    """
    Manipulates the probabilities for the next token before it gets sampled.
    Used in the logits_processor_modifier function below.
    """

    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        # probs[0] /= probs[0].sum()
        # scores = torch.log(probs / (1 - probs))
        return scores


def history_modifier(history):
    """
    Modifies the chat history.
    Only used in chat mode.
    """
    return history


def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    return state


def chat_input_modifier(text, visible_text, state):
    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """
    return text, visible_text


def input_modifier(string, state, is_chat=False):
    """
    In default/notebook modes, modifies the whole prompt.

    In chat mode, it is the same as chat_input_modifier but only applied
    to "text", here called "string", and not to "visible_text".
    """
    return string


def bot_prefix_modifier(string, state):
    """
    Modifies the prefix for the next bot reply in chat mode.
    By default, the prefix will be something like "Bot Name:".
    """
    return string


def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Modifies the input ids and embeds.
    Used by the multimodal extension to put image embeddings in the prompt.
    Only used by loaders that use the transformers library for sampling.
    """
    return prompt, input_ids, input_embeds


def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    processor_list.append(MyLogits())
    return processor_list


def output_modifier(string, state, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    return string


def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """
    print(user_input)

    inference_specific_results = perform_similarity_search(
        user_input, k=parameters.get_inference_specific_context_chunks()
    )
    print(f"ISC similarity search results: {inference_specific_results}")

    model_rag_ret = MPC.perform_similarity_search(
        user_input,
        k=parameters.get_inference_specific_context_chunks()
    )
    print(f"MPC similarity search results: {model_rag_ret}")

    rag_rets = inference_specific_results + model_rag_ret
    relevant_chunks = [result[0].page_content for result in rag_rets]


    input = user_input.join(relevant_chunks)
    result = chat.generate_chat_prompt(input, state, **kwargs)
    return result


def custom_css():
    """
    Returns a CSS string that gets appended to the CSS for the webui.
    """
    return ""


def custom_js():
    """
    Returns a javascript string that gets appended to the javascript
    for the webui.
    """
    return ""


def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    _setup_persistent_context_module()


def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """

    with gr.Accordion(
        params["display_name"], open=params["open"], elem_id="habitllm-exension"
    ):
        with gr.Row():
            with gr.Column(min_width=600):
                with gr.Tab("File input"):
                    files_input = gr.Files(label="Input files", type="filepath")
                    update_files = gr.Button("Load files")
                    memorize_button = gr.Button("Ingest Documents")
                    # clear_button = gr.Button('❌ Clear Data')

                with gr.Tab("Settings"):
                    routine = gr.Radio(
                        choices=parameters.get_routine_choices(),
                        value=parameters.get_active_routine(),
                        label="Routine",
                        info="What routine should be run",
                    )
                    run_routine = gr.Button("Run routine")

                    inference_specific_context = gr.Slider(
                        value=parameters.get_inference_specific_context_chunks(),
                        label="Number of chunks retrieved from inference specific context module",
                        minimum=0,
                        maximum=10,
                        step=1,
                        info="Number of chunks used for RAG with uploaded files.",
                    )
                    model_persistent_context = gr.Slider(
                        value=parameters.get_model_persistent_context_chunks(),
                        label="Number of chunks retrieved from model persistent context module",
                        minimum=0,
                        maximum=10,
                        step=1,
                        info="Number of chunks used for RAG with persistent model context.",
                    )
                    update_context_store_config = gr.Button("Apply")

            with gr.Column():
                last_updated = gr.Markdown()

    update_files.click(
        _feed_data_into_vector_store, [files_input], last_updated, show_progress=True
    )
    memorize_button.click(_ingest_data_into_persistent_db, show_progress=True)
    run_routine.click(_run_routine, [routine], last_updated, show_progress=False)
    update_context_store_config.click(
        _update_context_store_config,
        [inference_specific_context, model_persistent_context],
        last_updated,
        show_progress=False,
    )
    # clear_button.click(_clear_data, [files_input], last_updated, show_progress=True)
