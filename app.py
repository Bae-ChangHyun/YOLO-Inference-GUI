import os
import re
import multiprocessing as mp
import signal
import sys
import queue
import threading

import gradio as gr
from ultralytics import YOLO
import yaml
import subprocess
from functools import partial
from logger import Logger
logger = Logger()


default_models = ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"]
debug_mode = True
process = None 
stop_event = threading.Event() 

def inference(params):
    global process, stop_event
    
    # 이전 프로세스가 있다면 중지
    if process and process.poll() is None:
        process.terminate()
        process.wait()
    
    # 중단 이벤트 초기화
    stop_event.clear()
    
    command = f"yolo detect predict {params}" # yolo command
    
    def run_command():
        global process
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid  # 프로세스 그룹 생성
            )
            logger.warning(f"PID {process.pid} Start.")
            
            for line in process.stdout:
                if stop_event.is_set():
                    break
                logger.info(line.strip())
            
        except Exception as e:
            logger.error(f"Error occured: {e}")
    
    thread = threading.Thread(target=run_command)
    thread.start()

def stop():
    global process, stop_event
    
    # 중단 이벤트 설정
    stop_event.set()
    
    if process and process.poll() is None:
        # 프로세스 그룹 전체 종료 (자식 프로세스 포함)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
            logger.warning(f"PID {process.pid}가 종료되었습니다.")
            return f"PID {process.pid}가 종료되었습니다."
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            logger.warning(f"PID {process.pid}가 강제 종료되었습니다.")
            return "프로세스를 강제 종료했습니다."
    else:
        logger.warning("종료할 프로세스가 없습니다.")
        return "종료할 프로세스가 없습니다."


def get_file_content(file, tail=0):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if(tail > 0): lines = lines[-tail:]
        return ''.join(lines)
    except Exception as e:
        return f"파일을 읽을 수 없습니다: {str(e)}"

def get_name(file):
    try:
        return file.name
    except Exception as e:
        print(e)
        return file
    
def get_model_path(model):
    try:
        return model
    except Exception as e:
        print(e)
        return None

params_state = {}

def update_params(value, label):
    # 파라미터 값을 딕셔너리에 업데이트
    if(type(value) == bool):
        params_state[label] = value
    else:
        params_state[label] = f"{value}"

    # 현재 딕셔너리의 값을 기반으로 final_params 재구성
    final_params = ' '.join(f"{k}={v}" for k, v in params_state.items() if v)
    return final_params

def setup_interface():
                
    with gr.Blocks() as demo:
        gr.Markdown("# YOLO INFERENCE TOOL")
        with gr.Tab("Inference"):
            
            gr.Markdown("## Select Model")
            model_choice = gr.Radio(["Default", "Custom"], label="Select model")
            default_model = gr.Dropdown(default_models, label="", visible=False, value=None)
            import_method = gr.Radio(["Path", "File"], label="모델 불러오기 방법", visible=False)

            model_choice.change(
                lambda choice: (
                    gr.update(visible=choice == "Default"),
                    gr.update(visible=choice == "Custom")
                ),
                inputs=model_choice, outputs=[default_model, import_method]
            )
            
            custom_model_path = gr.Textbox(label="model path", visible=False, value=None)
            custom_model_file = gr.File(label="Upload model", visible=False, value=None)
            
            import_method.change(
                lambda choice: (
                    gr.update(visible=choice == "Path"),
                    gr.update(visible=choice == "File")
                ),
                inputs=import_method, outputs=[custom_model_path, custom_model_file]
            )
            
            final_model = gr.Textbox(label="model", visible=debug_mode)
            for model_input in [default_model, custom_model_path, custom_model_file]:
                model_input.change(get_model_path, inputs=model_input, outputs=final_model)
            
            gr.Markdown("## Set arguments")
            with gr.Row():
                with gr.Column():
                    conf_limit = gr.Slider(0.1, 1.0, 0.25, label="conf", info="Set Confidence score limit")
                    iou_limit = gr.Slider(0.1, 1.0, 0.7, label="iou",info="Set IOU limit")
                with gr.Column():
                    save_bool = gr.Checkbox(label="save", info="Save the output")
                    show_label = gr.Checkbox(label="show_labels", info="Show labels")
                    show_conf = gr.Checkbox(label="show_conf", info="Show confidence score") 
                    show_conf = gr.Checkbox(label="show_boxes", info="Show bounding box") 
                    save_name = gr.Textbox(label="name", info="Save name")
                device = gr.Radio(["cpu", "cuda:0"], label="device")
                
            add_params = gr.Radio(['True', 'False'], label="Additional arguments", visible=True)
            default_params_code = gr.Code(lines=5, scale=2, language="yaml",linse = 20, visible=False)
            
            add_params.change(get_file_content, inputs=gr.Textbox("default.yaml",visible=False), outputs=default_params_code)

            params_arg = gr.Textbox(label="", placeholder="추가 파라미터를 입력하세요")
            
            add_params.change(
                lambda choice: (
                    gr.update(visible=choice == 'True'),
                    gr.update(visible=choice == 'True'),
                ),
                inputs=add_params,
                outputs=[default_params_code,params_arg]
            )
            
            with gr.Column():
                input_file_path = gr.Textbox(label="source", visible=True, value=None, info="Select the input file") 
                input_file_dir = gr.FileExplorer(
                    scale=1,
                    glob="*.mp4",
                    value=["themes/utils"],
                    file_count="single",
                    root_dir='/home',
                    elem_id="file",
                    interactive=True,
                    every=1,
                    height=200
                )
            input_file_dir.change(get_name, inputs=input_file_dir, outputs=input_file_path)
            
            final_params = gr.Textbox(label="Final Params", visible=debug_mode)
            for param_input in [final_model,input_file_path, conf_limit, iou_limit, save_bool, show_label, show_conf, device, save_name, params_arg]:
                param_input.change(
                    partial(update_params, label=param_input.label),
                    inputs=param_input,
                    outputs=final_params
                )
           
            run_button = gr.Button("Run")
            stop_button = gr.Button("Stop")
            with gr.Column():
                output = gr.Textbox(label="Result")
                with gr.Row():
                    refresh_button = gr.Button("Refresh")
                    log = gr.Code(lines=5, scale=2, language="yaml")
            refresh_button.click(get_file_content, inputs=[gr.Textbox("inference.log",visible=False),gr.Number(10, visible=False)], outputs=log)
            
            run_button.click(
                inference,
                inputs=final_params,
                outputs=output,
                show_progress=True
            )
            
            stop_button.click(
                stop,
                outputs=output
            )
    
    return demo

demo = setup_interface()
demo.launch()