import os
import subprocess
import threading
import signal

import yaml
import shutil
import gradio as gr
from ultralytics import YOLO
from functools import partial
from datetime import datetime

from logger import Logger
logger = Logger()

default_models = ["yolov10n", "yolov10s", "yolov10m", "yolov10l", "yolov10x"]
debug_mode = True
process = None 
stop_event = threading.Event() 

def save_log(file_name):
    
    log_path = f'runs/detect/{file_name}/inference.log'
    
    shutil.copy("inference.log", log_path)
    
    return log_path

def inference(params, save_name):
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
            logger.info(f"{command}")
            logger.debug(f"PID {process.pid} Start.")
            
            for line in process.stdout:
                if stop_event.is_set():
                    break
                logger.debug(line.strip())
            
            log_path = save_log(save_name)
            
        except Exception as e:
            logger.error(f"Error occured: {e}")
    
    thread = threading.Thread(target=run_command)
    thread.start()
    

def stop(save_name):
    global process, stop_event
    
    # 중단 이벤트 설정
    stop_event.set()
    
    if process and process.poll() is None:
        # 프로세스 그룹 전체 종료 (자식 프로세스 포함)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
            logger.warning(f"PID {process.pid} was terminated.")   
            log_path = save_log(save_name)
            
            return f"PID {process.pid} was terminated, log saved at {log_path}"
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            logger.warning(f"PID {process.pid}가 강제 종료되었습니다.")
            log_path = save_log(save_name)
            
            return f"PID {process.pid} was terminated, log saved at {log_path}"
    else:
        logger.warning("No process to stop.")
        return "No process to stop."


def get_file_content(file, tail=0):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if(tail > 0): lines = lines[-tail:]
        return ''.join(lines)
    except Exception as e:
        return f"Can't read File: {str(e)}"

def get_name(file):
    try:
        return file.name
    except Exception as e:
        print(e)
        return file

params_state = {}

def update_params(value, label):
    if(label != ""):
        if(type(value) == bool): params_state[label] = value
        else: params_state[label] = f"{value}"
    final_params = ' '.join(f"{k}={v}" for k, v in params_state.items() if v)
    return final_params

def setup_interface():
                
    with gr.Blocks() as demo:
        gr.Markdown("# YOLO GUI APP")
        with gr.Tab("Inference"):
            
            gr.Markdown("## Select Model")
            model_choice = gr.Radio(["Default", "Custom"], label="Select model")
            default_model = gr.Dropdown(default_models, label="", visible=False, value=None)
            import_method = gr.Radio(["Path", "File"], label="모델 불러오기 방법", visible=False)
            custom_model_path = gr.Textbox(label="Model path", visible=False, value=None)
            custom_model_file = gr.File(label="Upload model", file_types=['.pt'], visible=False, value=None)

            model_choice.change(
                lambda choice: (
                    gr.update(visible=choice == "Default"), # Must same with 'model_choice' gr.Radio(value[0])
                    gr.update(visible=choice == "Custom")   # Must same with 'model_choice' gr.Radio(value[1])
                ),
                inputs=model_choice, outputs=[default_model, import_method]
            )
            
            import_method.change(
                lambda choice: (
                    gr.update(visible=choice == "Path"), # Must same with 'import_methods' gr.Radio(value[0])
                    gr.update(visible=choice == "File")  # Must same with 'import_methods' gr.Radio(value[1])
                ),
                inputs=import_method, outputs=[custom_model_path, custom_model_file]
            )
            
            final_model = gr.Textbox(label="model", visible=debug_mode) # Only shows on debug mode
            for model_input in [default_model, custom_model_path, custom_model_file]:
                model_input.change(lambda x: x, inputs=model_input, outputs=final_model)
            
            gr.Markdown("## Set arguments")
            with gr.Row():
                with gr.Column():
                    conf_limit = gr.Slider(0.1, 1.0, 0.25, label="conf", info="Set Confidence score limit")
                    iou_limit = gr.Slider(0.1, 1.0, 0.7, label="iou",info="Set IOU limit")
                with gr.Column():
                    save_bool = gr.Checkbox(label="save",value=True, info="Save the output")
                    show_label = gr.Checkbox(label="show_labels", value=True, info="Show labels")
                    show_conf = gr.Checkbox(label="show_conf", value=True, info="Show confidence score") 
                    show_conf = gr.Checkbox(label="show_boxes", value=True, info="Show bounding box") 
                    save_name = gr.Textbox(label="name", value=f"{datetime.now().strftime('%y%m%d_%H%M')}", info="Save name")
                device = gr.Radio(["cpu", "cuda:0"], value="cpu", label="device")
                
            add_params = gr.Radio(['True', 'False'], value= 'False', label="Additional arguments", visible=True)
            default_params_code = gr.Code(lines=5, scale=2, language="yaml", visible=False) # Show default.yaml,  https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            
            add_params.change(get_file_content, inputs=gr.Textbox("default.yaml",visible=False), outputs=default_params_code)

            params_arg = gr.Textbox(label="", placeholder="Enter additional arguments, refer to the default.yaml \n Arguments must be passed as arg=val pairs, split by an equals = sign and delimited by spaces between pairs. Do not use -- argument prefixes or commas , between arguments.")
            
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
                root_dir = gr.Textbox(label="root_dir", value="/home", visible=True, info="Root directory for list of files")
                input_file_dir = gr.FileExplorer(
                    scale=1,
                    glob="*.mp4",
                    value=["themes/utils"],
                    file_count="single",
                    root_dir='/',
                    elem_id="file",
                    interactive=True,
                    every=1,
                    height=200
                )
                
            root_dir.change(
                    lambda x: gr.update(root_dir=x),
                    inputs=root_dir,
                    outputs=input_file_dir
            )
            input_file_dir.change(get_name, inputs=input_file_dir, outputs=input_file_path)
            
            final_params = gr.Textbox(label="Final Params", visible=debug_mode) # Only shows on debug mode
            for param_input in [final_model,input_file_path, conf_limit, iou_limit, save_bool, show_label, show_conf, device, save_name, params_arg]:
                update_params(param_input.value, param_input.label)
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
                inputs=[final_params,save_name],
                outputs=output,
                show_progress=True
            )
            
            stop_button.click(stop,
                              inputs=save_name,
                              outputs=output)
    
    return demo

demo = setup_interface()
demo.launch(share=True)