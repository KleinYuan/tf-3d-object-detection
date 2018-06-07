from models import base_server
from configs import configs


input_data = [0, 0, 0, 0]
srv = base_server.BaseServer(model_fp=configs.model_fp,
                             input_tensor_names=configs.input_tensor_names,
                             output_tensor_names=configs.output_tensor_names,
                             device=configs.device)
prediction = srv.inference(data=[input_data])
