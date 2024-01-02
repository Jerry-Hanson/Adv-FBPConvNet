from torchsummary import summary 
from thop import profile 



def model_summary(model, input_size, device = 'cpu'):
    return summary(model, input_size = input_size, device = device)


def model_profile(model, dummy_input):
    # 返回的单位分别是 G 和 M
    flops, params = profile(model, (dummy_input, ))
    return round(flops / (10 ** 9), 2), round(params / (10 ** 6), 2)