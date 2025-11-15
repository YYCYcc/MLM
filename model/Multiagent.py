
import torch
import torch.nn as nn

class MultiAgentFusion(nn.Module):
    def __init__(self, models, weights=None):
        super(MultiAgentFusion, self).__init__()
        self.models = models
  
        self.weights = weights if weights else [1.0] * len(models)
    
    def forward(self, *inputs):
        outputs = []
        
        # 获取每个模型的输出
        for model in self.models:
            output = model(*inputs)
            outputs.append(output)
        
        # 对输出进行加权平均
        weighted_output = sum(w * o for w, o in zip(self.weights, outputs))
        # 将加权输出通过 Softmax 转换为概率分布
        probabilities = torch.softmax(weighted_output, dim=-1)
        return probabilities


