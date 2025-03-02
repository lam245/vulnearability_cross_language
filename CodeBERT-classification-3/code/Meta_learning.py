import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

def maml_train(model, optimizer, episodes=1000):
    for episode in range(episodes):
        # Sample task (vulnerability type)
        task_classes = sample_task_classes()
        
        # Sample support and query sets
        support_data, query_data = sample_data_for_task(task_classes)
        
        # Create a copy of the model for this task
        task_model = copy.deepcopy(model)
        task_optimizer = torch.optim.SGD(task_model.parameters(), lr=0.01)
        
        # Inner loop: Update task model on support set
        for _ in range(5):  # Few gradient steps
            loss, _ = task_model(support_data['input_ids'], labels=support_data['labels'])
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()
        
        # Outer loop: Evaluate on query set and update original model
        loss, _ = task_model(query_data['input_ids'], labels=query_data['labels'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()