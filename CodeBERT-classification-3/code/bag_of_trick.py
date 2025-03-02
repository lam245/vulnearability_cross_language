# coding=utf-8
"""
Training code for Advanced SOTA Few-Shot Learning Methods
for Code Vulnerability Classification
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
import multiprocessing
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaForSequenceClassification, RobertaTokenizer)

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = ' '.join(js['code'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['label'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


#################################
# Few-Shot Learning Implementations
#################################

class CLIPForCode(nn.Module):
    def __init__(self, encoder, config, projection_dim=256):
        super(CLIPForCode, self).__init__()
        self.encoder = encoder
        self.code_projection = nn.Linear(config.hidden_size, projection_dim)
        self.num_labels = config.num_labels
        self.vulnerability_embeddings = nn.Parameter(torch.randn(config.num_labels, projection_dim))
        
    def forward(self, input_ids, labels=None, temperature=0.07):
        # Encode code snippets
        outputs = self.encoder(input_ids)
        code_features = outputs[0][:, 0, :]  # CLS token
        code_embeddings = self.code_projection(code_features)
        code_embeddings = F.normalize(code_embeddings, dim=1)
        
        # Normalize vulnerability class embeddings
        vuln_embeddings = F.normalize(self.vulnerability_embeddings, dim=1)
        
        # Calculate similarity matrix
        logits = torch.matmul(code_embeddings, vuln_embeddings.T) / temperature
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
        return (loss, logits) if loss is not None else logits


class AdaptivePrototypicalNetwork(nn.Module):
    def __init__(self, encoder, config):
        super(AdaptivePrototypicalNetwork, self).__init__()
        self.encoder = encoder
        self.hidden_size = config.hidden_size
        self.cross_attention = nn.MultiheadAttention(self.hidden_size, 8)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.num_labels = config.num_labels
        
    def forward(self, support_ids, support_labels, query_ids, query_labels=None):
        # Encode support and query examples
        support_outputs = self.encoder(support_ids)
        support_emb = support_outputs[0][:, 0, :]  # CLS token
        
        query_outputs = self.encoder(query_ids)
        query_emb = query_outputs[0][:, 0, :]
        
        # Calculate class prototypes
        prototypes = []
        for c in range(self.num_labels):
            if c in support_labels:
                class_mask = (support_labels == c)
                class_embeddings = support_emb[class_mask]
                if len(class_embeddings) > 0:
                    prototype = class_embeddings.mean(0)
                    prototypes.append(prototype)
                else:
                    # Fallback if no examples found
                    prototypes.append(torch.zeros_like(support_emb[0]))
            else:
                # Handle missing classes in support set
                prototypes.append(torch.zeros_like(support_emb[0]))
        
        prototypes = torch.stack(prototypes)
        
        # Cross-attention between query and prototypes
        # Reshape for attention
        query_emb = query_emb.unsqueeze(1)  # [batch_size, 1, dim]
        prototypes_expanded = prototypes.unsqueeze(0).expand(query_emb.size(0), -1, -1)  # [batch_size, n_classes, dim]
        
        # Apply cross-attention
        attn_output, _ = self.cross_attention(
            query_emb.transpose(0, 1),
            prototypes_expanded.transpose(0, 1),
            prototypes_expanded.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)
        
        # Residual connection and layer norm
        attn_output = self.layernorm(query_emb + attn_output)
        
        # Calculate distances to adapted prototypes
        distances = torch.cdist(attn_output.squeeze(1), prototypes)
        
        # Convert distances to logits (negative distance)
        logits = -distances
        
        loss = None
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)
            
        return (loss, logits) if loss is not None else logits


class TeacherStudentMetaLearner(nn.Module):
    def __init__(self, teacher_model, student_model):
        super(TeacherStudentMetaLearner, self).__init__()
        self.teacher = teacher_model
        self.student = student_model
        
    def forward(self, support_ids, support_labels, query_ids, query_labels=None, unlabeled_ids=None):
        # Get teacher predictions on unlabeled data
        teacher_logits = None
        if unlabeled_ids is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(unlabeled_ids)
                teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # Regular few-shot learning on support set
        student_loss, student_logits = self.student(support_ids, support_labels, query_ids, query_labels)
        
        # Get student predictions on unlabeled data
        student_unlabeled_logits = None
        if unlabeled_ids is not None:
            student_unlabeled_logits = self.student(support_ids, support_labels, unlabeled_ids)
            
        # If we're training, compute the combined loss
        if query_labels is not None and unlabeled_ids is not None:
            # Consistency loss on unlabeled data (KL divergence)
            consistency_loss = F.kl_div(
                F.log_softmax(student_unlabeled_logits, dim=1),
                teacher_probs,
                reduction='batchmean'
            )
            
            # Combined loss (alpha is a hyperparameter)
            alpha = 0.5
            combined_loss = student_loss + alpha * consistency_loss
            return combined_loss, student_logits
        
        return (student_loss, student_logits) if query_labels is not None else student_logits


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, adj):
        # x: Node features [batch_size, num_nodes, in_features]
        # adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        # Matrix multiplication: XW
        support = torch.matmul(x, self.weight)
        
        # Graph convolution: AXW
        output = torch.matmul(adj, support)
        
        return output


class CodeGNN(nn.Module):
    def __init__(self, encoder, config, hidden_dim=256):
        super(CodeGNN, self).__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_labels = config.num_labels
        
        self.gnn_layers = nn.ModuleList([
            GraphConvolution(config.hidden_size, hidden_dim),
            GraphConvolution(hidden_dim, hidden_dim)
        ])
        self.prototype_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def encode_code(self, code_ids, adjacency_matrix):
        # Get token embeddings
        outputs = self.encoder(code_ids)
        token_embeddings = outputs[0]
        
        # Apply GNN layers
        x = token_embeddings
        for layer in self.gnn_layers:
            x = F.relu(layer(x, adjacency_matrix))
            
        # Global pooling
        code_embedding = x.mean(dim=1)
        return code_embedding
        
    def forward(self, support_ids, support_adj, support_labels, query_ids, query_adj, query_labels=None):
        # Encode support and query examples
        support_embeddings = self.encode_code(support_ids, support_adj)
        query_embeddings = self.encode_code(query_ids, query_adj)
        
        # Calculate class prototypes
        prototypes = []
        for c in range(self.num_labels):
            if c in support_labels:
                class_mask = (support_labels == c)
                class_embeddings = support_embeddings[class_mask]
                if len(class_embeddings) > 0:
                    prototype = class_embeddings.mean(0)
                    prototypes.append(prototype)
                else:
                    # Fallback if no examples found
                    prototypes.append(torch.zeros_like(support_embeddings[0]))
            else:
                # Handle missing classes in support set
                prototypes.append(torch.zeros_like(support_embeddings[0]))
                
        prototypes = torch.stack(prototypes)
        
        # Project prototypes
        prototypes = self.prototype_layer(prototypes)
        
        # Calculate distances
        distances = torch.cdist(query_embeddings, prototypes)
        
        # Convert distances to logits
        logits = -distances
        
        loss = None
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)
            
        return (loss, logits) if loss is not None else logits


class PrototypicalPartsNetwork(nn.Module):
    def __init__(self, encoder, config, num_parts=5):
        super(PrototypicalPartsNetwork, self).__init__()
        self.encoder = encoder
        self.num_parts = num_parts
        self.hidden_dim = config.hidden_size
        self.num_labels = config.num_labels
        
        # Part attention
        self.parts_attention = nn.ModuleList([
            nn.Linear(self.hidden_dim, 1) for _ in range(num_parts)
        ])
        
        # Part projection
        self.parts_projection = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim // num_parts) for _ in range(num_parts)
        ])
        
    def extract_parts(self, features):
        # features: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len = features.shape[0], features.shape[1]
        
        parts = []
        for part_idx in range(self.num_parts):
            # Calculate attention scores
            attention = self.parts_attention[part_idx](features).squeeze(-1)
            attention = F.softmax(attention, dim=1).unsqueeze(-1)
            
            # Apply attention
            attended_features = (features * attention).sum(dim=1)
            
            # Project to part-specific space
            projected = self.parts_projection[part_idx](attended_features)
            parts.append(projected)
            
        # Concatenate all parts
        return torch.cat(parts, dim=1)
        
    def forward(self, support_ids, support_labels, query_ids, query_labels=None):
        # Get contextualized token representations
        support_outputs = self.encoder(support_ids)
        support_features = support_outputs[0]
        
        query_outputs = self.encoder(query_ids)
        query_features = query_outputs[0]
        
        # Extract part-based representations
        support_parts = self.extract_parts(support_features)
        query_parts = self.extract_parts(query_features)
        
        # Calculate class prototypes
        prototypes = []
        for c in range(self.num_labels):
            if c in support_labels:
                class_mask = (support_labels == c)
                class_embeddings = support_parts[class_mask]
                if len(class_embeddings) > 0:
                    prototype = class_embeddings.mean(0)
                    prototypes.append(prototype)
                else:
                    # Fallback if no examples found
                    prototypes.append(torch.zeros_like(support_parts[0]))
            else:
                # Handle missing classes in support set
                prototypes.append(torch.zeros_like(support_parts[0]))
                
        prototypes = torch.stack(prototypes)
        
        # Calculate distances
        distances = torch.cdist(query_parts, prototypes)
        
        # Convert distances to logits
        logits = -distances
        
        loss = None
        if query_labels is not None:
            loss = F.cross_entropy(logits, query_labels)
            
        return (loss, logits) if loss is not None else logits


#################################
# Main Training Functions
#################################

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_few_shot_batches(dataset, n_way, n_shot, n_query, n_tasks):
    """
    Create batches for few-shot learning
    
    Args:
        dataset: Dataset containing code examples and labels
        n_way: Number of classes per task
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        n_tasks: Number of tasks to generate
        
    Returns:
        list of (support_set, query_set) tuples
    """
    examples_per_label = {}
    
    # Group examples by label
    for i, (input_ids, label) in enumerate(dataset):
        label_idx = label.item()
        if label_idx not in examples_per_label:
            examples_per_label[label_idx] = []
        examples_per_label[label_idx].append((input_ids, label))
    
    # Filter labels with too few examples
    valid_labels = [label for label, examples in examples_per_label.items() 
                   if len(examples) >= n_shot + n_query]
    
    batches = []
    for _ in range(n_tasks):
        # Sample n_way classes
        if len(valid_labels) < n_way:
            selected_labels = valid_labels  # Use all available if not enough
        else:
            selected_labels = random.sample(valid_labels, n_way)
        
        support_inputs = []
        support_labels = []
        query_inputs = []
        query_labels = []
        
        for class_idx, label in enumerate(selected_labels):
            # Get examples for this label
            examples = examples_per_label[label]
            
            # Sample support and query examples
            selected_examples = random.sample(examples, n_shot + n_query)
            support_examples = selected_examples[:n_shot]
            query_examples = selected_examples[n_shot:n_shot + n_query]
            
            # Add to support set (with remapped labels to range 0...n_way-1)
            for input_ids, _ in support_examples:
                support_inputs.append(input_ids)
                support_labels.append(class_idx)
                
            # Add to query set
            for input_ids, _ in query_examples:
                query_inputs.append(input_ids)
                query_labels.append(class_idx)
        
        # Convert to tensors
        support_inputs = torch.stack(support_inputs)
        support_labels = torch.tensor(support_labels)
        query_inputs = torch.stack(query_inputs)
        query_labels = torch.tensor(query_labels)
        
        batches.append((support_inputs, support_labels, query_inputs, query_labels))
    
    return batches


def train_few_shot(args, train_dataset, model, tokenizer, model_type):
    """Train using few-shot learning approach"""
    
    # Create few-shot batches
    few_shot_batches = create_few_shot_batches(
        train_dataset, 
        args.n_way, 
        args.n_shot, 
        args.n_query,
        args.n_tasks
    )
    
    # Prepare optimizer and schedule
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(few_shot_batches) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=max_steps*0.1,
        num_training_steps=max_steps
    )

    # Train!
    logger.info("***** Running few-shot training *****")
    logger.info(f"  Model type = {model_type}")
    logger.info(f"  N-way = {args.n_way}")
    logger.info(f"  N-shot = {args.n_shot}")
    logger.info(f"  N-query = {args.n_query}")
    logger.info(f"  N-tasks = {args.n_tasks}")
    logger.info(f"  Num episodes = {len(few_shot_batches)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {max_steps}")
    
    best_acc = 0.0
    model.zero_grad()
    
    for epoch in range(args.num_train_epochs):
        # Shuffle batches for each epoch
        random.shuffle(few_shot_batches)
        
        episode_losses = []
        episode_accuracies = []
        
        for episode_idx, (support_inputs, support_labels, query_inputs, query_labels) in enumerate(
            tqdm(few_shot_batches, desc=f"Epoch {epoch}")
        ):
            model.train()
            
            # Move tensors to device
            support_inputs = support_inputs.to(args.device)
            support_labels = support_labels.to(args.device)
            query_inputs = query_inputs.to(args.device)
            query_labels = query_labels.to(args.device)
            
            # Forward pass (different for each model type)
            if model_type == 'clip':
                loss, logits = model(query_inputs, query_labels)
            elif model_type == 'proto' or model_type == 'parts':
                loss, logits = model(support_inputs, support_labels, query_inputs, query_labels)
            elif model_type == 'gnn':
                # For GNN, we need adjacency matrices (simplified here)
                batch_size = support_inputs.size(0)
                seq_len = support_inputs.size(1)
                # Simple adjacency: connect each token to next token (chain)
                support_adj = torch.zeros(batch_size, seq_len, seq_len, device=args.device)
                for i in range(seq_len-1):
                    support_adj[:, i, i+1] = 1
                    support_adj[:, i+1, i] = 1  # Make it bidirectional
                
                query_adj = torch.zeros(query_inputs.size(0), seq_len, seq_len, device=args.device)
                for i in range(seq_len-1):
                    query_adj[:, i, i+1] = 1
                    query_adj[:, i+1, i] = 1
                
                loss, logits = model(support_inputs, support_adj, support_labels, 
                                    query_inputs, query_adj, query_labels)
            elif model_type == 'teacher_student':
                # Use support set as unlabeled data for simplicity
                # In practice, you'd use a separate unlabeled dataset
                loss, logits = model(support_inputs, support_labels, query_inputs, query_labels, support_inputs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            # Calculate accuracy for this episode
            preds = logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean().item()
            
            episode_losses.append(loss.item())
            episode_accuracies.append(acc)
            
            if (episode_idx + 1) % 10 == 0:
                logger.info(f"Epoch {epoch}, Episode {episode_idx+1}, Loss: {np.mean(episode_losses):.4f}, "
                          f"Accuracy: {np.mean(episode_accuracies):.4f}")
        
        # Evaluate on validation set
        results = evaluate_few_shot(args, model, tokenizer, model_type)
        logger.info(f"***** Eval results at epoch {epoch} *****")
        for key, value in results.items():
            logger.info(f"  {key} = {round(value, 4)}")
            
        # Save model checkpoint if we have a new best model
        if results['eval_acc'] > best_acc:
            best_acc = results['eval_acc']
            logger.info("  " + "*" * 20)
            logger.info(f"  Best acc: {round(best_acc, 4)}")
            logger.info("  " + "*" * 20)
            
            checkpoint_prefix = f'checkpoint-best-acc-{model_type}'
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, 'model.bin')
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info(f"Saving model checkpoint to {output_dir}")


def evaluate_few_shot(args, model, tokenizer, model_type):
    """Evaluate using few-shot learning approach"""
    
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    
    # Create few-shot batches for evaluation
    few_shot_batches = create_few_shot_batches(
        eval_dataset, 
        args.n_way, 
        args.n_shot, 
        args.n_query,
        args.n_eval_tasks  # Use a different number of tasks for evaluation
    )
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Eval!
    logger.info("***** Running few-shot evaluation *****")
    logger.info(f"  N-way = {args.n_way}")
    logger.info(f"  N-shot = {args.n_shot}")
    logger.info(f"  Num episodes = {len(few_shot_batches)}")
    
    episode_losses = []
    episode_accuracies = []
    
    model.eval()
    
    for episode_idx, (support_inputs, support_labels, query_inputs, query_labels) in enumerate(
        tqdm(few_shot_batches, desc="Evaluating")
    ):
        # Move tensors to device
        support_inputs = support_inputs.to(args.device)
        support_labels = support_labels.to(args.device)
        query_inputs = query_inputs.to(args.device)
        query_labels = query_labels.to(args.device)
        
        with torch.no_grad():
            # Forward pass (different for each model type)
            if model_type == 'clip':
                loss, logits = model(query_inputs, query_labels)
            elif model_type == 'proto' or model_type == 'parts':
                loss, logits = model(support_inputs, support_labels, query_inputs, query_labels)
            elif model_type == 'gnn':
                # For GNN, we need adjacency matrices (simplified here)
                batch_size = support_inputs.size(0)
                seq_len = support_inputs.size(1)
                # Simple adjacency: connect each token to next token (chain)
                support_adj = torch.zeros(batch_size, seq_len, seq_len, device=args.device)
                for i in range(seq_len-1):
                    support_adj[:, i, i+1] = 1
                    support_adj[:, i+1, i] = 1  # Make it bidirectional
                
                query_adj = torch.zeros(query_inputs.size(0), seq_len, seq_len, device=args.device)
                for i in range(seq_len-1):
                    query_adj[:, i, i+1] = 1
                    query_adj[:, i+1, i] = 1
                
                loss, logits = model(support_inputs, support_adj, support_labels, 
                                    query_inputs, query_adj, query_labels)
            elif model_type == 'teacher_student':
                # Use support set as unlabeled data for simplicity
                loss, logits = model(support_inputs, support_labels, query_inputs, query_labels, support_inputs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Calculate accuracy for this episode
        preds = logits.argmax(dim=1)
        acc = (preds == query_labels).float().mean().item()
        
        episode_losses.append(loss.item())
        episode_accuracies.append(acc)
    
    eval_loss = np.mean(episode_losses)
    eval_acc = np.mean(episode_accuracies)
    
    result = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
    }
    
    return result


def test_few_shot(args, model, tokenizer, model_type):
    """Test using few-shot learning approach"""
    
    # Load the dataset
    test_dataset = TextDataset(tokenizer, args, args.test_data_file)
    
    # Load the train dataset to use as support set
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    
    # Create few-shot batches for testing
    # We'll use a single large batch with all test examples as queries
    
    # Group train examples by label to create support set
    examples_per_label = {}
    for i, (input_ids, label) in enumerate(train_dataset):
        label_idx = label.item()
        if label_idx not in examples_per_label:
            examples_per_label[label_idx] = []
        examples_per_label[label_idx].append((input_ids, label))
    
    # Select n_shot examples per class for support set
    support_inputs = []
    support_labels = []
    
    all_labels = sorted(examples_per_label.keys())
    for label in all_labels:
        examples = examples_per_label[label]
        selected = examples[:args.n_shot] if len(examples) >= args.n_shot else examples
        
        for input_ids, _ in selected:
            support_inputs.append(input_ids)
            support_labels.append(label)
    
    # Convert to tensors
    support_inputs = torch.stack(support_inputs)
    support_labels = torch.tensor(support_labels)
    
    # Create a dataloader for test examples
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
    
    # Test!
    logger.info("***** Running few-shot testing *****")
    logger.info(f"  Model type = {model_type}")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    
    all_preds = []
    all_labels = []
    
    model.eval()
    
    # Move support set to device
    support_inputs = support_inputs.to(args.device)
    support_labels = support_labels.to(args.device)
    
    # For GNN model, create adjacency matrix for support set
    if model_type == 'gnn':
        seq_len = support_inputs.size(1)
        support_adj = torch.zeros(support_inputs.size(0), seq_len, seq_len, device=args.device)
        for i in range(seq_len-1):
            support_adj[:, i, i+1] = 1
            support_adj[:, i+1, i] = 1  # Make it bidirectional
    
    for batch in tqdm(test_dataloader, desc="Testing"):
        query_inputs, query_labels = batch
        query_inputs = query_inputs.to(args.device)
        query_labels = query_labels.to(args.device)
        
        with torch.no_grad():
            # Forward pass (different for each model type)
            if model_type == 'clip':
                logits = model(query_inputs)
            elif model_type == 'proto' or model_type == 'parts':
                logits = model(support_inputs, support_labels, query_inputs)
            elif model_type == 'gnn':
                # Create adjacency matrix for query examples
                query_adj = torch.zeros(query_inputs.size(0), seq_len, seq_len, device=args.device)
                for i in range(seq_len-1):
                    query_adj[:, i, i+1] = 1
                    query_adj[:, i+1, i] = 1
                
                logits = model(support_inputs, support_adj, support_labels, 
                              query_inputs, query_adj)
            elif model_type == 'teacher_student':
                logits = model(support_inputs, support_labels, query_inputs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(query_labels.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    
    # Calculate precision, recall, and F1 score (assuming binary classification for simplicity)
    if len(np.unique(all_labels)) == 2:
        # Convert to binary case
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        # For multi-class, report macro average
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    # Save predictions
    output_test_file = os.path.join(args.output_dir, f"test_predictions_{model_type}.txt")
    with open(output_test_file, "w") as writer:
        writer.write("Index\tPrediction\tLabel\n")
        for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
            writer.write(f"{i}\t{pred}\t{label}\n")
    
    result = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    logger.info("***** Test results *****")
    for key, value in result.items():
        logger.info(f"  {key} = {round(value, 4)}")
    
    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a json file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a json file).")
    
    ## Other parameters
    parser.add_argument("--model_type", default="proto", type=str,
                        help="The model architecture to be fine-tuned: clip, proto, gnn, teacher_student, parts")
    parser.add_argument("--pretrained_model", default="microsoft/codebert-base", type=str,
                        help="The pre-trained model to use.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--n_way", default=2, type=int,
                        help="Number of classes per few-shot task.")
    parser.add_argument("--n_shot", default=5, type=int,
                        help="Number of shots (support examples per class).")
    parser.add_argument("--n_query", default=15, type=int,
                        help="Number of query examples per class.")
    parser.add_argument("--n_tasks", default=100, type=int,
                        help="Number of few-shot tasks for training.")
    parser.add_argument("--n_eval_tasks", default=50, type=int,
                        help="Number of few-shot tasks for evaluation.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--no_cuda', action='store_true',
                        help="Avoid using CUDA when available")
    
    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    # Set seed
    set_seed(args.seed)

    # Load pre-trained model and tokenizer
    config = RobertaConfig.from_pretrained(args.pretrained_model)
    config.num_labels = args.n_way  # Set number of labels for few-shot learning
    
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
    encoder = RobertaModel.from_pretrained(args.pretrained_model)
    
    logger.info(f"Training/evaluation parameters: {args}")
    
    # Initialize model based on model_type
    if args.model_type == 'clip':
        model = CLIPForCode(encoder, config)
    elif args.model_type == 'proto':
        model = AdaptivePrototypicalNetwork(encoder, config)
    elif args.model_type == 'gnn':
        model = CodeGNN(encoder, config)
    elif args.model_type == 'teacher_student':
        # For teacher-student, we need to initialize both models
        teacher_model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model, config=config)
        student_model = AdaptivePrototypicalNetwork(encoder, config)
        model = TeacherStudentMetaLearner(teacher_model, student_model)
    elif args.model_type == 'parts':
        model = PrototypicalPartsNetwork(encoder, config)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    model.to(args.device)
    
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train_few_shot(args, train_dataset, model, tokenizer, args.model_type)
    
    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = f'checkpoint-best-acc-{args.model_type}'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        output_dir = os.path.join(output_dir, 'model.bin')
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate_few_shot(args, model, tokenizer, args.model_type)
        results.update(result)
    
    # Testing
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-acc-{args.model_type}'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        output_dir = os.path.join(output_dir, 'model.bin')
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = test_few_shot(args, model, tokenizer, args.model_type)
        results.update(result)
    
    return results


if __name__ == "__main__":
    main()