class PrototypicalPartsNetwork(nn.Module):
    def __init__(self, encoder, num_parts=5, hidden_dim=768):
        super(PrototypicalPartsNetwork, self).__init__()
        self.encoder = encoder
        self.num_parts = num_parts
        
        # Part attention
        self.parts_attention = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_parts)
        ])
        
        # Part projection
        self.parts_projection = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // num_parts) for _ in range(num_parts)
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
        
    def forward(self, support_ids, support_labels, query_ids):
        # Get contextualized token representations
        support_features = self.encoder(support_ids)[0]
        query_features = self.encoder(query_ids)[0]
        
        # Extract part-based representations
        support_parts = self.extract_parts(support_features)
        query_parts = self.extract_parts(query_features)
        
        # Calculate class prototypes
        classes = torch.unique(support_labels)
        prototypes = torch.stack([
            support_parts[support_labels == c].mean(0)
            for c in classes
        ])
        
        # Calculate distances
        distances = torch.cdist(query_parts, prototypes)
        
        # Convert distances to logits
        return -distances