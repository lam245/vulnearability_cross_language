class CLIPForCode(nn.Module):
    def __init__(self, encoder, projection_dim=256):
        super(CLIPForCode, self).__init__()
        self.encoder = encoder
        self.code_projection = nn.Linear(768, projection_dim)
        self.vulnerability_embeddings = nn.Parameter(torch.randn(num_classes, projection_dim))
        
    def forward(self, code_ids, temperature=0.07):
        # Encode code snippets
        code_features = self.encoder(code_ids)[0][:, 0, :]  # CLS token
        code_embeddings = self.code_projection(code_features)
        code_embeddings = F.normalize(code_embeddings, dim=1)
        
        # Normalize vulnerability class embeddings
        vuln_embeddings = F.normalize(self.vulnerability_embeddings, dim=1)
        
        # Calculate similarity matrix
        logits = torch.matmul(code_embeddings, vuln_embeddings.T) / temperature
        return logits