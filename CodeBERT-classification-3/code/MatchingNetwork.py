class MatchingNetwork(nn.Module):
    def __init__(self, encoder, embedding_dim=768):
        super(MatchingNetwork, self).__init__()
        self.encoder = encoder
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        
    def forward(self, support_ids, support_labels, query_ids):
        # Encode support and query examples
        support_features = self.encoder(support_ids, attention_mask=support_ids.ne(1))[0][:, 0, :]
        query_features = self.encoder(query_ids, attention_mask=query_ids.ne(1))[0][:, 0, :]
        
        # Calculate attention scores
        query_features = query_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        support_features = support_features.unsqueeze(0).expand(
            query_features.size(0), -1, -1)  # [batch_size, n_support, embedding_dim]
            
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            query_features, support_features, dim=2)  # [batch_size, n_support]
            
        # Convert to probabilities
        attention = F.softmax(similarity, dim=1)
        
        # Get one-hot labels
        n_classes = len(torch.unique(support_labels))
        y_one_hot = F.one_hot(support_labels, n_classes).float()
        
        # Weight labels by attention
        predictions = torch.matmul(attention, y_one_hot)
        
        return predictions